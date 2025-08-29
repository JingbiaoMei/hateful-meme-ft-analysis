#!/usr/bin/env python3
"""
Hateful Memes Classification Inference using vLLM backend.

This script performs inference on hateful memes datasets using vLLM's offline inference engine
for maximum throughput and speed compared to the HuggingFace sequential generation approach.

Features:
- Uses vLLM batch processing for faster inference
- Supports multiple datasets (FB, MAMI, etc.)
- Evaluates on multiple data splits
- Logs results to wandb and CSV files
- Supports LoRA adapters
- Multimodal support for Qwen2.5-VL models

Usage:
    python qwen2_hm_inference_vllm.py --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
                                      --processor_path "Qwen/Qwen2.5-VL-7B-Instruct" \
                                      --data_split "dev_seen test_seen" \
                                      --dataset "FB" \
                                      --batch_size 32
"""

import sys
sys.path.append('./src')

# Set tokenizers parallelism before importing any HuggingFace libraries
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb
import argparse
import functools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

from transformers import AutoTokenizer
from dataset import get_Dataloader
import csv
import math
import gc

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def exact_match_reward(raw_output, solution, **kwargs):
    """Calculate exact match reward between prediction and ground truth."""
    answer_tag_pattern = r'<answer>(.*?)</answer>'
 
    reward = 0.0

    # Extract answer from solution if it has think/answer tags 
    sol_match = re.search(answer_tag_pattern, solution, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()
    
    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else raw_output.strip()

    ground_truth = ground_truth.replace(' ', '').replace('_', '').lower()
    student_answer = student_answer.replace(' ', '').replace('_', '').lower()
    
    # Compare the extracted answers
    if ground_truth in student_answer or student_answer in ground_truth:
        reward = 1.0
    if student_answer == "" or student_answer == " ":
        reward = 0.0

    return reward


def write_to_csv(log_path, metrics):
    """Write evaluation results to a CSV file."""
    with open(log_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["split", "acc", "precision", "recall", "f1", "reward"])
        for split, values in metrics.items():
            values = [round(value, 4) for value in values]
            writer.writerow([split] + values)


def evaluate_predictions(y_true, y_pred, rewards=None):
    """Calculate evaluation metrics."""
    metrics = []
    metrics.append(accuracy_score(y_true, y_pred))
    metrics.append(precision_score(y_true, y_pred, zero_division=0))
    metrics.append(recall_score(y_true, y_pred, zero_division=0))
    metrics.append(f1_score(y_true, y_pred, zero_division=0))
    if rewards is not None:
        metrics.append(np.mean(rewards))
    else:
        metrics.append(0.0)
    return metrics


class VLLMInferenceEngine:
    """vLLM-based inference engine for hateful meme classification."""
    
    def __init__(self, 
                 model_path: str,
                 base_model_path: str = None,
                 processor_path: str = None,
                 adapter_name_or_path: str = None,
                 tensor_parallel_size: int = None,
                 pipeline_parallel_size: int = 1,
                 max_model_len: int = 8192,
                 max_pixels: int = None,
                 dtype: str = "bfloat16",
                 trust_remote_code: bool = True,
                 **vllm_kwargs):
        """
        Initialize the vLLM inference engine.
        
        Args:
            model_path: Path to the model or model name
            base_model_path: Path to base model (for LoRA adapters)
            processor_path: Path to processor (for tokenizer)
            adapter_name_or_path: Path to LoRA adapter
            tensor_parallel_size: Number of GPUs for tensor parallelism
            pipeline_parallel_size: Number of GPUs for pipeline parallelism
            max_model_len: Maximum model context length
            max_pixels: Maximum pixels for image processing
            dtype: Model data type
            trust_remote_code: Whether to trust remote code
            **vllm_kwargs: Additional vLLM engine arguments
        """
        if not HAS_VLLM:
            raise ImportError("vLLM is required for this script. Please install it with: pip install vllm")
        
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.processor_path = processor_path or model_path
        self.adapter_name_or_path = adapter_name_or_path
        self.max_pixels = max_pixels
        
        # Determine GPU configuration
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        logger.info(f"Initializing vLLM engine...")
        logger.info(f"Model: {model_path}")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Adapter: {adapter_name_or_path}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Pipeline parallel size: {pipeline_parallel_size}")
        logger.info(f"Max model length: {max_model_len}")
        
        # Determine which model to load
        if base_model_path and adapter_name_or_path:
            # Use base model for LoRA
            model_name_or_path = base_model_path
            enable_lora = True
        else:
            # Use the main model path
            model_name_or_path = model_path
            enable_lora = False
        
        # Initialize vLLM engine
        engine_args = {
            "model": model_name_or_path,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "disable_log_stats": True,
            "enable_lora": enable_lora,
            "limit_mm_per_prompt": {"image": 1, "video": 1, "audio": 1},  # Each request has only 1 image
            **vllm_kwargs
        }
        
        self.llm = LLM(**engine_args)
        
        # Initialize tokenizer for chat template processing
        logger.info("Loading tokenizer for chat template processing...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.processor_path, 
            trust_remote_code=trust_remote_code
        )
        
        # Initialize sampling parameters for greedy decoding (like the original)
        self.sampling_params = SamplingParams(
            repetition_penalty=1.0,
            temperature=0.0,  # Greedy decoding
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
            skip_special_tokens=True,
        )
        
        # Initialize LoRA request if adapter is provided
        self.lora_request = None
        if adapter_name_or_path:
            # Handle multiple LoRA adapters separated by ":"
            if ":" in adapter_name_or_path:
                # For now, use the first adapter - vLLM might not support multiple LoRAs simultaneously
                first_adapter = adapter_name_or_path.split(":")[0]
                logger.warning(f"Multiple LoRA adapters detected. Using only the first one: {first_adapter}")
                self.lora_request = LoRARequest("adapter_0", 1, first_adapter)
            else:
                self.lora_request = LoRARequest("adapter_0", 1, adapter_name_or_path)
        
        logger.info("vLLM engine initialized successfully")
    
    def _prepare_conversation_format(self, prompt_text: str, image: Image.Image) -> str:
        """
        Prepare conversation format for vLLM with Qwen2.5-VL using chat template.
        This matches exactly the working implementation in generate_dpo_data_vllm.py
        
        Args:
            prompt_text: The text prompt
            image: PIL Image object
            
        Returns:
            Formatted prompt string ready for vLLM
        """
        # Remove the <image> token from the prompt template since we'll handle it through messages
        text_prompt = prompt_text.replace("<image>", "").strip()
        
        if image is not None:
            # Create conversation messages with image and text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        else:
            # Text-only conversation
            messages = [
                {
                    "role": "user",
                    "content": text_prompt
                }
            ]
        
        # Apply chat template - this is crucial for vLLM multimodal support
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return formatted_prompt
    
    def _prepare_multimodal_data(self, image: Image.Image) -> dict:
        """
        Prepare multimodal data for vLLM input.
        This matches exactly the working implementation in generate_dpo_data_vllm.py
        """
        if image is None:
            return None
        
        # For vLLM with Qwen2.5-VL, we need to prepare the image in the expected format
        return {
            "image": image  # vLLM expects a single PIL image for Qwen2.5-VL
        }
    
    def load_image_from_dataloader(self, img_tensor):
        """
        Convert dataloader image tensor or path to PIL Image.
        """
        try:
            # Handle string paths (this is what we're getting from the dataset)
            if isinstance(img_tensor, str):
                if os.path.exists(img_tensor):
                    return Image.open(img_tensor).convert('RGB')
                else:
                    logging.error(f"Image path does not exist: {img_tensor}")
                    return None
            # Convert tensor image to PIL if necessary
            elif torch.is_tensor(img_tensor):
                # Handle different tensor formats
                if img_tensor.dim() == 3:  # CHW format
                    img_tensor = img_tensor.permute(1, 2, 0)  # Convert to HWC
                elif img_tensor.dim() == 4:  # BCHW format - take first image
                    img_tensor = img_tensor[0].permute(1, 2, 0)  # Convert to HWC
                
                img_np = img_tensor.cpu().numpy()
                
                # Normalize to 0-255 if needed
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:  # Normalized to [0,1]
                        img_np = (img_np * 255).astype(np.uint8)
                    else:  # Already in [0,255] range
                        img_np = img_np.astype(np.uint8)
                
                img = Image.fromarray(img_np)
                return img
            elif isinstance(img_tensor, Image.Image):
                return img_tensor
            else:
                logging.error(f"Unsupported image type: {type(img_tensor)}")
                return None
        except Exception as e:
            logging.error(f"Error converting image: {e}")
            return None
    
    def run_inference_on_batch(self, batch, query: str):
        """
        Process a batch of data using vLLM.
        Uses the exact same approach as the working generate_dpo_data_vllm.py script.
        
        Args:
            batch: Batch of (images, texts, labels, image_ids)
            query: Query template to use
            
        Returns:
            Tuple of (batch_results, batch_text_preds, batch_labels, batch_rewards, batch_ids)
        """
        images, texts, labels, image_ids = batch
        
        # Process each image individually to avoid vLLM multimodal batching issues
        batch_results = []
        batch_text_preds = []
        batch_labels = []
        batch_rewards = []
        processed_ids = []
        
        question = "This is an image with \"{text}\" written on it." + query
        
        for idx, (img_tensor, text, img_id, label) in enumerate(zip(images, texts, image_ids, labels)):
            try:
                formatted_question = question.format(text=text)
                
                # Convert dataloader tensor to PIL Image using the new method
                image = self.load_image_from_dataloader(img_tensor)
                if image is None:
                    logger.error(f"Failed to convert image tensor for {img_id}")
                    continue
                
                # Use the working chat template approach from generate_dpo_data_vllm.py
                formatted_prompt = self._prepare_conversation_format(f"<image>{formatted_question}", image)
                
                # Prepare multimodal data using the working approach
                multi_modal_data = self._prepare_multimodal_data(image)
                
                # Create vLLM input format exactly like the working script
                if multi_modal_data is not None:
                    # For multimodal inputs, use the formatted prompt with multimodal data
                    vllm_input = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": multi_modal_data
                    }
                else:
                    # For text-only inputs, just use the formatted prompt
                    vllm_input = {
                        "prompt": formatted_prompt
                    }
                
                # Process this single image exactly like the working script
                try:
                    single_result = self.llm.generate([vllm_input], self.sampling_params, lora_request=self.lora_request)
                    response = single_result[0].outputs[0].text
                    
                    # Process the response
                    thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
                    thinking = thinking_match.group(1).strip() if thinking_match else ""
                    
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if not answer_match:
                        logger.warning(f"No answer tag found in response for image {img_id}: {response}")
                        continue
                        
                    answer = answer_match.group(1).strip().lower()
                    gt = 'yes' if int(label.item()) == 1 else 'no'
                    reward = exact_match_reward(response, gt)
                    
                    text_pred = 1 if "yes" in answer.lower() else 0

                    result = {
                        'image_id': img_id,
                        'true_label': int(label.item()),
                        'text_prediction': text_pred,
                        'raw_output': response,
                        'thinking': thinking,
                        'text_answer': answer,
                        'correct': text_pred == int(label.item()),
                        'reward': reward,
                    }
                    
                    batch_text_preds.append(text_pred)
                    batch_labels.append(label.item())
                    batch_results.append(result)
                    batch_rewards.append(reward)
                    processed_ids.append(img_id)
                    
                except Exception as e:
                    logger.error(f"Error during vLLM generation for image {img_id}: {e}")
                    continue
                
            except Exception as e:
                logger.error(f"Error preparing request for image {img_id}: {e}")
                continue
                
        return batch_results, batch_text_preds, batch_labels, batch_rewards, processed_ids


def process_split_vllm(inference_engine, dataloader, split_name, args):
    """
    Process a complete data split using vLLM.
    
    Args:
        inference_engine: VLLMInferenceEngine instance
        dataloader: DataLoader for the split
        split_name: Name of the split being processed
        args: Command line arguments
        
    Returns:
        Dictionary with results and metrics
    """
    logger.info(f"Processing {split_name} split using vLLM...")
    
    # Track predictions
    all_results = []
    all_text_predictions = []
    all_labels = []
    all_ids = []
    all_rewards = []
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
        # In debug mode, only process specified number of batches
        if args.debug and batch_idx >= args.debug_batches:
            logger.info(f"Debug mode: Stopping after {args.debug_batches} batches")
            break
        
        batch_results, batch_text_preds, batch_labels, batch_rewards, batch_ids = inference_engine.run_inference_on_batch(
            batch, args.query
        )
        
        all_results.extend(batch_results)
        all_text_predictions.extend(batch_text_preds)
        all_labels.extend(batch_labels)
        all_ids.extend(batch_ids)
        all_rewards.extend(batch_rewards)
        
        # Optional: Clear GPU memory periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate metrics if we have predictions
    if all_text_predictions:
        metrics = evaluate_predictions(all_labels, all_text_predictions, all_rewards)
        logger.info(f"{split_name} - Metrics: Acc={metrics[0]:.4f}, P={metrics[1]:.4f}, R={metrics[2]:.4f}, F1={metrics[3]:.4f}, Reward={metrics[4]:.4f}")
    else:
        metrics = [0.0, 0.0, 0.0, 0.0, 0.0]
        logger.warning(f"{split_name} - No valid predictions were collected")
    
    return {
        'metrics': metrics,
        'results': all_results,
        'ids': all_ids,
        'labels': all_labels,
        'predictions': all_text_predictions,
        'rewards': all_rewards,
    }


def main():
    parser = argparse.ArgumentParser(description='Hateful Memes Classification with Qwen2-VL using vLLM')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint or model name')
    parser.add_argument('--base_model_path', type=str, default=None, help='Path to base model checkpoint (for LoRA)')
    parser.add_argument('--processor_path', type=str, required=True, help='Path to processor/tokenizer')
    parser.add_argument('--adapter_name_or_path', type=str, default=None, help='Path to LoRA adapter')
    
    # Data arguments
    parser.add_argument('--data_split', type=str, default='dev_seen test_seen', help='Evaluate on which dataset split')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (smaller batches recommended for multimodal vLLM)')
    parser.add_argument('--dataset', type=str, default="FB", help='Dataset name')
    
    # Logging arguments
    parser.add_argument('--log_name', type=str, default="", help='Log name')
    parser.add_argument('--group_name', type=str, default="inference_classifier_vllm", help='Group name for wandb')
    parser.add_argument('--exp_name', type=str, default="")
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true', default=False, help='Run in debug mode (process limited batches)')
    parser.add_argument('--debug_batches', type=int, default=10, help='Number of batches to process in debug mode')
    
    # Query argument
    parser.add_argument('--query', type=str, 
                        default="Please identify if the meme is hateful or not.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n For the thinking process, consider both the image and any accompanying text. Evaluate the historical, cultural, and social context surrounding the meme. Analyze different interpretations, including potential humor or satirical elements versus actual hate content. Reflect on broader social and racial implications. Consider if the meme perpetuates harmful stereotypes or could be seen as targeting a particular group. Assess how various audiences might perceive the content. Weigh the intent behind the meme against its potential for harm. Synthesize your reasoning and decide if the meme qualifies as hateful.\n The answer should be Yes or No, if the meme is hateful than output <answer> Yes </answer>; if the meme is Benign than output <answer> No </answer>. The output format should be as follows:\n<think> ... </think> <answer> ... </answer>\n. Please strictly follow the format.", 
                        help='Query to pass to the model')
    
    # vLLM-specific arguments
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--pipeline_parallel_size', type=int, default=1, help='Number of GPUs for pipeline parallelism')
    parser.add_argument('--max_model_len', type=int, default=8192, help='Maximum model context length')
    parser.add_argument('--max_pixels', type=int, default=None, help='Maximum pixels for image processing')
    parser.add_argument('--dtype', type=str, default="bfloat16", help='Model data type')
    parser.add_argument('--trust_remote_code', action='store_true', help='Whether to trust remote code')
    
    args = parser.parse_args()
    
    # Setup debug mode
    if args.debug:
        logger.info(f"Debug mode enabled: Will process {args.debug_batches} batches per split")
        args.log_name += "_debug"
        args.group_name = "debug_vllm"
    
    # Setup logging paths
    model_name = os.path.basename(args.model_path)
    log_path = f"./logging/{args.exp_name}/{args.dataset}/{args.log_name}/"
    os.makedirs(log_path, exist_ok=True)
    log_path += args.log_name + ".csv"
    
    # Initialize wandb
    exp_name = f"{args.log_name}_{args.dataset}_vllm"
    tags = [args.dataset, model_name, "vllm"]
    if args.debug:
        tags.append("debug")
        
    run = wandb.init(
        name=exp_name,
        project="RFT-Inference-vLLM",
        config={
            "model": model_name,
            "dataset": args.dataset,
            "debug": args.debug,
            "backend": "vllm",
            "batch_size": args.batch_size,
            "tensor_parallel_size": args.tensor_parallel_size
        },
        group=args.group_name,
        tags=tags
    )
    
    # Initialize vLLM inference engine
    logger.info("Initializing vLLM inference engine...")
    
    # Handle LoRA adapters
    adapter_path = None
    if args.base_model_path and args.model_path != args.base_model_path:
        adapter_path = args.model_path
    elif args.adapter_name_or_path:
        adapter_path = args.adapter_name_or_path
    
    inference_engine = VLLMInferenceEngine(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        processor_path=args.processor_path,
        adapter_name_or_path=adapter_path,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=args.max_model_len,
        max_pixels=args.max_pixels,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code
    )
    
    # Define splits
    if args.dataset == "FB":
        splits = ["train", "dev_seen", "test_seen", "test_unseen"]
    else:
        splits = ["train", "val", "test"]
    
    # Get dataloaders
    logger.info("Loading dataloaders...")
    if args.dataset == "FB":
        train, dev_seen, test_seen, test_unseen = get_Dataloader(
            preprocess=None,
            batch_size=args.batch_size,
            train_batch_size=args.batch_size,
            num_workers=4,
            dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen, test_unseen]
    else:
        train, dev_seen, test_seen = get_Dataloader(
            preprocess=None,
            batch_size=args.batch_size,
            num_workers=4,
            dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen]
    
    # Run inference on specified splits
    metrics_dict = {}
    all_results = []
    
    for split in splits:
        if split not in args.data_split:
            continue
            
        logger.info(f"Running inference on {split} split")
        split_index = splits.index(split)
        dataloader = loader_list[split_index]
        
        # Process the split
        split_result = process_split_vllm(inference_engine, dataloader, split, args)
        
        # Store metrics and results
        metrics_dict[split] = split_result['metrics']
        all_results.extend(split_result['results'])
        
        # Create logging table for wandb
        logging_columns = ["image_id", "ground_truth", "image", "raw_output", "thinking", 
                          "text_prediction", "text_answer", "correct", "reward"]
        logging_table = wandb.Table(columns=logging_columns)
        
        # Fill the table with results
        for result in split_result['results']:
            logging_table.add_data(
                result['image_id'],
                result['true_label'],
                "dummy_image",  # Placeholder for image
                result['raw_output'],
                result['thinking'],
                result['text_prediction'],
                result['text_answer'],
                result['correct'],
                result['reward']
            )
        
        # Create metrics table for wandb
        metrics_columns = ["accuracy", "precision", "recall", "f1", "reward"]
        text_metrics_table = wandb.Table(columns=metrics_columns)
        text_metrics_table.add_data(*split_result['metrics'])
        
        # Log tables to wandb
        wandb.log({
            f"prediction_table_{split}": logging_table,
            f"metrics_table_{split}": text_metrics_table
        })
        
        # Log metrics
        wandb.log({
            f"{split}/accuracy": split_result['metrics'][0],
            f"{split}/precision": split_result['metrics'][1],
            f"{split}/recall": split_result['metrics'][2],
            f"{split}/f1": split_result['metrics'][3],
            f"{split}/reward": split_result['metrics'][4]
        })
        
        # Print results
        print(f"Model: {args.log_name}, Dataset: {args.dataset}, Backend: vLLM")
        print(f"Metrics for {split} split: Accuracy: {split_result['metrics'][0]:.4f}, "
              f"Precision: {split_result['metrics'][1]:.4f}, Recall: {split_result['metrics'][2]:.4f}, "
              f"F1: {split_result['metrics'][3]:.4f}, Reward: {split_result['metrics'][4]:.4f}")
    
    # Save results
    write_to_csv(log_path, metrics_dict)
    
    # Save detailed results
    results_path = log_path.replace('.csv', '_details.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {log_path}")
    logger.info(f"Detailed results saved to {results_path}")
    
    run.finish()


if __name__ == "__main__":
    main()
