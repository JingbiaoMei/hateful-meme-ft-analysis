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
import multiprocessing as mp
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from dataset import get_Dataloader
import csv
import math

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def exact_match_reward(raw_output, solution, **kwargs):

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
        writer.writerow(["split", "acc", "precision", "recall", "f1"])
        for split, values in metrics.items():
            values = [round(value, 4) for value in values]
            writer.writerow([split] + values)

def evaluate_predictions(y_true, y_pred, rewards=None):
    """Calculate evaluation metrics."""
    metrics = []
    metrics.append(accuracy_score(y_true, y_pred))
    metrics.append(precision_score(y_true, y_pred))
    metrics.append(recall_score(y_true, y_pred))
    metrics.append(f1_score(y_true, y_pred))
    if rewards is not None:
        metrics.append(np.mean(rewards))
    else:
        metrics.append(0.0)
    return metrics

def run_inference_on_batch(batch, model, processor, args):
    """Process a single batch of data."""
    images, texts, labels, image_ids = batch
    
    # Prepare batch of messages
    question = "This is an image with \"{text}\" written on it."+args.query
    # Create batch of messages
    messages = []
    for img, text in zip(images, texts):
        formatted_question = question.format(text=text)
            
        messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"<image>{formatted_question}"}
            ]
        }])
    
    # Prepare inputs for batch inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate responses for the batch
    generation_output = model.generate(
        **inputs, 
        max_new_tokens=1024, 
        use_cache=True,
        # Greedy search parameters
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_k=None,
        top_p=None,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation_output)
    ]
    responses = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    batch_results = []
    batch_text_preds = []
    batch_labels = []
    batch_reward = []
    
    # Process batch responses
    for response, label, img_id, img in zip(responses, labels, image_ids, images):
        try:
            if args.debug:
                logger.info(f"Response for image {img_id}: {response}")
            
            thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            thinking = thinking_match.group(1).strip() if thinking_match else ""
            
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if not answer_match:
                logger.warning(f"No answer tag found in regex for image {img_id}: {response}")
                continue
                
            answer = answer_match.group(1).strip().lower()
            gt = 'yes' if int(label.item()) == 1 else 'no'
            reward = exact_match_reward(response, gt, **{})
            
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
            batch_reward.append(reward)
            
        except Exception as e:
            logger.error(f"Error processing image {img_id}: {str(e)}")
            logger.error(f"Raw response: {response}")
            continue
            
    return batch_results, batch_text_preds, batch_labels, batch_reward, image_ids

def process_batches(rank, batch_data, args):
    """Process a set of batches on a specific GPU."""
    logger.info(f"Starting GPU {rank} with {len(batch_data)} batches to process")
    
    # Load model on specific GPU
    if "Qwen2-VL" in args.exp_name or "qwen2_vl" in args.exp_name:
        model_cls = Qwen2VLForConditionalGeneration
    elif "Qwen2.5-VL" in args.exp_name or "qwen25_vl" in args.exp_name or "qwen2_5_vl" in args.exp_name:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model: {args.exp_name}")
    
    if args.base_model_path:
        model = model_cls.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{rank}",
        ).eval()
        # Make the code compatible with multiple lora adapters
        if not ":" in args.model_path:
            model.load_adapter(args.model_path, adapter_name="adapter_0") # Assign a unique name
        else:
            lora_adapters = args.model_path.split(":")
            for i, lora_adapter_path in enumerate(lora_adapters):
                adapter_name = f"adapter_{i}" # Generate a unique name, e.g., "adapter_0", "adapter_1"
                print(f"Loading: {lora_adapter_path} as {adapter_name}")
                model.load_adapter(lora_adapter_path, adapter_name=adapter_name)
    else:
        model = model_cls.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{rank}",
        ).eval()
    model.eval()

    # Load processor
    if args.max_pixels:
        processor = AutoProcessor.from_pretrained(args.processor_path, max_pixels=args.max_pixels, use_fast=True)
    else:
        processor = AutoProcessor.from_pretrained(args.processor_path, use_fast=True)
    processor.tokenizer.padding_side = "left"
    
    # Track predictions
    all_results = []
    all_text_predictions = []
    all_labels = []
    all_ids = []
    all_rewards = []
    
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(batch_data, desc=f"GPU {rank}")):
        # In debug mode, only process specified number of batches
        if args.debug and batch_idx >= args.debug_batches:
            logger.info(f"Debug mode: Stopping after {args.debug_batches} batches")
            break
        
        batch_results, batch_text_preds, batch_labels, batch_reward, batch_ids = run_inference_on_batch(
            batch, model, processor, args
        )
        
        all_results.extend(batch_results)
        all_text_predictions.extend(batch_text_preds)
        all_labels.extend(batch_labels)
        all_ids.extend(batch_ids)
        all_rewards.extend(batch_reward)
    
    # Calculate metrics for predictions if we have any
    if all_text_predictions:
        text_metrics = evaluate_predictions(all_labels, all_text_predictions, all_rewards)
        logger.info(f"GPU {rank} - Local metrics: {text_metrics}")
    else:
        text_metrics = [0.0, 0.0, 0.0, 0.0, 0.0]
        logger.warning(f"GPU {rank} - No valid predictions were collected")
    
    # Return all the results
    return {
        'metrics': text_metrics,
        'results': all_results,
        'ids': all_ids,
        'labels': all_labels,
        'predictions': all_text_predictions,
        'rewards': all_rewards,
    }

def main():
    parser = argparse.ArgumentParser(description='Hateful Memes Classification with Qwen2-VL')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--base_model_path', type=str, default=None, help='Path to base model checkpoint')
    parser.add_argument('--processor_path', type=str, required=True, help='Path to processor')
    parser.add_argument('--data_split', type=str, default='dev_seen test_seen', help='Evaluate on which dataset split')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--dataset', type=str, default="FB", help='Dataset name')
    parser.add_argument('--log_name', type=str, default="", help='Log name')
    parser.add_argument('--group_name', type=str, default="inference_classifier", help='Group name for wandb')
    parser.add_argument('--debug', action='store_true', default=False, help='Run in debug mode (process limited batches)')
    parser.add_argument('--debug_batches', type=int, default=10, help='Number of batches to process in debug mode')
    parser.add_argument('--query', type=str, default="Please identify if the meme is hateful or not.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n For the thinking process, consider both the image and any accompanying text. Evaluate the historical, cultural, and social context surrounding the meme. Analyze different interpretations, including potential humor or satirical elements versus actual hate content. Reflect on broader social and racial implications. Consider if the meme perpetuates harmful stereotypes or could be seen as targeting a particular group. Assess how various audiences might perceive the content. Weigh the intent behind the meme against its potential for harm. Synthesize your reasoning and decide if the meme qualifies as hateful.\n The answer should be Yes or No, if the meme is hateful than output <answer> Yes </answer>; if the meme is Benign than output <answer> No </answer>. The output format should be as follows:\n<think> ... </think> <answer> ... </answer>\n. Please strictly follow the format.", help='Query to pass to the model')
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--max_pixels', type=int, default=None,
                        help='Set the max resolution')
    args = parser.parse_args()
    
    # Setup debugpy if in debug mode
    if args.debug:
        logger.info(f"Debug mode enabled: Will process {args.debug_batches} batches")
        try:
            import debugpy
            debugpy.listen(5678)
            debugpy.wait_for_client()
            logger.info("Debugger attached! Continuing execution.")
        except ImportError:
            logger.warning("debugpy not installed, skipping remote debugging setup.")
            logger.warning("To enable remote debugging, install debugpy: pip install debugpy")
    

    
    model_name = os.path.basename(args.model_path)

    
    # Add debug flag to log name and path if in debug mode
    if args.debug:
        args.log_name += "_debug"
        args.group_name = "debug"
        
    log_path = f"./logging/{args.exp_name}/{args.dataset}/{args.log_name}/"
    os.makedirs(log_path, exist_ok=True)
    log_path += args.log_name + ".csv"
    
    # Initialize wandb with debug tag if in debug mode
    exp_name = f"{args.log_name}_{args.dataset}"
    tags = [args.dataset, model_name]
    if args.debug:
        tags.append("debug")
        
    run = wandb.init(
        name=exp_name,
        project="RFT-Inference",
        config={
            "model": model_name,
            "dataset": args.dataset,
            "debug": args.debug
        },
        group=args.group_name,
        tags=tags
    )
    
    # Define splits
    if args.dataset == "FB":
        splits = ["train", "dev_seen", "test_seen", "test_unseen"]
    else:
        splits = ["train", "val", "test"]
    
    # Check for multi-GPU availability
    n_gpus = torch.cuda.device_count()
    multiprocess = n_gpus >= 2
    if multiprocess:
        logger.info(f"Multi-GPU processing enabled with {n_gpus} GPUs")
        mp.set_start_method('spawn', force=True)
    else:
        logger.info("Single GPU processing mode")
    
    # Get dataloaders in the main process
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
        
        # Convert dataloader into a list of batches in the main process
        # This avoids nested multiprocessing issues
        logger.info(f"Loading all batches for {split} split...")
        all_batches = []
        for batch in tqdm(dataloader):
            all_batches.append(batch)
        
        logger.info(f"Total batches: {len(all_batches)}")
        
        if multiprocess:
            # Split batches across GPUs
            gpu_batches = [[] for _ in range(n_gpus)]
            for i, batch in enumerate(all_batches):
                gpu_batches[i % n_gpus].append(batch)
            
            for i in range(n_gpus):
                logger.info(f"GPU {i} assigned {len(gpu_batches[i])} batches")
            
            # Process batches in parallel
            with Pool(n_gpus) as pool:
                gpu_results = pool.starmap(
                    process_batches, 
                    [(i, gpu_batches[i], args) for i in range(n_gpus)]
                )
            
            # Combine results from all GPUs
            combined_results = []
            combined_text_predictions = []
            combined_labels = []
            combined_ids = []
            combined_rewards = []
            
            for res in gpu_results:
                combined_results.extend(res['results'])
                combined_text_predictions.extend(res['predictions'])
                combined_labels.extend(res['labels'])
                combined_ids.extend(res['ids'])
                combined_rewards.extend(res['rewards'])
            
            # Calculate metrics on combined results
            if combined_text_predictions:
                text_metrics = evaluate_predictions(combined_labels, combined_text_predictions, combined_rewards)
            else:
                text_metrics = [0.0, 0.0, 0.0, 0.0, 0.0]
                logger.warning(f"No valid predictions were collected for {split} split")
        else:
            # Single GPU processing - process all batches on one GPU
            single_gpu_result = process_batches(0, all_batches, args)
            combined_results = single_gpu_result['results']
            combined_ids = single_gpu_result['ids']
            combined_labels = single_gpu_result['labels']
            combined_text_predictions = single_gpu_result['predictions']
            combined_rewards = single_gpu_result['rewards']

            if combined_text_predictions:
                text_metrics = evaluate_predictions(combined_labels, combined_text_predictions, combined_rewards)
            else:
                text_metrics = [0.0, 0.0, 0.0, 0.0, 0.0]
                logger.warning(f"No valid predictions were collected for {split} split")
        
        # Store metrics
        metrics_dict[f"{split}"] = text_metrics
        all_results.extend(combined_results)
        
        # Create logging table for wandb
        logging_columns = ["image_id", "ground_truth", "image", "raw_output", "thinking", 
                       "text_prediction", "text_answer"]
        logging_table = wandb.Table(columns=logging_columns)
        
        # Fill the table with results
        for result in combined_results:
            logging_table.add_data(
                result['image_id'],               # image_id
                result['true_label'],             # ground_truth
                "dummy_image",                    # image
                result['raw_output'],             # raw_output
                result['thinking'],               # thinking
                result['text_prediction'],        # text_prediction
                result['text_answer']             # text_answer
            )
        
        # Create metrics table for wandb
        metrics_columns = ["accuracy", "precision", "recall", "f1", "reward"]
        text_metrics_table = wandb.Table(columns=metrics_columns)
        text_metrics_table.add_data(*text_metrics)
        
        # Log tables to wandb
        wandb.log({
            f"prediction_table_{split}": logging_table,
            f"metrics_table_{split}": text_metrics_table
        })
        
        # Log metrics
        wandb.log({
            f"{split}/accuracy": text_metrics[0],
            f"{split}/precision": text_metrics[1],
            f"{split}/recall": text_metrics[2],
            f"{split}/f1": text_metrics[3],
            f"{split}/reward": text_metrics[4] if len(text_metrics) > 4 else 0.0
        })
        #print(f"Metrics for {split} split: {text_metrics}")
        print(f"Model: {args.log_name}, Dataset: {args.dataset}")
        print(f"Metrics for {split} split: Accuracy: {text_metrics[0]}, Precision: {text_metrics[1]}, Recall: {text_metrics[2]}, F1: {text_metrics[3]}, Reward: {text_metrics[4] if len(text_metrics) > 4 else 0.0}")

    # Save results
    write_to_csv(log_path, metrics_dict)
    
    # Save detailed results
    results_path = log_path.replace('.csv', '_details.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    run.finish()

if __name__ == "__main__":
    main() 