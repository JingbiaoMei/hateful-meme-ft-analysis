#!/usr/bin/env python3
"""
Generate DPO (Direct Preference Optimization) data for hateful meme classification using vLLM.

This script uses vLLM's offline inference engine to create preference data suitable for DPO training. 
The vLLM approach batches all requests together to maximize throughput and speed compared to 
the HuggingFace sequential generation approach.

Features:
- Generates multiple responses per sample using vLLM batch processing
- Creates DPO pairs (chosen/rejected) based on format correctness and answer accuracy
- Saves final DPO data in LLaMA-Factory format
- Optionally logs all generation entries for each sample with detailed scoring

Usage:
    python generate_dpo_data_vllm.py --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
                                     --input_data data/hateful_memes.json \
                                     --output_file dpo_hateful_memes.json \
                                     --generation_log_file generation_log.json \
                                     --image_base_path ./images/ \
                                     --num_responses 5
"""

import argparse
import json
import os
import sys
import gc
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# Import transformers for chat template processing
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from config import prompt


class VLLMResponseGenerator:
    """
    Response generator using vLLM's offline inference engine for maximum speed.
    """
    
    def __init__(self, 
                 model_name: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 repetition_penalty: float = 1.0,
                 tensor_parallel_size: int = None,
                 pipeline_parallel_size: int = 1,
                 max_model_len: int = 8192,
                 image_max_pixels: int = 768 * 768,
                 image_min_pixels: int = 32 * 32,
                 adapter_name_or_path: Optional[str] = None,
                 trust_remote_code: bool = True,
                 dtype: str = "bfloat16",
                 **vllm_kwargs):
        """
        Initialize the vLLM response generator.
        
        Args:
            model_name: vLLM-compatible model name
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            tensor_parallel_size: Number of GPUs for tensor parallelism
            pipeline_parallel_size: Number of GPUs for pipeline parallelism
            max_model_len: Maximum model context length
            image_max_pixels: Maximum pixels for image processing
            image_min_pixels: Minimum pixels for image processing
            adapter_name_or_path: Path to LoRA adapter (optional)
            trust_remote_code: Whether to trust remote code
            dtype: Model data type
            **vllm_kwargs: Additional vLLM engine arguments
        """
        if not HAS_VLLM:
            raise ImportError("vLLM is required for this script. Please install it with: pip install vllm")
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for chat template processing. Please install it with: pip install transformers")
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.image_max_pixels = image_max_pixels
        self.image_min_pixels = image_min_pixels
        self.adapter_name_or_path = adapter_name_or_path
        
        # Determine GPU configuration
        if tensor_parallel_size is None:
            import torch
            tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        print(f"Initializing vLLM engine...")
        print(f"Model: {model_name}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"Pipeline parallel size: {pipeline_parallel_size}")
        print(f"Max model length: {max_model_len}")
        
        # Initialize vLLM engine
        engine_args = {
            "model": model_name,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "disable_log_stats": True,
            "enable_lora": adapter_name_or_path is not None,
            "limit_mm_per_prompt": {"image": 4, "video": 2, "audio": 2},
            **vllm_kwargs
        }
        
        self.llm = LLM(**engine_args)
        
        # Initialize tokenizer for chat template processing
        print("Loading tokenizer for chat template processing...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=trust_remote_code
        )
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams(
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        
        # Initialize LoRA request if adapter is provided
        self.lora_request = None
        if adapter_name_or_path:
            self.lora_request = LoRARequest("default", 1, adapter_name_or_path)
        
        print("vLLM engine initialized successfully")
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load image from local path."""
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None
                
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            return None
    
    def _prepare_conversation_format(self, prompt_template: str, image: Optional[Image.Image]) -> str:
        """
        Prepare conversation format for vLLM with Qwen2.5-VL using chat template.
        
        This is the key method that makes multimodal generation work with vLLM.
        vLLM with Qwen2.5-VL requires the chat template to be applied before generation.
        """
        # Remove the <image> token from the prompt template since we'll handle it through messages
        text_prompt = prompt_template.replace("<image>", "").strip()
        
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
    
    def _prepare_multimodal_data(self, image: Optional[Image.Image]) -> Optional[Dict[str, Any]]:
        """Prepare multimodal data for vLLM input."""
        if image is None:
            return None
        
        # For vLLM with Qwen2.5-VL, we need to prepare the image in the expected format
        return {
            "image": image  # vLLM expects a single PIL image for Qwen2.5-VL
        }
    
    def generate_all_responses(self, 
                              entries: List[Dict[str, Any]], 
                              prompt_template: str,
                              image_base_path: Optional[str],
                              num_responses: int,
                              batch_size: int = 1024) -> List[Dict[str, Any]]:
        """
        Generate all responses for all entries in one large batch using vLLM.
        This maximizes throughput by processing everything together.
        
        Args:
            entries: List of data entries
            prompt_template: Prompt template to use
            image_base_path: Base path for images
            num_responses: Number of responses per entry
            batch_size: Maximum batch size for processing
            
        Returns:
            List of results with entry data and generated responses
        """
        print(f"Preparing {len(entries)} entries × {num_responses} responses = {len(entries) * num_responses} total requests")
        
        # Prepare all requests
        all_requests = []
        request_metadata = []
        
        for entry_idx, entry in enumerate(entries):
            # Load image once per entry
            image_path = None
            if image_base_path and "image" in entry:
                image_path = os.path.join(image_base_path, entry["image"])
            elif "image_path" in entry:
                image_path = entry["image_path"]
            
            image = None
            if image_path and os.path.exists(image_path):
                image = self.load_image(image_path)
                if image is None:
                    print(f"Warning: Failed to load image for entry {entry.get('id', 'unknown')}")
            
            # Create multiple requests for this entry (for different responses)
            for response_idx in range(num_responses):
                # Prepare the conversation format using chat template
                formatted_prompt = self._prepare_conversation_format(prompt_template, image)
                
                # Prepare multimodal data
                multi_modal_data = self._prepare_multimodal_data(image)
                
                # Create vLLM input format
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
                
                all_requests.append(vllm_input)
                request_metadata.append({
                    "entry_idx": entry_idx,
                    "response_idx": response_idx,
                    "entry_id": entry.get("id", "unknown")
                })
        
        print(f"Generated {len(all_requests)} requests for vLLM processing")
        
        # Process in batches to avoid memory issues
        all_responses = []
        
        for i in tqdm(range(0, len(all_requests), batch_size), desc="Processing batches"):
            batch_requests = all_requests[i:i + batch_size]
            batch_metadata = request_metadata[i:i + batch_size]
            
            try:
                # Generate responses for this batch
                batch_results = self.llm.generate(batch_requests, self.sampling_params, lora_request=self.lora_request)
                
                # Extract generated text
                batch_responses = [result.outputs[0].text for result in batch_results]
                
                # Combine with metadata
                for response_text, metadata in zip(batch_responses, batch_metadata):
                    all_responses.append({
                        "entry_idx": metadata["entry_idx"],
                        "response_idx": metadata["response_idx"],
                        "entry_id": metadata["entry_id"],
                        "response": response_text
                    })
                
                # Clear memory
                del batch_results
                gc.collect()
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add None responses for failed batch
                for metadata in batch_metadata:
                    all_responses.append({
                        "entry_idx": metadata["entry_idx"],
                        "response_idx": metadata["response_idx"], 
                        "entry_id": metadata["entry_id"],
                        "response": None
                    })
        
        # Group responses by entry
        results = []
        for entry_idx, entry in enumerate(entries):
            entry_responses = []
            
            # Collect all responses for this entry
            for resp in all_responses:
                if resp["entry_idx"] == entry_idx and resp["response"] is not None:
                    entry_responses.append({
                        "prompt": prompt_template,  # Keep the original prompt with <image> token for consistency
                        "response": resp["response"],
                        "temperature": self.temperature,
                        "response_id": resp["response_idx"]
                    })
            
            results.append({
                "entry": entry,
                "responses": entry_responses
            })
            
            #print(f"Entry {entry.get('id', 'unknown')}: {len(entry_responses)}/{num_responses} responses generated")
        
        return results


def load_input_data(input_path: str) -> List[Dict[str, Any]]:
    """
    Load input data from JSON or JSONL file.
    
    Supports multiple formats:
    1. Facebook Hateful Memes format (JSONL):
       {"id": "08291", "img": "img/08291.png", "label": 1, "text": "meme text"}
    
    2. MAMI format (JSONL, same as FB):
       {"id": "8716", "img": "img/8716.jpg", "label": 1, "text": "GETS MARRIED TO GIRL..."}
    
    3. HarMeme format (JSONL):
       {"id": "covid_memes_18", "image": "covid_memes_18.png", "labels": ["somewhat harmful", "individual"], "text": "Bernie or Elizabeth?..."}
    
    4. PrideMM format (JSONL, same as FB):
       {"id": "img_1", "img": "img/img_1.png", "label": 0, "text": "transgirls who grow boobs..."}
    """
    data = []
    
    # Determine file format by extension
    if input_path.endswith('.jsonl'):
        # Load JSONL format
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                standardized_entry = standardize_entry_format(entry, line_num)
                if standardized_entry:
                    data.append(standardized_entry)
    else:
        # Load JSON format
        with open(input_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for i, entry in enumerate(json_data):
                standardized_entry = standardize_entry_format(entry, i+1)
                if standardized_entry:
                    data.append(standardized_entry)

    print(f"Loaded {len(data)} entries from {input_path}")
    return data


def standardize_entry_format(entry: Dict[str, Any], line_num: int) -> Optional[Dict[str, Any]]:
    """
    Standardize different dataset formats to a common format.
    
    Args:
        entry: Raw entry from dataset
        line_num: Line number for error reporting
        
    Returns:
        Standardized entry or None if invalid
    """
    # Required fields
    if "id" not in entry:
        print(f"Warning: Missing 'id' field on line {line_num}")
        return None
    
    if "text" not in entry:
        print(f"Warning: Missing 'text' field on line {line_num}")
        return None
    
    # Handle different image field names
    image_path = None
    if "img" in entry:
        image_path = entry["img"].replace("img/", "")  # FB, MAMI, PrideMM format
    elif "image" in entry:
        image_path = entry["image"].replace("img/", "")  # HarMeme format
    else:
        print(f"Warning: Missing image field ('img' or 'image') on line {line_num}")
        return None
    
    # Handle different label formats
    label = None
    if "label" in entry:
        # Simple binary label (FB, MAMI, PrideMM)
        label = entry["label"]
        if not isinstance(label, int) or label not in [0, 1]:
            print(f"Warning: Invalid label '{label}' on line {line_num}, should be 0 or 1")
            return None
    elif "labels" in entry:
        # HarMeme format with multiple labels
        labels = entry["labels"]
        if isinstance(labels, list) and len(labels) > 0:
            # Convert HarMeme labels to binary
            # "not harmful" -> 0, anything else -> 1
            if "not harmful" in labels:
                label = 0
            else:
                label = 1
        else:
            print(f"Warning: Invalid labels format on line {line_num}")
            return None
    else:
        print(f"Warning: Missing label field on line {line_num}")
        return None
    
    # Create standardized entry
    standardized_entry = {
        "id": str(entry["id"]),
        "image": image_path,
        "text": entry["text"],
        "label": label
    }
    
    # Add original labels for reference if they exist
    if "labels" in entry:
        standardized_entry["original_labels"] = entry["labels"]
    
    return standardized_entry


def format_reward(predict_str: str) -> float:
    """
    Check if the response follows the correct format with <think> and <answer> tags.
    
    Args:
        predict_str: The generated response string
        
    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    Custom accuracy reward function based on exact match with answer tags.
    Expects answers to be either "yes" or "no" and uses <answer> tags for extraction.
    
    Args:
        predict_str: The generated response string
        ground_truth: The ground truth label as string ("yes" for label=1, "no" for label=0)
        
    Returns:
        1.0 if prediction is correct, 0.0 otherwise
    """
    try:
        # Convert to lowercase for comparison
        predict_str_lower = predict_str.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Pattern for extracting content from <answer> tags
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        
        # Extract answer from ground truth if it has answer tags
        sol_match = re.search(answer_tag_pattern, ground_truth_lower, re.DOTALL)
        extracted_ground_truth = sol_match.group(1).strip() if sol_match else ground_truth_lower.strip()
        
        # Extract answer from prediction if it has answer tags
        content_match = re.search(answer_tag_pattern, predict_str_lower, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else predict_str_lower.strip()
        
        # Normalize by removing spaces and underscores
        extracted_ground_truth = extracted_ground_truth.replace(' ', '').replace('_', '')
        student_answer = student_answer.replace(' ', '').replace('_', '')
        
        # Check if student answer is empty
        if student_answer == "" or student_answer == " ":
            return 0.0
            
        # Check if answer is valid (only "yes" or "no" allowed)
        if not (student_answer == "yes" or student_answer == "no"):
            return 0.0
            
        # Compare the extracted answers (bidirectional containment check)
        if extracted_ground_truth in student_answer or student_answer in extracted_ground_truth:
            return 1.0
            
        return 0.0
        
    except Exception:
        return 0.0


def select_chosen_and_rejected(responses: List[Dict[str, Any]], label: int) -> tuple:
    """
    Select chosen and rejected responses based on format correctness and answer accuracy.
    
    For DPO training:
    - chosen: Response that has correct format AND correct prediction (label=1 -> "yes", label=0 -> "no")
    - rejected: Response that has incorrect format OR incorrect prediction
    
    Args:
        responses: List of generated responses
        label: Ground truth label (0 = not harmful/hateful, 1 = harmful/hateful)
        
    Returns:
        Tuple of (chosen_response, rejected_response)
    """
    if len(responses) < 2:
        return None, None
    
    # Convert label to expected answer
    ground_truth = "yes" if label == 1 else "no"
    
    scored_responses = []
    for resp in responses:
        response_text = resp["response"]
        
        # Calculate format reward (1.0 if correct format, 0.0 otherwise)
        format_score = format_reward(response_text)
        
        # Calculate accuracy reward (1.0 if correct answer, 0.0 otherwise)
        accuracy_score = acc_reward(response_text, ground_truth)
        
        # Combined score: only responses with both correct format AND correct answer get high score
        # This ensures chosen responses meet both criteria
        combined_score = format_score * accuracy_score
        
        scored_responses.append((combined_score, resp))
    
    # Sort by score (highest first)
    scored_responses.sort(key=lambda x: x[0], reverse=True)
    
    # Find chosen (highest scoring response with both format and accuracy = 1.0)
    chosen = None
    rejected = None
    
    # Look for a response with perfect score (both format and accuracy correct)
    for score, resp in scored_responses:
        if score == 1.0:  # Perfect format and accuracy
            chosen = resp["response"]
            break
    
    # Find rejected (lowest scoring response, preferably with score < 1.0)
    for score, resp in reversed(scored_responses):
        if score < 1.0:  # Imperfect format or accuracy
            rejected = resp["response"]
            break
    
    # Ensure chosen and rejected are different
    if chosen == rejected and len(scored_responses) > 1:
        chosen = None
        rejected = None
    
    return chosen, rejected


def convert_to_llamafactory_dpo_format(
    data_with_responses: List[Dict[str, Any]],
    image_base_path: Optional[str] = None
) -> tuple:
    """
    Convert generated responses to LLaMA-Factory DPO format.
    
    Expected output format:
    [
        {
            "messages": [
                {
                    "content": "<image>Analyze this meme for hateful content...",
                    "role": "user"
                }
            ],
            "chosen": {
                "content": "This meme is not hateful because...",
                "role": "assistant"
            },
            "rejected": {
                "content": "This meme is hateful because...",
                "role": "assistant"
            },
            "images": ["path/to/image.jpg"]
        },
        ...
    ]
    """
    llamafactory_data = []
    
    # Statistics tracking
    stats = {
        'total_entries': len(data_with_responses),
        'skip': 0,
        'successful_pairs': 0,
        'no_chosen': 0,
        'no_rejected': 0,
        'both_missing': 0
    }
    
    for item in data_with_responses:
        entry = item["entry"]
        responses = item["responses"]
        
        # Select chosen and rejected responses
        chosen_text, rejected_text = select_chosen_and_rejected(responses, entry["label"])
        
        # Track different failure cases
        if not chosen_text and not rejected_text:
            #print(f"Skipping entry {entry.get('id', 'unknown')} - both chosen and rejected are missing")
            stats['both_missing'] += 1
            stats['skip'] += 1
            continue
        elif not chosen_text:
            #print(f"Skipping entry {entry.get('id', 'unknown')} - no chosen response available")
            stats['no_chosen'] += 1
            stats['skip'] += 1
            continue
        elif not rejected_text:
            #print(f"Skipping entry {entry.get('id', 'unknown')} - no rejected response available")
            stats['no_rejected'] += 1
            stats['skip'] += 1
            continue
        
        # Use the first response's prompt (they should all be the same now)
        prompt_text = responses[0]["prompt"]
        
        # Construct the conversation
        user_content = f"{prompt_text}"
        
        # Create LLaMA-Factory format entry
        llamafactory_entry = {
            "messages": [
                {
                    "content": user_content,
                    "role": "user"
                }
            ],
            "chosen": {
                "content": chosen_text,
                "role": "assistant"
            },
            "rejected": {
                "content": rejected_text,
                "role": "assistant"
            }
        }
        
        # Add image path
        if entry.get("image"):
            # Use relative path for LLaMA-Factory
            image_path = os.path.join(image_base_path or "", entry["image"])
            llamafactory_entry["images"] = [image_path]
        
        llamafactory_data.append(llamafactory_entry)
        stats['successful_pairs'] += 1
    
    return llamafactory_data, stats


def save_dpo_data(data: List[Dict[str, Any]], output_path: str):
    """Save DPO data in LLaMA-Factory format."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"DPO data saved to: {output_path}")
        print(f"Generated {len(data)} DPO training examples")
        
    except Exception as e:
        print(f"Error saving DPO data: {e}")
        sys.exit(1)


def save_generation_log(data_with_responses: List[Dict[str, Any]], output_path: str):
    """
    Save all generation entries for each sample to a log file.
    
    This creates a comprehensive log that includes:
    - Original entry data (id, text, label, image path)
    - All generated responses with metadata
    - Response quality scores (format and accuracy)
    
    Args:
        data_with_responses: List of results with entry data and generated responses
        output_path: Path to save the generation log file
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        generation_log = []
        
        for item in data_with_responses:
            entry = item["entry"]
            responses = item["responses"]
            
            # Convert label to expected answer for scoring
            ground_truth = "yes" if entry["label"] == 1 else "no"
            
            # Score each response
            scored_responses = []
            for resp in responses:
                response_text = resp["response"]
                
                # Calculate format reward (1.0 if correct format, 0.0 otherwise)
                format_score = format_reward(response_text)
                
                # Calculate accuracy reward (1.0 if correct answer, 0.0 otherwise)
                accuracy_score = acc_reward(response_text, ground_truth)
                
                # Combined score
                combined_score = format_score * accuracy_score
                
                scored_responses.append({
                    "response_id": resp.get("response_id", "unknown"),
                    "prompt": resp["prompt"],
                    "response": response_text,
                    "temperature": resp.get("temperature", "unknown"),
                    "format_score": format_score,
                    "accuracy_score": accuracy_score,
                    "combined_score": combined_score
                })
            
            # Sort responses by combined score (highest first)
            scored_responses.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Select chosen and rejected for reference
            chosen_text, rejected_text = select_chosen_and_rejected(responses, entry["label"])
            
            # Create log entry
            log_entry = {
                "entry_id": entry.get("id", "unknown"),
                "original_entry": {
                    "id": entry.get("id"),
                    "text": entry.get("text"),
                    "label": entry.get("label"),
                    "image": entry.get("image"),
                    "original_labels": entry.get("original_labels")
                },
                "ground_truth_answer": ground_truth,
                "total_responses": len(scored_responses),
                "responses": scored_responses,
                "dpo_selection": {
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "has_valid_pair": chosen_text is not None and rejected_text is not None
                },
                "statistics": {
                    "perfect_responses": sum(1 for r in scored_responses if r["combined_score"] == 1.0),
                    "format_correct": sum(1 for r in scored_responses if r["format_score"] == 1.0),
                    "accuracy_correct": sum(1 for r in scored_responses if r["accuracy_score"] == 1.0),
                    "average_combined_score": sum(r["combined_score"] for r in scored_responses) / len(scored_responses) if scored_responses else 0.0
                }
            }
            
            generation_log.append(log_entry)
        
        # Save the log
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(generation_log, f, indent=2, ensure_ascii=False)
        
        print(f"Generation log saved to: {output_path}")
        print(f"Logged {len(generation_log)} entries with all generation details")
        
        # Print summary statistics
        total_responses = sum(len(entry["responses"]) for entry in generation_log)
        total_perfect = sum(entry["statistics"]["perfect_responses"] for entry in generation_log)
        total_format_correct = sum(entry["statistics"]["format_correct"] for entry in generation_log)
        total_accuracy_correct = sum(entry["statistics"]["accuracy_correct"] for entry in generation_log)
        valid_pairs = sum(1 for entry in generation_log if entry["dpo_selection"]["has_valid_pair"])
        
        print(f"Generation log summary:")
        print(f"  - Total responses: {total_responses}")
        print(f"  - Perfect responses (format + accuracy): {total_perfect} ({total_perfect/total_responses*100:.1f}%)")
        print(f"  - Format correct: {total_format_correct} ({total_format_correct/total_responses*100:.1f}%)")
        print(f"  - Accuracy correct: {total_accuracy_correct} ({total_accuracy_correct/total_responses*100:.1f}%)")
        print(f"  - Valid DPO pairs: {valid_pairs} ({valid_pairs/len(generation_log)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error saving generation log: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO data for hateful meme classification using vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="vLLM-compatible model name for response generation"
    )
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)"
    )
    
    # Data arguments
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to input JSON file with hateful meme data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSON file for DPO data"
    )
    parser.add_argument(
        "--generation_log_file",
        type=str,
        default=None,
        help="Path to output JSON file for generation log (optional, saves all generation entries for each sample)"
    )
    parser.add_argument(
        "--image_base_path",
        type=str,
        default=None,
        help="Base path for images (if not absolute paths in data)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--num_responses",
        type=int,
        default=5,
        help="Number of responses to generate per entry"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for vLLM processing (default: 1024)"
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Maximum number of entries to process (for testing)"
    )
    
    # vLLM-specific arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (auto-detect if not specified)"
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for pipeline parallelism"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--image_max_pixels",
        type=int,
        default=768 * 768,
        help="Maximum pixels for image processing"
    )
    parser.add_argument(
        "--image_min_pixels",
        type=int,
        default=32 * 32,
        help="Minimum pixels for image processing"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model data type (auto, float16, bfloat16, etc.)"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("vLLM DPO Data Generation for Hateful Meme Classification")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Input data: {args.input_data}")
    print(f"Output file: {args.output_file}")
    print(f"Generation log file: {args.generation_log_file or 'Not specified'}")
    print(f"Image base path: {args.image_base_path}")
    print(f"Number of responses per entry: {args.num_responses}")
    print(f"Temperature: {args.temperature}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 60)
    
    # Load input data
    print("Loading input data...")
    input_data = load_input_data(args.input_data)
    
    if args.max_entries:
        input_data = input_data[:args.max_entries]
        print(f"Limited to {len(input_data)} entries for testing")
    
    # Initialize vLLM generator
    print("Initializing vLLM generator...")
    generator = VLLMResponseGenerator(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=args.max_model_len,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        adapter_name_or_path=args.adapter_name_or_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype
    )
    
    if args.seed is not None:
        generator.sampling_params.seed = args.seed
    
    # Generate all responses using vLLM batch processing
    print("Generating responses using vLLM...")
    data_with_responses = generator.generate_all_responses(
        entries=input_data,
        prompt_template=prompt,
        image_base_path=args.image_base_path,
        num_responses=args.num_responses,
        batch_size=args.batch_size
    )
    
    print(f"\nGenerated responses for {len(data_with_responses)}/{len(input_data)} entries")
    
    # Convert to LLaMA-Factory DPO format
    print("Converting to LLaMA-Factory DPO format...")
    dpo_data, stats = convert_to_llamafactory_dpo_format(
        data_with_responses,
        image_base_path=args.image_base_path
    )
    
    # Print detailed statistics
    print("\n" + "=" * 60)
    print("DPO DATA GENERATION STATISTICS")
    print("=" * 60)
    print(f"Total entries processed: {stats['total_entries']}")
    print(f"Missing chosen responses: {stats['no_chosen']}")
    print(f"Missing rejected responses: {stats['no_rejected']}")
    print(f"Missing both chosen & rejected: {stats['both_missing']}")
    print(f"Total failed chosen/rejected pairs: {stats['skip']}")
    print(f"Successful DPO pairs created: {stats['successful_pairs']}")
    print(f"Success rate: {stats['successful_pairs']/stats['total_entries']*100:.1f}%" if stats['total_entries'] > 0 else "Success rate: 0.0%")
    print("=" * 60)
    
    # Save DPO data
    print("Saving DPO data...")
    save_dpo_data(dpo_data, args.output_file)
    
    # Save generation log if requested
    if args.generation_log_file:
        print("Saving generation log...")
        save_generation_log(data_with_responses, args.generation_log_file)
    
    print("\n" + "=" * 60)
    print("vLLM DPO data generation completed successfully!")
    print(f"DPO data saved to: {args.output_file}")
    if args.generation_log_file:
        print(f"Generation log saved to: {args.generation_log_file}")
    print(f"Total DPO examples: {len(dpo_data)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
