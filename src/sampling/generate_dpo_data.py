#!/usr/bin/env python3
"""
Generate DPO (Direct Preference Optimization) data for hateful meme classification.

This script uses the HuggingFaceResponseGenerator from generate_w_hf.py to create
preference data suitable for DPO training. The output format follows LLaMA-Factory's
multimodal DPO format with chosen/rejected responses.

Usage:
    python generate_dpo_data.py --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
                                --input_data data/hateful_memes.json \
                                --output_file dpo_hateful_memes.json \
                                --image_base_path ./images/ \
                                --num_responses 3
"""

import argparse
import json
import os
import sys
import torch
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the HuggingFaceResponseGenerator from generate_w_hf.py
from generate_w_hf import HuggingFaceResponseGenerator
from config import prompt

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
                entry = json.loads(line)
                standardized_entry = standardize_entry_format(entry, line_num)

                data.append(standardized_entry)

    else:
        # Load JSON format
        with open(input_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for i, entry in enumerate(json_data):
                standardized_entry = standardize_entry_format(entry, i+1)

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
        image_path = entry["image"].replace("img/", "")  # HarMeme format format
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

def generate_responses_for_entry(
    entry: Dict[str, Any],
    generator: HuggingFaceResponseGenerator,
    prompt: str,
    image_base_path: Optional[str],
    num_responses: int,
    temperature: float = 0.7,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Generate multiple responses for a single data entry using batch operations.
    
    Args:
        entry: Data entry with id, image, text, label
        generator: HuggingFaceResponseGenerator instance
        prompt: Single prompt to use for response generation
        image_base_path: Base path for images
        num_responses: Number of responses to generate
        temperature: Fixed temperature to use for sampling
        batch_size: Number of responses to generate in each batch (default: 8)
        
    Returns:
        Dictionary with original entry data and generated responses
    """
    # Construct full image path
    image_path = None
    if image_base_path and "image" in entry:
        image_path = os.path.join(image_base_path, entry["image"])
    elif "image_path" in entry:
        image_path = entry["image_path"]
    
    if image_path and not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        image_path = None
    
    responses = []
    
    # Calculate number of batches needed
    num_batches = (num_responses + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        # Calculate responses for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_responses)
        current_batch_size = end_idx - start_idx
        
        print(f"Generating batch {batch_idx + 1}/{num_batches} with {current_batch_size} responses for entry {entry.get('id', 'unknown')}")
        
        try:
            # Generate batch of responses
            batch_responses = generator.generate_batch_responses(
                prompts=[prompt] * current_batch_size,
                image_paths=[image_path] * current_batch_size,
                temperature=temperature,
                do_sample=True
            )
            print(batch_responses)
            # Process each response in the batch
            for i, response in enumerate(batch_responses):
                if response:
                    responses.append({
                        "prompt": prompt,
                        "response": response,
                        "temperature": temperature,
                        "response_id": start_idx + i
                    })
                else:
                    print(f"Failed to generate response {start_idx + i + 1} for entry {entry.get('id', 'unknown')}")
                    
        except Exception as e:
            print(f"Error generating batch {batch_idx + 1} for entry {entry.get('id', 'unknown')}: {e}")
            # Fallback to single generation for this batch
            print(f"Falling back to single response generation for batch {batch_idx + 1}")
            for i in range(current_batch_size):
                try:
                    response = generator.generate_response(
                        prompt=prompt,
                        image_path=image_path,
                        temperature=temperature,
                        do_sample=True
                    )
                    
                    if response:
                        responses.append({
                            "prompt": prompt,
                            "response": response,
                            "temperature": temperature,
                            "response_id": start_idx + i
                        })
                    else:
                        print(f"Failed to generate response {start_idx + i + 1} for entry {entry.get('id', 'unknown')}")
                        
                except Exception as single_e:
                    print(f"Error generating single response {start_idx + i + 1} for entry {entry.get('id', 'unknown')}: {single_e}")
    
    return {
        "entry": entry,
        "responses": responses
    }


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
) -> List[Dict[str, Any]]:
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
            print(f"Skipping entry {entry.get('id', 'unknown')} - both chosen and rejected are missing")
            stats['both_missing'] += 1
            stats['skip'] += 1
            continue
        elif not chosen_text:
            print(f"Skipping entry {entry.get('id', 'unknown')} - no chosen response available")
            stats['no_chosen'] += 1
            stats['skip'] += 1
            continue
        elif not rejected_text:
            print(f"Skipping entry {entry.get('id', 'unknown')} - no rejected response available")
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
            image_path = os.path.join(image_base_path, entry["image"])
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO data for hateful meme classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model name for response generation"
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
        "--batch_size",
        type=int,
        default=8,
        help="Number of responses to generate in each batch (default: 8)"
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Maximum number of entries to process (for testing)"
    )
    
    # Model-specific arguments
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use flash attention for Qwen models"
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=None,
        help="Minimum pixels for image processing"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Maximum pixels for image processing"
    )
    
    args = parser.parse_args()
    
    
    print("=" * 60)
    print("DPO Data Generation for Hateful Meme Classification")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Input data: {args.input_data}")
    print(f"Output file: {args.output_file}")
    print(f"Image base path: {args.image_base_path}")
    print(f"Number of responses per entry: {args.num_responses}")
    print(f"Temperature: {args.temperature}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Load input data
    print("Loading input data...")
    input_data = load_input_data(args.input_data)
    
    if args.max_entries:
        input_data = input_data[:args.max_entries]
        print(f"Limited to {len(input_data)} entries for testing")
    
    # Initialize model
    print("Initializing model...")
    model_kwargs = {}
    if args.use_flash_attention:
        model_kwargs["use_flash_attention"] = True
    if args.min_pixels:
        model_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels:
        model_kwargs["max_pixels"] = args.max_pixels
    
    generator = HuggingFaceResponseGenerator(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        **model_kwargs
    )

    
    # Generate responses for each entry
    print("Generating responses...")
    data_with_responses = []
    
    for i, entry in enumerate(input_data):
        print(f"\nProcessing entry {i+1}/{len(input_data)}: {entry.get('id', 'unknown')}")
        
        try:
            result = generate_responses_for_entry(
                entry=entry,
                generator=generator,
                prompt=prompt,
                image_base_path=args.image_base_path,
                num_responses=args.num_responses,
                temperature=args.temperature,
                batch_size=args.batch_size
            )
            
            if result["responses"]:
                data_with_responses.append(result)
            else:
                print(f"No responses generated for entry {entry.get('id', 'unknown')}")
                
        except Exception as e:
            print(f"Error processing entry {entry.get('id', 'unknown')}: {e}")
            continue
    
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
    
    print("\n" + "=" * 60)
    print("DPO data generation completed successfully!")
    print(f"Output saved to: {args.output_file}")
    print(f"Total DPO examples: {len(dpo_data)}")
    print("=" * 60)

if __name__ == "__main__":
    main()