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
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the HuggingFaceResponseGenerator from generate_w_hf.py
from generate_w_hf import HuggingFaceResponseGenerator


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
    
    5. Custom JSON format:
       [{"id": "unique_id", "image": "path/to/image.jpg", "text": "content", "label": 0}, ...]
    """
    try:
        data = []
        
        # Determine file format by extension
        if input_path.endswith('.jsonl'):
            # Load JSONL format
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            standardized_entry = standardize_entry_format(entry, line_num)
                            if standardized_entry:
                                data.append(standardized_entry)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num}: {e}")
                            continue
        else:
            # Load JSON format
            with open(input_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    for i, entry in enumerate(json_data):
                        standardized_entry = standardize_entry_format(entry, i+1)
                        if standardized_entry:
                            data.append(standardized_entry)
                else:
                    print(f"Error: JSON file should contain an array of entries")
                    return []
        
        print(f"Loaded {len(data)} entries from {input_path}")
        return data
    except Exception as e:
        print(f"Error loading input data from {input_path}: {e}")
        sys.exit(1)


def standardize_entry_format(entry: Dict[str, Any], line_num: int) -> Optional[Dict[str, Any]]:
    """
    Standardize different dataset formats to a common format.
    
    Args:
        entry: Raw entry from dataset
        line_num: Line number for error reporting
        
    Returns:
        Standardized entry or None if invalid
    """
    try:
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
            image_path = entry["img"]  # FB, MAMI, PrideMM format
        elif "image" in entry:
            image_path = entry["image"]  # HarMeme format or custom format
        else:
            print(f"Warning: Missing image field ('img' or 'image') on line {line_num}")
            return None
        
        # Handle different label formats
        label = None
        if "label" in entry:
            # Simple binary label (FB, MAMI, PrideMM, custom)
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
        
    except Exception as e:
        print(f"Error processing entry on line {line_num}: {e}")
        return None


def get_classification_prompts() -> List[str]:
    """
    Get different prompts for harmful/hateful meme classification.
    These work across different datasets (FB, MAMI, HarMeme, PrideMM).
    """
    return [
        "Analyze this meme and determine if it contains harmful or hateful content. Consider both the image and text. Provide a clear classification (harmful/hateful or not harmful/hateful) and explain your reasoning.",
        
        "Look at this meme carefully. Does it contain hate speech, discriminatory content, or offensive material targeting individuals or groups? Classify it as harmful or not harmful and justify your decision.",
        
        "Examine this meme for any harmful, hateful, discriminatory, or offensive content. Consider the interaction between the image and text. Is this meme harmful? Explain your analysis.",
        
        "Please classify this meme as either harmful or not harmful. Harmful content includes material that attacks, threatens, demeans, or spreads negativity about individuals or groups. Provide detailed reasoning.",
        
        "Evaluate whether this meme contains harmful content by analyzing both visual and textual elements. Consider context, stereotypes, potential harm, and social impact. Give a classification and explanation.",
        
        "Assess this meme for harmful content including but not limited to: hate speech, discrimination, harassment, stereotyping, or content that could cause psychological or social harm. Classify and explain.",
        
        "Determine if this meme promotes harmful ideologies, contains offensive material, or could be considered inappropriate or hurtful. Analyze the content and provide a clear classification with reasoning."
    ]


def generate_responses_for_entry(
    entry: Dict[str, Any],
    generator: HuggingFaceResponseGenerator,
    prompts: List[str],
    image_base_path: Optional[str],
    num_responses: int,
    temperature_range: tuple = (0.3, 1.0)
) -> Dict[str, Any]:
    """
    Generate multiple responses for a single data entry using different prompts and temperatures.
    
    Args:
        entry: Data entry with id, image, text, label
        generator: HuggingFaceResponseGenerator instance
        prompts: List of different prompts to use
        image_base_path: Base path for images
        num_responses: Number of responses to generate
        temperature_range: Range of temperatures to sample from
        
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
    
    # Generate responses with different prompts and temperatures
    responses = []
    
    for i in range(num_responses):
        # Select prompt (cycle through available prompts)
        prompt = prompts[i % len(prompts)]
        
        # Sample temperature from range
        import random
        temperature = random.uniform(temperature_range[0], temperature_range[1])
        
        print(f"Generating response {i+1}/{num_responses} for entry {entry.get('id', 'unknown')} (temp={temperature:.2f})")
        
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
                    "response_id": i
                })
            else:
                print(f"Failed to generate response {i+1} for entry {entry.get('id', 'unknown')}")
                
        except Exception as e:
            print(f"Error generating response {i+1} for entry {entry.get('id', 'unknown')}: {e}")
    
    return {
        "entry": entry,
        "responses": responses
    }


def select_chosen_and_rejected(responses: List[Dict[str, Any]], label: int) -> tuple:
    """
    Select chosen and rejected responses based on label and response quality.
    
    For DPO training:
    - chosen: Response that better aligns with the ground truth label
    - rejected: Response that contradicts or poorly explains the label
    
    Args:
        responses: List of generated responses
        label: Ground truth label (0 = not harmful/hateful, 1 = harmful/hateful)
        
    Returns:
        Tuple of (chosen_response, rejected_response)
    """
    if len(responses) < 2:
        return None, None
    
    # Keywords for harmful content (broader than just "hateful")
    harmful_keywords = [
        "harmful", "hateful", "hate", "offensive", "discriminatory", "inappropriate", 
        "toxic", "hurtful", "problematic", "concerning", "dangerous", "negative",
        "attacking", "targeting", "stereotyping", "harassment", "threatening"
    ]
    
    # Keywords for non-harmful content
    safe_keywords = [
        "not harmful", "not hateful", "not offensive", "harmless", "acceptable", 
        "appropriate", "safe", "benign", "neutral", "positive", "wholesome",
        "inoffensive", "non-threatening", "respectful"
    ]
    
    target_keywords = harmful_keywords if label == 1 else safe_keywords
    opposite_keywords = safe_keywords if label == 1 else harmful_keywords
    
    scored_responses = []
    for resp in responses:
        response_text = resp["response"].lower()
        
        # Score based on keyword alignment
        target_score = sum(1 for kw in target_keywords if kw in response_text)
        opposite_score = sum(1 for kw in opposite_keywords if kw in response_text)
        
        # Bonus for explicit classification statements
        if label == 1:
            if any(phrase in response_text for phrase in ["is harmful", "is hateful", "contains harmful", "contains hateful"]):
                target_score += 2
        else:
            if any(phrase in response_text for phrase in ["is not harmful", "is not hateful", "not harmful", "not hateful"]):
                target_score += 2
        
        # Prefer responses that are longer and more detailed (but not too long)
        length = len(resp["response"])
        if 100 <= length <= 800:  # Sweet spot for detailed but concise responses
            length_score = 1
        elif length > 800:
            length_score = 0.5  # Penalize very long responses
        else:
            length_score = 0.2  # Penalize very short responses
        
        # Bonus for providing reasoning/explanation
        reasoning_indicators = ["because", "due to", "reason", "analysis", "evidence", "example"]
        reasoning_score = sum(0.5 for indicator in reasoning_indicators if indicator in response_text)
        
        final_score = target_score - opposite_score + length_score + reasoning_score
        scored_responses.append((final_score, resp))
    
    # Sort by score (highest first)
    scored_responses.sort(key=lambda x: x[0], reverse=True)
    
    # Select best and worst as chosen/rejected
    chosen = scored_responses[0][1]["response"]
    rejected = scored_responses[-1][1]["response"]
    
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
    
    for item in data_with_responses:
        entry = item["entry"]
        responses = item["responses"]
        
        if len(responses) < 2:
            print(f"Skipping entry {entry.get('id', 'unknown')} - insufficient responses")
            continue
        
        # Select chosen and rejected responses
        chosen_text, rejected_text = select_chosen_and_rejected(responses, entry["label"])
        
        if not chosen_text or not rejected_text:
            print(f"Skipping entry {entry.get('id', 'unknown')} - could not select chosen/rejected")
            continue
        
        # Use the first response's prompt (they should be similar)
        prompt = responses[0]["prompt"]
        
        # Add meme text to prompt if available
        if entry.get("text"):
            prompt = f"{prompt}\n\nMeme text: \"{entry['text']}\""
        
        # Construct the conversation
        user_content = f"<image>{prompt}"
        
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
            image_path = entry["image"]
            llamafactory_entry["images"] = [image_path]
        
        llamafactory_data.append(llamafactory_entry)
    
    return llamafactory_data


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
        "--temperature_min",
        type=float,
        default=0.7,
        help="Minimum temperature for sampling"
    )
    parser.add_argument(
        "--temperature_max",
        type=float,
        default=1.0,
        help="Maximum temperature for sampling"
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
    print(f"Temperature range: {args.temperature_min} - {args.temperature_max}")
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
    
    # Get classification prompts
    prompts = get_classification_prompts()
    print(f"Using {len(prompts)} different prompts for diversity")
    
    # Generate responses for each entry
    print("Generating responses...")
    data_with_responses = []
    
    for i, entry in enumerate(input_data):
        print(f"\nProcessing entry {i+1}/{len(input_data)}: {entry.get('id', 'unknown')}")
        
        try:
            result = generate_responses_for_entry(
                entry=entry,
                generator=generator,
                prompts=prompts,
                image_base_path=args.image_base_path,
                num_responses=args.num_responses,
                temperature_range=(args.temperature_min, args.temperature_max)
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
    dpo_data = convert_to_llamafactory_dpo_format(
        data_with_responses,
        image_base_path=args.image_base_path
    )
    
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