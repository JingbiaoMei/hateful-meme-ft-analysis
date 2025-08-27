"""
Generate preference data for DPO training for hateful meme classification using Hugging Face models.
This script provides functionality to generate responses using various vision-language models.
"""

import os
import torch
import json
from typing import Optional, Dict, Any, List, Union
from PIL import Image

from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM
)

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False


class HuggingFaceResponseGenerator:
    """
    A unified class for generating responses using various Hugging Face vision-language models.
    Supports models like Qwen2.5-VL, LLaVA, and other VL models for DPO dataset generation.
    """
    
    def __init__(self, 
                 model_name: str,
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 do_sample: bool = True,
                 **model_kwargs):
        """
        Initialize the response generator.
        
        Args:
            model_name: Hugging Face model name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
            device: Device to run the model on ("auto", "cuda", "cpu")
            torch_dtype: Torch data type for the model
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling for generation
            **model_kwargs: Additional arguments for model initialization
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        
        # Determine model type
        self.model_type = self._determine_model_type(model_name)
        
        print(f"Loading model: {model_name}")
        print(f"Model type: {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Torch dtype: {torch_dtype}")
        
        # Load model and processor based on type
        self._load_model_and_processor(**model_kwargs)
        
        print("Model loaded successfully")
    
    def _determine_model_type(self, model_name: str) -> str:
        """Determine the type of model based on the model name."""
        model_name_lower = model_name.lower()
        
        if "qwen2.5-vl" in model_name_lower or "qwen2_5-vl" in model_name_lower:
            return "qwen2.5-vl"
        elif "qwen-vl" in model_name_lower:
            return "qwen-vl"
        elif "llava" in model_name_lower:
            return "llava"
        elif "blip" in model_name_lower:
            return "blip"
        elif "instructblip" in model_name_lower:
            return "instructblip"
        else:
            return "generic"
    
    def _load_model_and_processor(self, **model_kwargs):
        """Load model and processor based on the model type."""
        if self.model_type == "qwen2.5-vl":
            self._load_qwen2_5_vl(**model_kwargs)
        else:
            # Generic loading for other models
            self._load_generic_model(**model_kwargs)
    
    def _load_qwen2_5_vl(self, **model_kwargs):
        """Load Qwen2.5-VL model and processor."""
        if not HAS_QWEN_VL_UTILS:
            raise ImportError("qwen_vl_utils is required for Qwen2.5-VL models. Please install it.")
        
        # Extract Qwen-specific kwargs
        use_flash_attention = model_kwargs.get('use_flash_attention', False)
        min_pixels = model_kwargs.get('min_pixels', None)
        max_pixels = model_kwargs.get('max_pixels', None)
        
        # Load model
        if use_flash_attention:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                attn_implementation="flash_attention_2",
                device_map=self.device if self.device != "auto" else "auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                torch_dtype=self.torch_dtype, 
                device_map=self.device if self.device != "auto" else "auto"
            )
        
        # Load processor
        if min_pixels is not None and max_pixels is not None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                min_pixels=min_pixels, 
                max_pixels=max_pixels
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
    
    def _load_generic_model(self, **model_kwargs):
        """Load generic vision-language model and processor."""
        try:
            # Try loading as VL model first
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != "auto" else "auto",
                **{k: v for k, v in model_kwargs.items() if k not in ['use_flash_attention', 'min_pixels', 'max_pixels']}
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Error loading as VL model: {e}")
            raise ValueError(f"Unsupported model type or failed to load: {self.model_name}")
    
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
    
    def generate_response(self, 
                         prompt: str, 
                         image_path: Optional[str] = None,
                         image: Optional[Image.Image] = None,
                         **generation_kwargs) -> Optional[str]:
        """
        Generate response for a given prompt and optional image.
        
        Args:
            prompt: Text prompt for generation
            image_path: Path to image file (optional)
            image: PIL Image object (optional, takes precedence over image_path)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated response text or None if generation failed
        """
        try:
            # Load image if path provided
            if image is None and image_path is not None:
                image = self.load_image(image_path)
                if image is None:
                    return None
            
            # Generate based on model type
            if self.model_type == "qwen2.5-vl":
                return self._generate_qwen2_5_vl(prompt, image, **generation_kwargs)
            else:
                return self._generate_generic(prompt, image, **generation_kwargs)
                
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_qwen2_5_vl(self, prompt: str, image: Optional[Image.Image], **generation_kwargs) -> Optional[str]:
        """Generate response using Qwen2.5-VL model."""
        # Prepare messages
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image", "image": image})
        
        messages = [{"role": "user", "content": content}]
        
        # Apply chat template
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process inputs
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            **generation_kwargs
        }
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response if response else None
    
    def _generate_generic(self, prompt: str, image: Optional[Image.Image], **generation_kwargs) -> Optional[str]:
        """Generate response using generic VL model."""
        # This is a simplified implementation - may need adjustment based on specific model
        if image is not None:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=prompt, return_tensors="pt", padding=True)
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            **generation_kwargs
        }
        
        # Generate
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            else:
                raise NotImplementedError(f"Generation not supported for model type: {self.model_type}")
        
        # Decode response
        if hasattr(inputs, 'input_ids'):
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        else:
            generated_ids_trimmed = generated_ids
            
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response if response else None

    def generate_batch_responses(self, 
                               prompts: List[str], 
                               image_paths: Optional[List[str]] = None,
                               images: Optional[List[Image.Image]] = None,
                               **generation_kwargs) -> List[Optional[str]]:
        """
        Generate multiple responses in batch for efficiency.
        
        Args:
            prompts: List of text prompts for generation
            image_paths: List of image file paths (optional)
            images: List of PIL Image objects (optional, takes precedence over image_paths)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated response texts (None for failed generations)
        """
        batch_size = len(prompts)
        responses = []
        
        try:
            # Load images if paths provided
            if images is None and image_paths is not None:
                images = []
                for image_path in image_paths:
                    if image_path is not None:
                        image = self.load_image(image_path)
                        images.append(image)
                    else:
                        images.append(None)
            elif images is None:
                images = [None] * batch_size
            
            # Generate based on model type
            if self.model_type == "qwen2.5-vl":
                responses = self._generate_batch_qwen2_5_vl(prompts, images, **generation_kwargs)
            else:
                # Fallback to individual generation for non-Qwen models
                for i, (prompt, image) in enumerate(zip(prompts, images)):
                    response = self._generate_generic(prompt, image, **generation_kwargs)
                    responses.append(response)
                
        except Exception as e:
            print(f"Error in batch generation, falling back to individual: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to individual generation
            responses = []
            for i, (prompt, image) in enumerate(zip(prompts, images)):
                try:
                    response = self.generate_response(prompt, image=image, **generation_kwargs)
                    responses.append(response)
                except Exception as individual_e:
                    print(f"Error in individual generation {i}: {individual_e}")
                    responses.append(None)
        
        return responses
    
    def _generate_batch_qwen2_5_vl(self, prompts: List[str], images: List[Optional[Image.Image]], **generation_kwargs) -> List[Optional[str]]:
        """Generate batch responses using Qwen2.5-VL model."""
        # For now, let's use a simpler approach that mimics the single generation
        # but processes multiple at once when possible
        
        batch_text_inputs = []
        batch_image_inputs = []
        batch_video_inputs = []
        
        # Prepare each message individually but collect for batch processing
        for prompt, image in zip(prompts, images):
            # Prepare messages the same way as single generation
            content = [{"type": "text", "text": prompt}]
            if image is not None:
                content.insert(0, {"type": "image", "image": image})
            
            messages = [{"role": "user", "content": content}]
            
            # Apply chat template
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_text_inputs.append(text_input)
            
            # Process vision info for this sample
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                # Handle the case where process_vision_info returns None or empty
                if image_inputs:
                    batch_image_inputs.extend(image_inputs)
                if video_inputs:
                    batch_video_inputs.extend(video_inputs)
            except Exception as vision_e:
                print(f"Warning: Error processing vision info for one sample: {vision_e}")
                # Continue without this sample's vision data
                continue
        
        # If we have no valid samples, return empty list
        if not batch_text_inputs:
            return [None] * len(prompts)
        
        # Process batch inputs
        try:
            inputs = self.processor(
                text=batch_text_inputs,
                images=batch_image_inputs if batch_image_inputs else None,
                videos=batch_video_inputs if batch_video_inputs else None,
                padding=True,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"Error in processor batch processing: {e}")
            raise e
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            **generation_kwargs
        }
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode responses
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Ensure we return the correct number of responses
        result = [response if response else None for response in responses]
        
        # Pad with None if we have fewer responses than expected
        while len(result) < len(prompts):
            result.append(None)
            
        return result[:len(prompts)]  # Truncate if we somehow have too many


def generate_dpo_responses(data_entries: List[Dict[str, Any]], 
                          generator: HuggingFaceResponseGenerator,
                          prompts: List[str],
                          image_base_path: str = None,
                          num_responses_per_prompt: int = 2) -> List[Dict[str, Any]]:
    """
    Generate multiple responses for DPO dataset creation.
    
    Args:
        data_entries: List of data entries containing image and text information
        generator: HuggingFaceResponseGenerator instance
        prompts: List of different prompts to generate responses with
        image_base_path: Base path for images (optional)
        num_responses_per_prompt: Number of responses to generate per prompt
        
    Returns:
        List of entries with generated responses
    """
    results = []
    
    for entry in data_entries:
        entry_results = {
            "id": entry.get("id"),
            "text": entry.get("text", ""),
            "label": entry.get("label"),
            "responses": {}
        }
        
        # Get image path
        image_path = None
        if image_base_path and "image" in entry:
            image_path = os.path.join(image_base_path, entry["image"])
        elif "image_path" in entry:
            image_path = entry["image_path"]
        
        # Generate responses for each prompt
        for prompt_idx, prompt in enumerate(prompts):
            prompt_key = f"prompt_{prompt_idx}"
            entry_results["responses"][prompt_key] = {
                "prompt": prompt,
                "generated_responses": []
            }
            
            # Generate multiple responses for diversity
            for response_idx in range(num_responses_per_prompt):
                print(f"Generating response {response_idx + 1}/{num_responses_per_prompt} for entry {entry.get('id')} with prompt {prompt_idx + 1}")
                
                response = generator.generate_response(
                    prompt=prompt,
                    image_path=image_path
                )
                
                if response:
                    entry_results["responses"][prompt_key]["generated_responses"].append({
                        "response_id": response_idx,
                        "text": response
                    })
                else:
                    print(f"Failed to generate response {response_idx + 1} for entry {entry.get('id')}")
        
        results.append(entry_results)
    
    return results


def save_dpo_responses(responses: List[Dict[str, Any]], output_path: str):
    """Save DPO responses to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        print(f"DPO responses saved to: {output_path}")
    except Exception as e:
        print(f"Error saving DPO responses: {e}")


# Example usage function
def example_usage():
    """Example of how to use the HuggingFaceResponseGenerator for DPO dataset generation."""
    
    # Initialize generator
    generator = HuggingFaceResponseGenerator(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="auto",
        torch_dtype=torch.bfloat16,
        max_new_tokens=512,
        temperature=0.7
    )
    
    # Example prompts for hateful meme classification
    prompts = [
        "Analyze this meme and determine if it contains hateful content. Explain your reasoning.",
        "Is this meme hateful or not? Provide a detailed analysis.",
        "Classify this meme as hateful or non-hateful and justify your decision."
    ]
    
    # Example data entries
    data_entries = [
        {
            "id": "example_1",
            "text": "Example meme text",
            "label": 1,
            "image": "example_image.jpg"
        }
    ]
    
    # Generate responses
    responses = generate_dpo_responses(
        data_entries=data_entries,
        generator=generator,
        prompts=prompts,
        image_base_path="./images/",
        num_responses_per_prompt=3
    )
    
    # Save responses
    save_dpo_responses(responses, "./dpo_responses.json")
    
    return responses


if __name__ == "__main__":
    # Run example
    example_usage()
