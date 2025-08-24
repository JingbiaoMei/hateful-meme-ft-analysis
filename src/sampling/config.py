
"""
Configuration file for DPO dataset generation using Hugging Face models.
Contains prompts, model configurations, and sampling parameters.
"""

import torch

# =============================================================================
# PROMPTS FOR DPO GENERATION
# =============================================================================

# Original prompt for structured analysis
prompt_v1 = """
You are a helpful assistant trained to assist content moderators. Given a problematic meme, for example harmful or hateful, consisting of an image and associated text, your task is to construct a structured explanation that helps a human moderator judge whether the meme violates hate speech policy.

Instructions:

- Carefully analyze both the image and text together.
    
- Provide a structured explanation using this template:
    
Template: {content in text and image} violates {policy category} because {reason}.

- {content}: Highlight specific parts of the meme (words, phrases, visual elements) that contribute to the violation.
    
- {policy category}: The explanation should contain the most appropriate policy that the content violates.
    
- {reason}: Concisely explain why this content violates the policy (e.g., historical context, hate symbol, derogatory implication).

Please strictly follow the template and do not include the curly braces in your response.
"""

# Alternative prompts for DPO generation diversity
prompt_v2_classification = """
Analyze this meme and classify it as either "hateful" or "non-hateful". Consider both the text and visual elements.

If hateful, explain:
1. What specific elements make it hateful
2. What groups or individuals it targets
3. Why this content is harmful

If non-hateful, explain:
1. Why it doesn't constitute hate speech
2. What makes it acceptable content

Provide a clear classification followed by your reasoning.
"""

prompt_v3_detailed_analysis = """
You are an expert content moderator specializing in hateful meme detection. Examine this meme carefully and provide a comprehensive analysis.

Consider:
- Visual elements (images, symbols, gestures)
- Textual content (words, phrases, implications)
- Cultural context and potential interpretations
- Target groups or individuals
- Potential harm or offense

Conclude with whether this meme should be classified as hateful content and provide strong justification for your decision.
"""

prompt_v4_step_by_step = """
Analyze this meme step by step to determine if it contains hateful content:

Step 1: Describe what you see in the image
Step 2: Analyze the text content
Step 3: Identify any concerning elements (symbols, language, implications)
Step 4: Consider the overall message and intent
Step 5: Make a final determination (hateful/non-hateful) with reasoning

Be thorough in your analysis and explain your reasoning clearly.
"""

prompt_v5_policy_focused = """
As a content policy expert, evaluate this meme against hate speech guidelines:

Evaluation criteria:
- Does it attack individuals or groups based on protected characteristics?
- Does it promote discrimination, hostility, or violence?
- Does it use derogatory language or symbols?
- Does it perpetuate harmful stereotypes?

Provide a clear policy violation assessment with specific examples from the content.
"""

# Collection of all prompts for DPO generation
DPO_PROMPTS = {
    "structured_analysis": prompt_v1,
    "classification": prompt_v2_classification,
    "detailed_analysis": prompt_v3_detailed_analysis,
    "step_by_step": prompt_v4_step_by_step,
    "policy_focused": prompt_v5_policy_focused
}

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Supported models with their configurations
MODEL_CONFIGS = {
    "qwen2.5-vl-7b": {
        "hf_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_type": "qwen2.5-vl",
        "supports_flash_attention": True,
        "recommended_dtype": torch.bfloat16,
        "recommended_device": "auto"
    },
    "qwen2.5-vl-3b": {
        "hf_model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_type": "qwen2.5-vl",
        "supports_flash_attention": True,
        "recommended_dtype": torch.bfloat16,
        "recommended_device": "auto"
    },
    "qwen2.5-vl-72b": {
        "hf_model_name": "Qwen/Qwen2.5-VL-72B-Instruct",
        "model_type": "qwen2.5-vl",
        "supports_flash_attention": True,
        "recommended_dtype": torch.bfloat16,
        "recommended_device": "auto"
    },
    "llava-1.5-7b": {
        "hf_model_name": "llava-hf/llava-1.5-7b-hf",
        "model_type": "llava",
        "supports_flash_attention": False,
        "recommended_dtype": torch.float16,
        "recommended_device": "auto"
    },
    "llava-1.5-13b": {
        "hf_model_name": "llava-hf/llava-1.5-13b-hf",
        "model_type": "llava",
        "supports_flash_attention": False,
        "recommended_dtype": torch.float16,
        "recommended_device": "auto"
    }
}

# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

# Default generation parameters
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 3
}

# Conservative generation parameters (more focused responses)
CONSERVATIVE_GENERATION_PARAMS = {
    "max_new_tokens": 256,
    "temperature": 0.3,
    "do_sample": True,
    "top_p": 0.8,
    "top_k": 30,
    "repetition_penalty": 1.05,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 2
}

# Creative generation parameters (more diverse responses)
CREATIVE_GENERATION_PARAMS = {
    "max_new_tokens": 768,
    "temperature": 0.9,
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 100,
    "repetition_penalty": 1.15,
    "length_penalty": 0.9,
    "no_repeat_ngram_size": 4
}

# =============================================================================
# DPO GENERATION SETTINGS
# =============================================================================

DPO_CONFIG = {
    # Number of responses to generate per prompt
    "responses_per_prompt": 3,
    
    # Which prompts to use for generation
    "active_prompts": ["structured_analysis", "classification", "detailed_analysis"],
    
    # Generation parameter sets to use
    "generation_param_sets": ["default", "conservative", "creative"],
    
    # Whether to generate responses with different parameters
    "use_multiple_param_sets": True,
    
    # Batch size for processing
    "batch_size": 1,
    
    # Delay between generations (seconds)
    "generation_delay": 0.1
}

# =============================================================================
# QWEN2.5-VL SPECIFIC CONFIGURATIONS
# =============================================================================

QWEN_VL_CONFIG = {
    # Vision processing parameters
    "min_pixels": 256 * 28 * 28,  # Minimum image resolution
    "max_pixels": 1280 * 28 * 28,  # Maximum image resolution
    
    # Flash attention settings
    "use_flash_attention": True,
    
    # Processing parameters
    "padding": True,
    "return_tensors": "pt"
}

# =============================================================================
# FILE AND PATH CONFIGURATIONS
# =============================================================================

DEFAULT_PATHS = {
    "base_dir": "./data/gt/",
    "image_base": "./data/image/",
    "output_dir": "./output/dpo_responses/",
    "cache_dir": "./cache/",
    "checkpoint_dir": "./checkpoints/"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name, {})

def get_generation_params(param_set: str = "default") -> dict:
    """Get generation parameters for a specific set."""
    param_sets = {
        "default": DEFAULT_GENERATION_PARAMS,
        "conservative": CONSERVATIVE_GENERATION_PARAMS,
        "creative": CREATIVE_GENERATION_PARAMS
    }
    return param_sets.get(param_set, DEFAULT_GENERATION_PARAMS)

def get_active_prompts() -> dict:
    """Get currently active prompts for DPO generation."""
    active_prompt_names = DPO_CONFIG["active_prompts"]
    return {name: DPO_PROMPTS[name] for name in active_prompt_names if name in DPO_PROMPTS}

def get_all_prompts() -> dict:
    """Get all available prompts."""
    return DPO_PROMPTS.copy()

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_model_config(model_name: str) -> bool:
    """Validate if model configuration exists."""
    return model_name in MODEL_CONFIGS

def validate_prompt_name(prompt_name: str) -> bool:
    """Validate if prompt name exists."""
    return prompt_name in DPO_PROMPTS