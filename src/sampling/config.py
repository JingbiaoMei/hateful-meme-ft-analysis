
"""
Configuration file for DPO dataset generation using Hugging Face models.
Contains prompts, model configurations, and sampling parameters.
"""

import torch

# =============================================================================
# PROMPTS FOR DPO GENERATION
# =============================================================================

# Original prompt for structured analysis
prompt = """<image>Is it hateful? Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The answer should be Yes or No, if the meme is hateful than output Yes ; if the meme is Benign than output No. The output format should be as follows:
<think> ... </think> <answer>...</answer>
Please strictly follow the format."""