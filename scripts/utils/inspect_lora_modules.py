#!/usr/bin/env python3
"""
Script to inspect LoRA checkpoint and show which modules are fine-tuned.
This helps determine if vision tower was included in LoRA training.
"""

import argparse
import os
import torch
from safetensors import safe_open
import json
from pathlib import Path

def load_adapter_config(checkpoint_path):
    """Load adapter configuration if available."""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def inspect_safetensors_lora(checkpoint_path):
    """Inspect LoRA modules from safetensors format."""
    safetensors_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    
    if not os.path.exists(safetensors_path):
        print(f"❌ No adapter_model.safetensors found in {checkpoint_path}")
        return None
    
    print(f"📁 Loading LoRA checkpoint from: {safetensors_path}")
    
    # Load safetensors file
    modules = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if ".lora_A." in key or ".lora_B." in key:
                # Extract module name (remove .lora_A/B.default suffix)
                module_name = key.replace(".lora_A.default", "").replace(".lora_B.default", "")
                if module_name not in modules:
                    modules[module_name] = {"lora_A": False, "lora_B": False, "shape": None}
                
                if ".lora_A." in key:
                    modules[module_name]["lora_A"] = True
                    modules[module_name]["shape"] = f.get_tensor(key).shape
                elif ".lora_B." in key:
                    modules[module_name]["lora_B"] = True
    
    return modules

def inspect_pytorch_lora(checkpoint_path):
    """Inspect LoRA modules from PyTorch format."""
    pytorch_path = os.path.join(checkpoint_path, "adapter_model.bin")
    
    if not os.path.exists(pytorch_path):
        print(f"❌ No adapter_model.bin found in {checkpoint_path}")
        return None
    
    print(f"📁 Loading LoRA checkpoint from: {pytorch_path}")
    
    # Load PyTorch checkpoint
    checkpoint = torch.load(pytorch_path, map_location="cpu")
    
    modules = {}
    for key in checkpoint.keys():
        if ".lora_A." in key or ".lora_B." in key:
            # Extract module name
            module_name = key.replace(".lora_A.default", "").replace(".lora_B.default", "")
            if module_name not in modules:
                modules[module_name] = {"lora_A": False, "lora_B": False, "shape": None}
            
            if ".lora_A." in key:
                modules[module_name]["lora_A"] = True
                modules[module_name]["shape"] = checkpoint[key].shape
            elif ".lora_B." in key:
                modules[module_name]["lora_B"] = True
    
    return modules

def categorize_modules(modules):
    """Categorize modules by component type."""
    categories = {
        "vision_tower": [],
        "language_model": [],
        "multimodal_projector": [],
        "other": []
    }
    
    for module_name in modules.keys():
        if "visual" in module_name or "vision" in module_name or "image" in module_name:
            categories["vision_tower"].append(module_name)
        elif "language_model" in module_name or "model.layers" in module_name or "lm_head" in module_name:
            categories["language_model"].append(module_name)
        elif "multi_modal_projector" in module_name or "projector" in module_name:
            categories["multimodal_projector"].append(module_name)
        else:
            categories["other"].append(module_name)
    
    return categories

def print_analysis(modules, config=None):
    """Print detailed analysis of LoRA modules."""
    if not modules:
        print("❌ No LoRA modules found!")
        return
    
    print(f"\n📊 LoRA Analysis Summary")
    print("=" * 50)
    
    # Print adapter config if available
    if config:
        print(f"📝 Adapter Configuration:")
        print(f"   - LoRA Rank (r): {config.get('r', 'N/A')}")
        print(f"   - LoRA Alpha: {config.get('lora_alpha', 'N/A')}")
        print(f"   - LoRA Dropout: {config.get('lora_dropout', 'N/A')}")
        print(f"   - Target Modules: {config.get('target_modules', 'N/A')}")
        print(f"   - Task Type: {config.get('task_type', 'N/A')}")
        print()
    
    # Categorize modules
    categories = categorize_modules(modules)
    
    print(f"🔍 Total LoRA Modules Found: {len(modules)}")
    print()
    
    # Print by category
    for category, module_list in categories.items():
        if module_list:
            print(f"🎯 {category.upper().replace('_', ' ')} ({len(module_list)} modules):")
            for module in sorted(module_list):
                shape = modules[module].get("shape", "Unknown")
                print(f"   ✓ {module} (shape: {shape})")
            print()
    
    # Vision tower analysis
    vision_modules = categories["vision_tower"]
    if vision_modules:
        print("🖼️  VISION TOWER IS FINE-TUNED!")
        print(f"   Found {len(vision_modules)} vision-related LoRA modules")
    else:
        print("🚫 VISION TOWER IS NOT FINE-TUNED")
        print("   No vision-related LoRA modules found")
    
    print()
    
    # Detailed module list
    print(f"📋 All LoRA Modules ({len(modules)} total):")
    print("-" * 50)
    for i, (module_name, info) in enumerate(sorted(modules.items()), 1):
        shape = info.get("shape", "Unknown")
        print(f"{i:2d}. {module_name}")
        print(f"     Shape: {shape}")

def main():
    parser = argparse.ArgumentParser(description="Inspect LoRA checkpoint modules")
    parser.add_argument("checkpoint_path", help="Path to LoRA checkpoint directory")
    parser.add_argument("--format", choices=["auto", "safetensors", "pytorch"], default="auto",
                       help="Checkpoint format (default: auto-detect)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint path does not exist: {args.checkpoint_path}")
        return
    
    print(f"🔍 Inspecting LoRA checkpoint: {args.checkpoint_path}")
    print("=" * 60)
    
    # Load adapter config
    config = load_adapter_config(args.checkpoint_path)
    
    # Try to load modules
    modules = None
    
    if args.format == "auto" or args.format == "safetensors":
        modules = inspect_safetensors_lora(args.checkpoint_path)
    
    if modules is None and (args.format == "auto" or args.format == "pytorch"):
        modules = inspect_pytorch_lora(args.checkpoint_path)
    
    if modules is None:
        print("❌ Could not find adapter_model.safetensors or adapter_model.bin")
        print("   Make sure the checkpoint path contains LoRA weights")
        return
    
    # Print analysis
    print_analysis(modules, config)

if __name__ == "__main__":
    main()
