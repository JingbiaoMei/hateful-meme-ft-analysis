#!/bin/bash

echo "🔍 Checking LoRA modules in all checkpoints..."
echo "=============================================="

# 3B model checkpoints
echo ""
echo "🤖 QWEN 2.5-VL 3B CHECKPOINTS:"
echo "=============================="

checkpoints_3b=(
    "checkpoints/fb/qwen2_5vl_3b/dpo-rank64-beta0.1"
    "checkpoints/fb/qwen2_5vl_3b/dpo-beta0.1"
    "checkpoints/fb/qwen2_5vl_3b/dpo-beta0.3"
    "checkpoints/fb/qwen2_5vl_3b/dpo-beta0.5"
    "checkpoints/fb/qwen2_5vl_3b/dpo-beta0.7"
    "checkpoints/fb/qwen2_5vl_3b/dpo-beta0.9"
)

for checkpoint in "${checkpoints_3b[@]}"; do
    if [ -d "$checkpoint" ]; then
        echo ""
        echo "📁 Checking: $checkpoint"
        echo "----------------------------------------"
        python scripts/utils/inspect_lora_modules.py "$checkpoint" | grep -E "(VISION TOWER|LoRA Analysis Summary|Total LoRA Modules|vision_tower|📁 Loading)"
    else
        echo "❌ Not found: $checkpoint"
    fi
done

# 7B model checkpoints
echo ""
echo ""
echo "🤖 QWEN 2.5-VL 7B CHECKPOINTS:"
echo "=============================="

checkpoints_7b=(
    "checkpoints/fb/qwen2_5vl_7b/dpo-rank64-beta0.1"
    "checkpoints/fb/qwen2_5vl_7b/dpo-beta0.1"
    "checkpoints/fb/qwen2_5vl_7b/dpo-beta0.3"
    "checkpoints/fb/qwen2_5vl_7b/dpo-beta0.5"
    "checkpoints/fb/qwen2_5vl_7b/dpo-beta0.7"
    "checkpoints/fb/qwen2_5vl_7b/dpo-beta0.9"
)

for checkpoint in "${checkpoints_7b[@]}"; do
    if [ -d "$checkpoint" ]; then
        echo ""
        echo "📁 Checking: $checkpoint"
        echo "----------------------------------------"
        python scripts/utils/inspect_lora_modules.py "$checkpoint" | grep -E "(VISION TOWER|LoRA Analysis Summary|Total LoRA Modules|vision_tower|📁 Loading)"
    else
        echo "❌ Not found: $checkpoint"
    fi
done

echo ""
echo "✅ Checkpoint inspection completed!"
echo ""
echo "💡 To see full details for a specific checkpoint, run:"
echo "   python scripts/utils/inspect_lora_modules.py <checkpoint_path>"
