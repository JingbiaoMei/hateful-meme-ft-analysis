#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
INPUT_DATA="data/gt/MAMI/train.jsonl"
IMAGE_BASE_PATH="data/image/MAMI/All"
OUTPUT_FILE="dpo_data/test/test_mami_dpo.jsonl"
NUM_RESPONSES=8
# Run the vLLM DPO generation script
python src/sampling/generate_dpo_data_vllm.py \
    --model_name "$MODEL_NAME" \
    --input_data "$INPUT_DATA" \
    --output_file "$OUTPUT_FILE" \
    --image_base_path "$IMAGE_BASE_PATH" \
    --num_responses $NUM_RESPONSES \
    --max_entries 50 \
    --batch_size 32 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --max_new_tokens 512 \
    --trust_remote_code

echo ""
echo "=== Test completed ==="

# Check if output file was created
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Output file created: $OUTPUT_FILE"
    echo "✓ File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "✓ First few lines:"
    head -20 "$OUTPUT_FILE"
else
    echo "✗ Output file not created"
    exit 1
fi

echo ""
echo "=== vLLM DPO generation test successful ==="
