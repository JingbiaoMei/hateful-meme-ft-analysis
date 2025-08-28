#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hm-ana

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
INPUT_DATA="data/gt/FB/train.jsonl"
IMAGE_BASE_PATH="data/image/FB/All"
OUTPUT_FILE="dpo_data/FB/FB_qwen25vl3b.jsonl"
NUM_RESPONSES=16
# Run the vLLM DPO generation script
python src/sampling/generate_dpo_data_vllm.py \
    --model_name "$MODEL_NAME" \
    --input_data "$INPUT_DATA" \
    --output_file "$OUTPUT_FILE" \
    --generation_log_file ./generation_log_3b.json \
    --image_base_path "$IMAGE_BASE_PATH" \
    --num_responses $NUM_RESPONSES \
    --max_entries 50 \
    --batch_size 400 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --max_new_tokens 512 \
    --trust_remote_code