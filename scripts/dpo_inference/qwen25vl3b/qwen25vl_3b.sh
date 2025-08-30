#!/bin/bash

# List of all experiments to run
experiments=(
    "dpo-rank64-beta0.1"
    "dpo-beta0.1"
    "dpo-beta0.3"
    "dpo-beta0.5"
    "dpo-beta0.7"
    "dpo-beta0.9"
)

# Run inference for each experiment
for exp in "${experiments[@]}"; do
    echo "Running inference for experiment: $exp"
    
    python src/inference/qwen2_hm_inference_vllm.py \
        --base_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
        --model_path "checkpoints/fb/qwen2_5vl_3b/$exp" \
        --processor_path "Qwen/Qwen2.5-VL-3B-Instruct" \
        --data_split "dev_seen test_seen" \
        --dataset "FB" \
        --batch_size 1000 \
        --log_name "qwen25vl_3b_${exp}"
    
    echo "Completed inference for experiment: $exp"
    echo "----------------------------------------"
done

echo "All experiments completed!"