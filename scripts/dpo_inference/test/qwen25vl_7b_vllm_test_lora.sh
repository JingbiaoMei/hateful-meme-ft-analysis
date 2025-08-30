#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hm-ana
export CUDA_VISIBLE_DEVICES=1
python src/inference/qwen2_hm_inference_vllm.py \
    --base_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --model_path checkpoints/fb/qwen2_5vl-3b/dpo \
    --processor_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --data_split "dev_seen test_seen" \
    --dataset "FB" \
    --batch_size 1000 \
    --log_name "qwen25vl_3b_vllm_batch"