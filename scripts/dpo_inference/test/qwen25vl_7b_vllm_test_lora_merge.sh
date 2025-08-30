#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hm-ana
export CUDA_VISIBLE_DEVICES=0
python src/inference/qwen2_hm_inference_vllm.py \
    --model_path checkpoints/fb/qwen2_5vl-7b/dpo/merge \
    --processor_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --data_split "dev_seen test_seen" \
    --dataset "FB" \
    --batch_size 1000 \
    --log_name "qwen25vl_7b_lora_test_merge"