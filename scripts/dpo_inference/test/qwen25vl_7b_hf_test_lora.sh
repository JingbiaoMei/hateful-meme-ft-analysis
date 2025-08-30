#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hm-ana
export CUDA_VISIBLE_DEVICES=1

python src/inference/qwen2_hm_inference_multiprocess.py \
    --base_model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --model_path checkpoints/fb/qwen2_5vl-7b/dpo \
    --processor_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --batch_size 2 \
    --data_split dev_seen \
    --dataset "FB" \
    --log_name qwen25vl_7b_lora_test_hf
            