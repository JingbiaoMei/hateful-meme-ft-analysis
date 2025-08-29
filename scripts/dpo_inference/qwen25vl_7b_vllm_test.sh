#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hm-ana
export CUDA_VISIBLE_DEVICES=0
python src/inference/qwen2_hm_inference_vllm.py \
    --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --processor_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --data_split "dev_seen test_seen" \
    --dataset "FB" \
    --batch_size 8 \
    --log_name "qwen25vl_7b_vllm"