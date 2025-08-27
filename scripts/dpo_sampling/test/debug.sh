python src/sampling/generate_dpo_data.py \
    --input_data data/gt/MAMI/train.jsonl \
    --image_base_path data/image/MAMI/All \
    --output_file dpo_data/test/test_mami_dpo.jsonl \
    --max_entries 10 \
    --num_responses 8 \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --temperature 0.7 \
    --max_new_tokens 150 \
    --max_pixels 401408