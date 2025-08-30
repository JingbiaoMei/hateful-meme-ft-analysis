model_name=qwen2_5vl_3b
dataset_name=hm
current_date=$(date +'%Y-%m-%d')
export WANDB_PROJECT="LLAMAFACTORY_hateful_DPO"

export WANDB_RUN_GROUP="Finetuning_${dataset_name}_${model_name}_${current_date}"

export WANDB_NAME=${dataset_name}_${model_name}_dpo-rank64-beta0.1
llamafactory-cli train scripts/dpo/${dataset_name}/${model_name}_lora_dpo.yaml \
    lora_rank=64 \
    lora_alpha=128 \
    output_dir=checkpoints/fb/${model_name}/dpo-rank64-beta0.1

export WANDB_NAME=${dataset_name}_${model_name}_dpo-beta0.1-ep3
llamafactory-cli train scripts/dpo/${dataset_name}/${model_name}_lora_dpo.yaml \
    output_dir=checkpoints/fb/${model_name}/dpo-beta0.1 \
    pref_beta=0.1 \
    num_train_epochs=3.0

export WANDB_NAME=${dataset_name}_${model_name}_dpo-beta0.1
llamafactory-cli train scripts/dpo/${dataset_name}/${model_name}_lora_dpo.yaml \
    output_dir=checkpoints/fb/${model_name}/dpo-beta0.1 \
    pref_beta=0.1



export WANDB_NAME=${dataset_name}_${model_name}_dpo-beta0.3
llamafactory-cli train scripts/dpo/${dataset_name}/${model_name}_lora_dpo.yaml \
    output_dir=checkpoints/fb/${model_name}/dpo-beta0.3 \
    pref_beta=0.3

export WANDB_NAME=${dataset_name}_${model_name}_dpo-beta0.5
llamafactory-cli train scripts/dpo/${dataset_name}/${model_name}_lora_dpo.yaml \
    output_dir=checkpoints/fb/${model_name}/dpo-beta0.5 \
    pref_beta=0.5

export WANDB_NAME=${dataset_name}_${model_name}_dpo-beta0.7
llamafactory-cli train scripts/dpo/${dataset_name}/${model_name}_lora_dpo.yaml \
    output_dir=checkpoints/fb/${model_name}/dpo-beta0.7 \
    pref_beta=0.7

export WANDB_NAME=${dataset_name}_${model_name}_dpo-beta0.9
llamafactory-cli train scripts/dpo/${dataset_name}/${model_name}_lora_dpo.yaml \
    output_dir=checkpoints/fb/${model_name}/dpo-beta0.9 \
    pref_beta=0.9