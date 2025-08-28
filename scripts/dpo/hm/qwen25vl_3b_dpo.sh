model_name=qwen2_5vl_3b
dataset_name=hm
current_date=$(date +'%Y-%m-%d')
export WANDB_PROJECT="LLAMAFACTORY_hateful_DPO"
export WANDB_NAME=$Name
export WANDB_RUN_GROUP="Finetuning_${model_name}_${dataset_name}_${current_date}"


llamafactory-cli train scripts/dpo/${dataset_name}/${model_name}_lora_dpo.yaml