#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export RANK=-1
export LOCAL_RANK=-1
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export CUDA_MEMORY_FRACTION=0.9
export MAX_SPLIT_SIZE_MB=512

DATASET_PATH="processed/gloss_translation_dataset.jsonl"
if [ -f "$DATASET_PATH" ]; then
    echo "Dataset file found: $(du -h $DATASET_PATH | cut -f1)"
else
    echo "WARNING: Dataset file not found at $DATASET_PATH"
fi

# Print environment for debugging
echo "Running with environment:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

: '
long 56hrs
python finetune_qwen25.py \
    --dataset_path processed/gloss_translation_dataset_glosslm.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./output_qwen25_glosslm \
    --num_train_epochs 4 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --max_seq_length 256
'

: '
short 30hrs 
python finetune_qwen25.py \
    --dataset_path processed/gloss_translation_dataset_glosslm.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./output_qwen25_glosslm \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --max_seq_length 256
'

: '
medium 49hrs
'
# Run the fine-tuning script with lower batch size for stability
python finetune_qwen25.py \
    --dataset_path processed/gloss_translation_dataset_glosslm.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./output_qwen25_glosslm \
    --resume_from_checkpoint ./output_qwen25_glosslm/checkpoint-9500 \
    --num_train_epochs 3 \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 3 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 256
