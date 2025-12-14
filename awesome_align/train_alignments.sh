#!/bin/bash

export KMP_DUPLICATE_LIB_OK=TRUE

TRAIN_FILE=${1:-"tr_en_parallel_clean_60k.txt"}
EVAL_FILE=${2:-"sample_corpora.txt"}
OUTPUT_DIR=${3:-"awesome_align_model"}
MODEL_NAME=${4:-"bert-base-multilingual-cased"}

mkdir -p $OUTPUT_DIR

echo "Starting training..."
awesome-train \
    --output_dir=$OUTPUT_DIR \
    --model_name_or_path=$MODEL_NAME \
    --extraction 'softmax' \
    --do_train \
    --train_tlm \
    --train_so \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --save_steps 4000 \
    --max_steps 20000 \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
    --encoding 'utf-8' \
    --num_workers 2 