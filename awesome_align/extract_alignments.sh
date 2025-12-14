#!/bin/bash

export KMP_DUPLICATE_LIB_OK=TRUE

DATA_FILE=${1:-"awesome_align_lmm_dict_2k_data.txt"}
MODEL_NAME_OR_PATH=${2:-"awesome_align_model_llm_dict_2k"}
OUTPUT_FILE=${3:-"alignments_llm_dict_2k_output.txt"}

awesome-align \
    --output_file="$OUTPUT_FILE" \
    --model_name_or_path="$MODEL_NAME_OR_PATH" \
    --data_file="$DATA_FILE" \
    --extraction "softmax" \
    --batch_size 32
