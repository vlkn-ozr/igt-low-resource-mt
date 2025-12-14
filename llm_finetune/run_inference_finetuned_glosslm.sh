#!/bin/bash

MODEL_PATH=${1:-"./output_qwen25_glosslm"}
BASE_MODEL=${2:-"Qwen/Qwen2.5-7B-Instruct"}
TEMPLATES=${3:-"zero_shot"}

echo "Using adapter model at: $MODEL_PATH"
echo "Using base model: $BASE_MODEL"
echo "Using templates: $TEMPLATES"

python run_tests_glosslm.py \
  --model finetuned_qwen_glosslm \
  --custom-model-path "$MODEL_PATH" \
  --only-chrf \
  --template $TEMPLATES

echo "Inference completed!" 
