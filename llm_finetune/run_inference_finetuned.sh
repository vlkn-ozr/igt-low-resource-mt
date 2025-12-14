#!/bin/bash

MODEL_PATH=${1:-"./output"}
BASE_MODEL=${2:-"Qwen/Qwen3-8B"}
LIMIT=${3:-""}

LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
  LIMIT_FLAG="--limit $LIMIT"
fi

echo "Using adapter model at: $MODEL_PATH"
echo "Using base model: $BASE_MODEL"

if [ -f "$MODEL_PATH/adapter_config.json" ]; then
  python run_single_inference.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --input_text "spending-V-VN:INF-NOM-PL 1,6-NUM-ARA-PERC increase-V-PST-3S"
else
  python run_tests.py \
    --model finetuned \
    --custom-model-path "$MODEL_PATH" \
    --template zero_shot \
    $LIMIT_FLAG
fi

echo "Inference completed!" 