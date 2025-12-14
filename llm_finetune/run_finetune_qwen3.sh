#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export RANK=-1
export LOCAL_RANK=-1
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

python finetune_qwen3.py "$@" 