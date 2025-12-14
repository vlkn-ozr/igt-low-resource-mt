#!/usr/bin/env python
"""Wrapper script to run fine-tuning with proper environment variables set."""
import os
import sys
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

args = sys.argv[1:] if len(sys.argv) > 1 else []

cmd = [sys.executable, "finetune_qwen3.py"] + args
print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd, check=True) 