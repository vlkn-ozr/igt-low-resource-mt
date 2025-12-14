#!/usr/bin/env python3
"""
Complete pipeline for Turkish-English word alignment finetuning.
Runs data processing, training, and provides inference examples.
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    result = subprocess.run(command, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        return False
    
    print(f"✓ {description} completed successfully")
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        "parallel_data_40k/merged_transcriptions_40k.txt",
        "parallel_data_40k/merged_translations_40k.txt", 
        "dict_40k.txt",
        "glosslm_lemma.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✓ All required data files found")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Turkish-English word alignment finetuning pipeline")
    parser.add_argument("--step", choices=["data", "train", "inference", "all"], 
                       default="all", help="Which step to run")
    parser.add_argument("--max_examples", type=int, default=2000,
                       help="Maximum number of training examples to process")
    parser.add_argument("--config", type=str, default="configs/config_example.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    print("Turkish-English Word Alignment Finetuning Pipeline")
    print("=" * 60)
    
    if not check_data_files():
        print("\nPlease ensure all required data files are in the correct locations.")
        return 1
    
    success = True
    
    if args.step in ["data", "all"]:
        if not run_command(
            f"python src/data_processor.py --max_examples {args.max_examples}",
            "Data processing and prompt generation"
        ):
            success = False
    
    if args.step in ["train", "all"] and success:
        if not run_command(
            f"python src/trainer.py --config {args.config}",
            "Model finetuning with PEFT/LoRA"
        ):
            success = False
    
    if args.step in ["inference", "all"] and success:
        model_path = "./outputs/qwen-alignment-model"
        if os.path.exists(model_path):
            if not run_command(
                f"python src/inference.py --model_path {model_path} --config {args.config}",
                "Running inference examples"
            ):
                success = False
        else:
            print(f"Warning: Model not found at {model_path}. Skipping inference.")
    
    print(f"\n{'='*60}")
    if success:
        print("✓ Pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Check training logs in ./outputs/qwen-alignment-model/")
        print("2. Run inference with: python src/inference.py --model_path ./outputs/qwen-alignment-model --interactive")
        print("3. Use the model for word alignment tasks")
    else:
        print("✗ Pipeline failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 