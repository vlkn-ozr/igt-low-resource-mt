#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Run the BPE preprocessing and training pipeline')
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help='Vocabulary size for SentencePiece model')
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram'],
                        help='SentencePiece model type (bpe or unigram)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--experiment_name', type=str, default='bpe_training',
                        help='Name of the experiment for logging')
    parser.add_argument('--monitor_resources', action='store_true',
                        help='Monitor system resources during training')
    return parser.parse_args()

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"Running {description}...")
    print(f"Command: {command}")
    print(f"{'='*80}\n")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: {description} failed with return code {process.returncode}")
            sys.exit(process.returncode)
        
        print(f"\n{'='*80}")
        print(f"{description} completed successfully!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    args = parse_args()

    train_cmd = (
        f"python {os.path.join(current_dir, 'train_onmt.py')} "
        f"--gpu {args.gpu} "
        f"--experiment_name {args.experiment_name}_bpe_{args.vocab_size} "
        f"--config {os.path.join(project_root, 'configs', 'config_example.yaml')} "
    )
    
    if args.monitor_resources:
        train_cmd += " --monitor_resources"
    
    run_command(train_cmd, "Model Training with BPE")

if __name__ == '__main__':
    main() 