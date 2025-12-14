#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys
import re
import sacrebleu
from typing import List
from nmt_logger import NMTLogger

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate NMT model translations')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference file (ground truth translations)')
    parser.add_argument('--hypothesis', type=str, required=True,
                        help='Path to hypothesis file (model output)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input file (source/gloss)')
    parser.add_argument('--log_dir', type=str, default=os.path.join(project_root, 'logs_rnn_small_multi_100k_plus_1200_sampled_setimes_no_pos_preset_eval_tur', 'fieldwork', 'kara', 'eval'),
                        help='Directory to save logs (default: logs/eval)')
    parser.add_argument('--experiment_name', type=str, default='onmt_evaluation',
                        help='Name of the experiment for logging')
    parser.add_argument('--reference_type', type=str, choices=['raw', 'tokenized'], default='raw',
                        help='Type of reference (raw or tokenized with SentencePiece)')
    return parser.parse_args()

def read_file(file_path: str) -> List[str]:
    """Read a file and return a list of lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def calculate_bleu(references: List[str], hypotheses: List[str]):
    """Calculate BLEU score."""
    try:
        return sacrebleu.corpus_bleu(hypotheses, [references])
    except TypeError:
        # For older versions of sacrebleu
        return sacrebleu.corpus_bleu(hypotheses, [references], force=True)

def calculate_bleu_n(references: List[str], hypotheses: List[str], n: int):
    """Calculate BLEU-n score (n=1,2,3)."""
    # For older versions of sacrebleu
    if n == 1:
        weights = (1, 0, 0, 0)
    elif n == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n == 3:
        weights = (0.33, 0.33, 0.33, 0)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)  # Default BLEU-4
    
    try:
        # Try with newer versions of sacrebleu
        return sacrebleu.corpus_bleu(hypotheses, [references], weights=weights)
    except TypeError:
        # For older versions of sacrebleu
        return sacrebleu.corpus_bleu(hypotheses, [references], force=True, 
                                    smooth_method='floor', smooth_value=0.01, 
                                    tokenize='none', weights=weights)

def main():
    args = parse_args()
    
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = NMTLogger(args.experiment_name, log_dir)
    
    logger.log_info(f"Starting evaluation: {args.experiment_name}")
    logger.log_info(f"Reference file: {args.reference}")
    logger.log_info(f"Hypothesis file: {args.hypothesis}")
    logger.log_info(f"Input file: {args.input}")
    
    references = read_file(args.reference)
    hypotheses = read_file(args.hypothesis)
    inputs = read_file(args.input)
    
    if len(references) != len(hypotheses) or len(references) != len(inputs):
        logger.log_warning(f"Warning: Files have different number of lines: references={len(references)}, hypotheses={len(hypotheses)}, inputs={len(inputs)}")
        min_len = min(len(references), len(hypotheses), len(inputs))
        references = references[:min_len]
        hypotheses = hypotheses[:min_len]
        inputs = inputs[:min_len]
        logger.log_warning(f"Truncating to {min_len} lines for evaluation")
    
    logger.log_info("\n===== Translation Evaluation Results =====\n")
    evaluation_metrics = {}
    detailed_metrics = {}
    
    try:
        bleu = calculate_bleu(references, hypotheses)
        bleu_score = bleu.score
        evaluation_metrics['bleu'] = bleu_score
        detailed_metrics['bleu'] = bleu
        
        try:
            bleu1 = calculate_bleu_n(references, hypotheses, 1)
            bleu2 = calculate_bleu_n(references, hypotheses, 2)
            bleu3 = calculate_bleu_n(references, hypotheses, 3)
            evaluation_metrics['bleu1'] = bleu1.score
            evaluation_metrics['bleu2'] = bleu2.score
            evaluation_metrics['bleu3'] = bleu3.score
            detailed_metrics['bleu1'] = bleu1
            detailed_metrics['bleu2'] = bleu2
            detailed_metrics['bleu3'] = bleu3
            
            logger.log_info(f"BLEU: {bleu_score:.2f}")
            logger.log_info(f"BLEU-1: {bleu1.score:.2f}")
            logger.log_info(f"BLEU-2: {bleu2.score:.2f}")
            logger.log_info(f"BLEU-3: {bleu3.score:.2f}")
            logger.log_metrics({'bleu': bleu_score, 'bleu1': bleu1.score, 'bleu2': bleu2.score, 'bleu3': bleu3.score})
        except Exception as e:
            logger.log_warning(f"Error calculating BLEU-n scores: {e}")
            logger.log_info(f"BLEU: {bleu_score:.2f}")
            logger.log_metrics({'bleu': bleu_score})
    except Exception as e:
        logger.log_error(f"Error calculating BLEU score: {e}")
    
    results_file = os.path.join(log_dir, f"evaluation_results_{args.experiment_name}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for {args.experiment_name}\n")
        f.write(f"Reference file: {args.reference}\n")
        f.write(f"Reference type: {args.reference_type}\n")
        f.write(f"Hypothesis file: {args.hypothesis}\n")
        f.write(f"Input file: {args.input}\n\n")
        
        for metric_name, metric_value in evaluation_metrics.items():
            f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
            
            if metric_name in detailed_metrics:
                f.write(f"{metric_name} details: {detailed_metrics[metric_name]}\n\n")
    
    logger.log_info(f"Evaluation results saved to {results_file}")
    
    num_samples = min(5, len(references))
    logger.log_info(f"\n===== Sample Translations =====\n")
    
    for i in range(num_samples):
        logger.log_info(f"Example {i+1}:")
        logger.log_info(f"Input: {inputs[i]}")
        logger.log_info(f"Reference: {references[i]}")
        logger.log_info(f"Hypothesis: {hypotheses[i]}")

if __name__ == '__main__':
    main()