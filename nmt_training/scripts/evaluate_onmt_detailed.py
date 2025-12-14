#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys
import json
import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sacrebleu
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

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
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'logs_rnn_small_multi_100k_plus_1200_sampled_setimes_no_pos_preset_eval_tur', 'fieldwork', 'kara', 'eval'), help='Directory to save evaluation results (default: logs/eval)')
    parser.add_argument('--model_name', type=str, default='nmt_model',
                        help='Name of the model being evaluated (for reporting)')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of random translation samples to include in the report')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='Path to previous evaluation results JSON for comparison')
    parser.add_argument('--bootstrap_samples', type=int, default=1000,
                        help='Number of bootstrap samples for confidence intervals (0 to disable)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
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

def calculate_bleu(references: List[str], hypotheses: List[str]) -> sacrebleu.metrics.bleu.BLEUScore:
    """Calculate BLEU score."""
    return sacrebleu.corpus_bleu(hypotheses, [references])

def calculate_sentence_bleu(references: List[str], hypotheses: List[str]) -> List[float]:
    """Calculate BLEU score for each sentence."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = sacrebleu.sentence_bleu(hyp, [ref]).score
        scores.append(score)
    return scores

def calculate_length_ratio(references: List[str], hypotheses: List[str]) -> float:
    """Calculate the average length ratio between hypotheses and references."""
    ref_lengths = [len(ref.split()) for ref in references]
    hyp_lengths = [len(hyp.split()) for hyp in hypotheses]
    
    if sum(ref_lengths) == 0:
        return 0.0
    
    return sum(hyp_lengths) / sum(ref_lengths)

def bootstrap_resample(references: List[str], hypotheses: List[str], 
                      metric_fn, num_samples: int = 1000, 
                      sample_ratio: float = 1.0, seed: int = 42) -> Tuple[float, float, float]:
    """
    Perform bootstrap resampling to estimate confidence intervals.
    
    Args:
        references: List of reference translations
        hypotheses: List of hypothesis translations
        metric_fn: Function that calculates the metric
        num_samples: Number of bootstrap samples
        sample_ratio: Ratio of data to sample each time
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (lower_bound, mean, upper_bound) for 95% confidence interval
    """
    if num_samples <= 0:
        score = metric_fn(references, hypotheses).score
        return (score, score, score)
    
    random.seed(seed)
    n = len(references)
    sample_size = int(n * sample_ratio)
    scores = []
    
    for _ in range(num_samples):
        indices = [random.randint(0, n-1) for _ in range(sample_size)]
        sampled_refs = [references[i] for i in indices]
        sampled_hyps = [hypotheses[i] for i in indices]
        score = metric_fn(sampled_refs, sampled_hyps).score
        scores.append(score)
    
    scores.sort()
    lower_idx = int(num_samples * 0.025)
    upper_idx = int(num_samples * 0.975)
    
    return (scores[lower_idx], sum(scores) / len(scores), scores[upper_idx])

def analyze_by_length(references: List[str], hypotheses: List[str], 
                     metric_fn, bin_size: int = 10) -> Dict[str, float]:
    """Analyze performance by reference sentence length."""
    length_bins = defaultdict(list)
    
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        ref_len = len(ref.split())
        bin_idx = ref_len // bin_size
        length_bins[bin_idx].append((ref, hyp))
    
    results = {}
    for bin_idx, pairs in sorted(length_bins.items()):
        min_len = bin_idx * bin_size
        max_len = (bin_idx + 1) * bin_size - 1
        bin_refs = [p[0] for p in pairs]
        bin_hyps = [p[1] for p in pairs]
        
        if len(bin_refs) > 1:
            score = metric_fn(bin_refs, bin_hyps).score
            results[f"{min_len}-{max_len}"] = {
                "score": score,
                "count": len(bin_refs)
            }
    
    return results

def save_results_json(results: Dict[str, Any], output_path: str):
    """Save evaluation results to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

def save_results_csv(results: Dict[str, Any], output_path: str):
    """Save main evaluation metrics to a CSV file."""
    metrics = ["bleu", "length_ratio"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Score", "Lower Bound", "Upper Bound"])
        
        for metric in metrics:
            if metric in results["metrics"]:
                data = results["metrics"][metric]
                writer.writerow([
                    metric, 
                    data.get("score", "N/A"),
                    data.get("lower_bound", "N/A"),
                    data.get("upper_bound", "N/A")
                ])

def save_sentence_scores(sentence_scores: Dict[str, List[float]], output_path: str):
    """Save per-sentence scores to a CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        header = ["Sentence ID"]
        for metric in sentence_scores.keys():
            header.append(metric)
        writer.writerow(header)
        
        num_sentences = len(next(iter(sentence_scores.values())))
        for i in range(num_sentences):
            row = [i+1]
            for metric, scores in sentence_scores.items():
                row.append(scores[i])
            writer.writerow(row)

def generate_plots(results: Dict[str, Any], output_dir: str):
    """Generate plots for evaluation results."""
    if "sentence_scores" in results and "bleu" in results["sentence_scores"]:
        plt.figure(figsize=(10, 6))
        plt.hist(results["sentence_scores"]["bleu"], bins=20, alpha=0.7)
        plt.xlabel("BLEU Score")
        plt.ylabel("Number of Sentences")
        plt.title("Distribution of Sentence-level BLEU Scores")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "bleu_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    if "analysis" in results and "by_length" in results["analysis"]:
        length_data = results["analysis"]["by_length"]["bleu"]
        if length_data:
            lengths = []
            scores = []
            counts = []
            
            for length_range, data in length_data.items():
                min_len, max_len = map(int, length_range.split('-'))
                avg_len = (min_len + max_len) / 2
                lengths.append(avg_len)
                scores.append(data["score"])
                counts.append(data["count"])
            
            sorted_data = sorted(zip(lengths, scores, counts))
            lengths, scores, counts = zip(*sorted_data)
            
            plt.figure(figsize=(12, 6))
            
            ax1 = plt.gca()
            line1 = ax1.plot(lengths, scores, 'b-', marker='o', label='BLEU Score')
            ax1.set_xlabel("Sentence Length (words)")
            ax1.set_ylabel("BLEU Score", color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            line2 = ax2.bar(lengths, counts, alpha=0.3, width=5, label='Sentence Count')
            ax2.set_ylabel("Number of Sentences", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            lines = line1 + [line2[0]]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            
            plt.title("BLEU Score by Sentence Length")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "bleu_by_length.png"), dpi=300, bbox_inches="tight")
            plt.close()

def generate_markdown_report(results: Dict[str, Any], output_path: str):
    """Generate a Markdown report of evaluation results."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# NMT Evaluation Report\n\n")
        f.write(f"**Model:** {results['model_name']}\n")
        f.write(f"**Date:** {results['timestamp']}\n")
        f.write(f"**Dataset:** {results['dataset_info']['reference_file']}\n\n")
        
        f.write(f"## Corpus Statistics\n\n")
        f.write(f"- Number of sentences: {results['dataset_info']['num_sentences']}\n")
        f.write(f"- Average reference length: {results['dataset_info']['avg_ref_length']:.2f} words\n")
        f.write(f"- Average hypothesis length: {results['dataset_info']['avg_hyp_length']:.2f} words\n")
        f.write(f"- Length ratio (hyp/ref): {results['metrics']['length_ratio']['score']:.2f}\n\n")
        
        f.write(f"## Evaluation Metrics\n\n")
        f.write("| Metric | Score | 95% Confidence Interval |\n")
        f.write("|--------|-------|-------------------------|\n")
        
        for metric_name, metric_data in results['metrics'].items():
            if metric_name != "length_ratio":  # Already reported above
                score = metric_data['score']
                if 'lower_bound' in metric_data and 'upper_bound' in metric_data:
                    ci = f"{metric_data['lower_bound']:.2f} - {metric_data['upper_bound']:.2f}"
                else:
                    ci = "N/A"
                f.write(f"| {metric_name.upper()} | {score:.2f} | {ci} |\n")
        
        f.write("\n")
        
        if "comparison" in results:
            f.write(f"## Comparison with Previous Results\n\n")
            f.write("| Metric | Current | Previous | Difference |\n")
            f.write("|--------|---------|----------|------------|\n")
            
            for metric_name, comparison_data in results['comparison'].items():
                current = comparison_data['current']
                previous = comparison_data['previous']
                diff = comparison_data['difference']
                diff_str = f"{diff:+.2f}" if diff is not None else "N/A"
                
                f.write(f"| {metric_name.upper()} | {current:.2f} | {previous:.2f} | {diff_str} |\n")
            
            f.write("\n")
        
        f.write(f"## Sample Translations\n\n")
        for i, sample in enumerate(results['samples']):
            f.write(f"### Sample {i+1}\n\n")
            f.write(f"**Input:** {sample['source']}\n\n")
            f.write(f"**Reference:** {sample['reference']}\n\n")
            f.write(f"**Hypothesis:** {sample['hypothesis']}\n\n")
            if "bleu" in sample:
                f.write(f"**BLEU Score:** {sample['bleu']:.2f}\n\n")
        
        f.write(f"## Visualizations\n\n")
        f.write(f"- [BLEU Score Distribution](bleu_distribution.png)\n")
        f.write(f"- [BLEU Score by Sentence Length](bleu_by_length.png)\n\n")

def compare_with_previous(current_results: Dict[str, Any], 
                         previous_results_path: str) -> Dict[str, Dict[str, float]]:
    """Compare current results with previous evaluation results."""
    try:
        with open(previous_results_path, 'r', encoding='utf-8') as f:
            previous_results = json.load(f)
        
        comparison = {}
        for metric_name, current_data in current_results['metrics'].items():
            if metric_name in previous_results.get('metrics', {}):
                previous_score = previous_results['metrics'][metric_name].get('score')
                current_score = current_data.get('score')
                
                if previous_score is not None and current_score is not None:
                    comparison[metric_name] = {
                        'current': current_score,
                        'previous': previous_score,
                        'difference': current_score - previous_score
                    }
        
        return comparison
    except Exception as e:
        print(f"Warning: Could not compare with previous results: {e}")
        return {}

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    references = read_file(args.reference)
    hypotheses = read_file(args.hypothesis)
    sources = read_file(args.input)
    
    if len(references) != len(hypotheses):
        print(f"Warning: Number of references ({len(references)}) does not match number of hypotheses ({len(hypotheses)})")
        min_len = min(len(references), len(hypotheses))
        references = references[:min_len]
        hypotheses = hypotheses[:min_len]
        if sources:
            sources = sources[:min_len]
    
    results = {
        "model_name": args.model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "reference_file": os.path.basename(args.reference),
            "hypothesis_file": os.path.basename(args.hypothesis),
            "source_file": os.path.basename(args.input),
            "num_sentences": len(references),
            "avg_ref_length": sum(len(ref.split()) for ref in references) / len(references),
            "avg_hyp_length": sum(len(hyp.split()) for hyp in hypotheses) / len(hypotheses)
        },
        "metrics": {},
        "sentence_scores": {},
        "analysis": {
            "by_length": {}
        },
        "samples": []
    }
    
    print("\n===== Translation Evaluation Results =====\n")
    
    try:
        bleu = calculate_bleu(references, hypotheses)
        lower, mean, upper = bootstrap_resample(
            references, hypotheses, calculate_bleu, 
            num_samples=args.bootstrap_samples, seed=args.seed
        )
        
        results["metrics"]["bleu"] = {
            "score": bleu.score,
            "lower_bound": lower,
            "upper_bound": upper,
            "details": {
                "precisions": bleu.precisions,
                "bp": bleu.bp,
                "sys_len": bleu.sys_len,
                "ref_len": bleu.ref_len
            }
        }
        
        results["sentence_scores"]["bleu"] = calculate_sentence_bleu(references, hypotheses)
        
        results["analysis"]["by_length"]["bleu"] = analyze_by_length(
            references, hypotheses, calculate_bleu
        )
        
        print(f"BLEU score: {bleu.score:.2f} (95% CI: {lower:.2f} - {upper:.2f})")
        print(f"BLEU details: {bleu}")
        print()
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
    
    length_ratio = calculate_length_ratio(references, hypotheses)
    results["metrics"]["length_ratio"] = {
        "score": length_ratio
    }
    print(f"Length ratio (hyp/ref): {length_ratio:.4f}")
    print()
    
    if args.compare_with:
        comparison = compare_with_previous(results, args.compare_with)
        if comparison:
            results["comparison"] = comparison
            print("===== Comparison with Previous Results =====\n")
            for metric_name, data in comparison.items():
                diff_str = f"{data['difference']:+.2f}"
                print(f"{metric_name.upper()}: {data['current']:.2f} vs {data['previous']:.2f} ({diff_str})")
            print()
    
    num_samples = min(args.num_samples, len(references))
    sample_indices = random.sample(range(len(references)), num_samples)
    
    for idx in sample_indices:
        sample = {
            "source": sources[idx],
            "reference": references[idx],
            "hypothesis": hypotheses[idx]
        }
        
        if "bleu" in results["sentence_scores"]:
            sample["bleu"] = results["sentence_scores"]["bleu"][idx]
        
        results["samples"].append(sample)
    
    print("===== Sample Translations =====\n")
    for i, idx in enumerate(sample_indices[:5]):
        print(f"Example {i+1}:")
        print(f"Input: {sources[idx]}")
        print(f"Reference: {references[idx]}")
        print(f"Hypothesis: {hypotheses[idx]}")
        if "bleu" in results["sentence_scores"]:
            print(f"BLEU: {results['sentence_scores']['bleu'][idx]:.2f}")
        print()
    
    json_path = os.path.join(args.output_dir, "evaluation_results.json")
    csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    sentence_scores_path = os.path.join(args.output_dir, "sentence_scores.csv")
    markdown_path = os.path.join(args.output_dir, "evaluation_report.md")
    
    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    
    if results["sentence_scores"]:
        save_sentence_scores(results["sentence_scores"], sentence_scores_path)
    
    generate_plots(results, args.output_dir)
    generate_markdown_report(results, markdown_path)
    
    print(f"Evaluation results saved to {args.output_dir}")
    print(f"Full report: {markdown_path}")

if __name__ == '__main__':
    main() 