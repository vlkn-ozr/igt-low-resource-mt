#!/usr/bin/env python3
import json
import sys
import os
import nltk
import numpy as np
import torch
import argparse
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import sacrebleu
from sacrebleu.metrics import CHRF
import copy

# Import COMET if available
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("WARNING: COMET package not available. XCOMET scores will not be calculated.")
    print("To install, run: pip install unbabel-comet")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate metrics for translation evaluation data')
    parser.add_argument('input_file', help='Input JSON file to process')
    parser.add_argument('--zero-shot', action='store_true', help='Calculate metrics for zero_shot field')
    parser.add_argument('--embeddings-few-shot', action='store_true', help='Calculate metrics for embeddings_few_shot field')
    parser.add_argument('--chrf-advanced-few-shot', action='store_true', help='Calculate metrics for chrf_advanced_few_shot field')
    parser.add_argument('--outdir', type=str, default=None, help='Directory to save the output file (defaults to input file\'s directory)')
    
    args = parser.parse_args()
    
    # If no specific fields are specified, process all by default
    if not any([args.zero_shot, args.embeddings_few_shot, args.chrf_advanced_few_shot]):
        fields_to_process = {"zero_shot": True, "embeddings_few_shot": True, "chrf_advanced_few_shot": True}
    else:
        fields_to_process = {
            "zero_shot": args.zero_shot,
            "embeddings_few_shot": args.embeddings_few_shot,
            "chrf_advanced_few_shot": args.chrf_advanced_few_shot
        }
    
    input_file = args.input_file
    
    print(f"Processing {input_file}...")
    print(f"Fields to process: {', '.join([field for field, include in fields_to_process.items() if include])}")
    
    # Determine output directory. If --outdir is not provided, default to the input file's directory
    output_dir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(input_file))

    # Ensure the output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Generate output filename in chosen directory
    basename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, f"all_metrics_scored_{basename}")
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Calculate BLEU scores
    print("Calculating BLEU scores...")
    calculate_bleu(data, fields_to_process)
    
    # Calculate chrF++ scores
    print("\nCalculating chrF++ scores...")
    calculate_chrf(data, fields_to_process)
    
    # Calculate XCOMET scores if available
    if COMET_AVAILABLE:
        print("\nCalculating XCOMET scores...")
        try:
            calculate_xcomet(data, fields_to_process)
            print("\nXCOMET scores have been calculated and added to each item.")
        except Exception as e:
            print(f"\nError calculating XCOMET scores: {e}")
            print("Failed to calculate XCOMET scores.")
    
    # Calculate summary statistics and reorganize data
    summary = calculate_summary(data)
    
    # Create the final output structure
    output_data = {
        'summary': summary,
        'results': data['results']
    }
    
    # Write the results to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, indent=2, ensure_ascii=False, fp=f)
    
    # Print summary information
    print_summary(summary)
    print(f"\nResults saved to {output_file}")

def calculate_bleu(data, fields_to_process):
    """Calculate BLEU scores for specified translations"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    smoothing = SmoothingFunction().method1
    
    expected_tokenized_corpus = []
    zero_shot_tokenized_corpus = []
    embeddings_few_shot_tokenized_corpus = []
    chrf_advanced_few_shot_tokenized_corpus = []
    
    zero_shot_scores = []
    embeddings_few_shot_scores = []
    chrf_advanced_few_shot_scores = []
    
    for item in data['results']:
        expected = item['expected']
        zero_shot = item.get('zero_shot', '') if fields_to_process.get('zero_shot') else ''
        embeddings_few_shot = item.get('embeddings_few_shot', '') if fields_to_process.get('embeddings_few_shot') else ''
        chrf_advanced_few_shot = item.get('chrf_advanced_few_shot', '') if fields_to_process.get('chrf_advanced_few_shot') else ''
        
        expected_tokens = word_tokenize(expected.lower())
        
        expected_tokenized_corpus.append([expected_tokens])
        
        if fields_to_process.get('zero_shot') and zero_shot:
            zero_shot_tokens = word_tokenize(zero_shot.lower())
            zero_shot_tokenized_corpus.append(zero_shot_tokens)
            zero_bleu = sentence_bleu([expected_tokens], zero_shot_tokens, smoothing_function=smoothing)
            zero_shot_scores.append(zero_bleu)
            item['zero_shot_bleu'] = round(zero_bleu * 100, 2)
        elif fields_to_process.get('zero_shot'):
            zero_shot_tokenized_corpus.append([])
            item['zero_shot_bleu'] = 0.0
        
        if fields_to_process.get('embeddings_few_shot') and embeddings_few_shot:
            embeddings_few_shot_tokens = word_tokenize(embeddings_few_shot.lower())
            embeddings_few_shot_tokenized_corpus.append(embeddings_few_shot_tokens)
            embeddings_few_shot_bleu = sentence_bleu([expected_tokens], embeddings_few_shot_tokens, smoothing_function=smoothing)
            embeddings_few_shot_scores.append(embeddings_few_shot_bleu)
            item['embeddings_few_shot_bleu'] = round(embeddings_few_shot_bleu * 100, 2)
        elif fields_to_process.get('embeddings_few_shot'):
            embeddings_few_shot_tokenized_corpus.append([])
            item['embeddings_few_shot_bleu'] = 0.0
        
        if fields_to_process.get('chrf_advanced_few_shot') and chrf_advanced_few_shot:
            chrf_advanced_few_shot_tokens = word_tokenize(chrf_advanced_few_shot.lower())
            chrf_advanced_few_shot_tokenized_corpus.append(chrf_advanced_few_shot_tokens)
            chrf_advanced_few_shot_bleu = sentence_bleu([expected_tokens], chrf_advanced_few_shot_tokens, smoothing_function=smoothing)
            chrf_advanced_few_shot_scores.append(chrf_advanced_few_shot_bleu)
            item['chrf_advanced_few_shot_bleu'] = round(chrf_advanced_few_shot_bleu * 100, 2)
        elif fields_to_process.get('chrf_advanced_few_shot'):
            chrf_advanced_few_shot_tokenized_corpus.append([])
            item['chrf_advanced_few_shot_bleu'] = 0.0
    
    weights = (0.25, 0.25, 0.25, 0.25)
    
    if fields_to_process.get('zero_shot'):
        zero_shot_corpus_bleu = corpus_bleu(expected_tokenized_corpus, zero_shot_tokenized_corpus, 
                                          weights=weights, smoothing_function=smoothing)
        data['zero_shot_corpus_bleu'] = round(zero_shot_corpus_bleu * 100, 2)
    
    if fields_to_process.get('embeddings_few_shot'):
        embeddings_few_shot_corpus_bleu = corpus_bleu(expected_tokenized_corpus, embeddings_few_shot_tokenized_corpus, 
                                         weights=weights, smoothing_function=smoothing)
        data['embeddings_few_shot_corpus_bleu'] = round(embeddings_few_shot_corpus_bleu * 100, 2)
    
    if fields_to_process.get('chrf_advanced_few_shot'):
        chrf_advanced_few_shot_corpus_bleu = corpus_bleu(expected_tokenized_corpus, chrf_advanced_few_shot_tokenized_corpus, 
                                                 weights=weights, smoothing_function=smoothing)
        data['chrf_advanced_few_shot_corpus_bleu'] = round(chrf_advanced_few_shot_corpus_bleu * 100, 2)

def calculate_chrf(data, fields_to_process):
    """Calculate chrF++ scores for specified translations"""
    chrf_metric = CHRF(char_order=6, word_order=2, beta=2)
    
    for item in data['results']:
        reference = [item['expected']]
        
        if fields_to_process.get('zero_shot') and 'zero_shot' in item:
            zero_shot_chrf = chrf_metric.corpus_score(
                [item['zero_shot']], [reference]).score
            item['zero_shot_chrf'] = round(zero_shot_chrf, 2)
        
        if fields_to_process.get('embeddings_few_shot') and 'embeddings_few_shot' in item:
            embeddings_few_shot_chrf = chrf_metric.corpus_score(
                [item['embeddings_few_shot']], [reference]).score
            item['embeddings_few_shot_chrf'] = round(embeddings_few_shot_chrf, 2)
        
        if fields_to_process.get('chrf_advanced_few_shot') and 'chrf_advanced_few_shot' in item:
            chrf_advanced_few_shot_chrf = chrf_metric.corpus_score(
                [item['chrf_advanced_few_shot']], [reference]).score
            item['chrf_advanced_few_shot_chrf'] = round(chrf_advanced_few_shot_chrf, 2)

def calculate_xcomet(data, fields_to_process):
    """Calculate XCOMET scores for specified translations"""
    if not COMET_AVAILABLE:
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Downloading and loading XCOMET-XL model...")
    try:
        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)
        print("Successfully loaded XCOMET-XL model")
    except Exception as e:
        print(f"Error loading XCOMET-XL model: {e}")
        print("Falling back to standard COMET model...")
        model_path = download_model("wmt20-comet-da")
        model = load_from_checkpoint(model_path)
        print("Successfully loaded standard COMET model")
    
    zero_shot_data = []
    few_shot_data = []
    advanced_few_shot_data = []
    
    for item in data['results']:
        source = item.get('input', '')
        reference = item.get('expected', '')
        
        if fields_to_process.get('zero_shot') and 'zero_shot' in item:
            zero_shot_data.append({
                "src": source,
                "mt": item['zero_shot'],
                "ref": reference
            })
            
        if fields_to_process.get('embeddings_few_shot') and 'embeddings_few_shot' in item:
            few_shot_data.append({
                "src": source,
                "mt": item['embeddings_few_shot'],
                "ref": reference
            })
            
        if fields_to_process.get('chrf_advanced_few_shot') and 'chrf_advanced_few_shot' in item:
            advanced_few_shot_data.append({
                "src": source,
                "mt": item['chrf_advanced_few_shot'],
                "ref": reference
            })
    
    batch_size = 6
    
    if fields_to_process.get('zero_shot'):
        process_xcomet_batch(model, zero_shot_data, data, 'zero_shot', batch_size, device)
    
    if fields_to_process.get('embeddings_few_shot'):
        process_xcomet_batch(model, few_shot_data, data, 'embeddings_few_shot', batch_size, device)
    
    if fields_to_process.get('chrf_advanced_few_shot'):
        process_xcomet_batch(model, advanced_few_shot_data, data, 'chrf_advanced_few_shot', batch_size, device)

def process_xcomet_batch(model, batch_data, data, prefix, batch_size, device):
    """Helper function to process a batch of data with XCOMET model"""
    print(f"Calculating scores for {len(batch_data)} {prefix} translations...")
    if not batch_data:
        return
    
    model_output = model.predict(batch_data, batch_size=batch_size, gpus=0 if device == "cpu" else 1)
    
    scores = None
    
    if hasattr(model_output, 'scores'):
        scores = model_output.scores
    elif isinstance(model_output, dict) and 'scores' in model_output:
        scores = model_output['scores']
    else:
        scores = model_output
    
    for i in range(len(scores)):
        if i < len(data['results']) and f'{prefix}' in data['results'][i]:
            score_value = scores[i]
            if isinstance(score_value, list):
                score_value = score_value[0]
            data['results'][i][f'{prefix}_xcomet'] = round(float(score_value), 4)
            
            if hasattr(model_output, 'metadata') and hasattr(model_output.metadata, 'error_spans') and i < len(model_output.metadata.error_spans):
                data['results'][i][f'{prefix}_error_spans'] = model_output.metadata.error_spans[i]
    
    if hasattr(model_output, 'system_score'):
        data[f'{prefix}_system_xcomet'] = round(float(model_output.system_score), 4)
    else:
        flat_scores = [s[0] if isinstance(s, list) else s for s in scores]
        data[f'{prefix}_system_xcomet'] = round(float(np.mean(flat_scores)), 4)

def calculate_summary(data):
    """Calculate summary statistics for all metrics"""
    sample_count = len(data['results'])
    
    summary = {
        'sample_count': sample_count,
        'metrics': {
            'bleu': {
                'zero_shot': data.get('zero_shot_corpus_bleu', 0),
                'embeddings_few_shot': data.get('embeddings_few_shot_corpus_bleu', 0),
                'chrf_advanced_few_shot': data.get('chrf_advanced_few_shot_corpus_bleu', 0)
            },
            'chrf': {
                'zero_shot': round(np.mean([item.get('zero_shot_chrf', 0) for item in data['results'] if 'zero_shot_chrf' in item]), 2),
                'embeddings_few_shot': round(np.mean([item.get('embeddings_few_shot_chrf', 0) for item in data['results'] if 'embeddings_few_shot_chrf' in item]), 2),
                'chrf_advanced_few_shot': round(np.mean([item.get('chrf_advanced_few_shot_chrf', 0) for item in data['results'] if 'chrf_advanced_few_shot_chrf' in item]), 2)
            }
        }
    }
    
    if COMET_AVAILABLE:
        summary['metrics']['xcomet'] = {
            'zero_shot': data.get('zero_shot_system_xcomet', 0),
            'embeddings_few_shot': data.get('embeddings_few_shot_system_xcomet', 0),
            'chrf_advanced_few_shot': data.get('chrf_advanced_few_shot_system_xcomet', 0)
        }
    
    return summary

def print_summary(summary):
    """Print a summary of all metrics to the console"""
    print("\n=== METRICS SUMMARY ===")
    print(f"Number of samples: {summary['sample_count']}")
    
    print("\nBLEU Scores (0-100 scale):")
    print(f"  Zero-shot: {summary['metrics']['bleu']['zero_shot']:.2f}")
    print(f"  chrF Few-shot: {summary['metrics']['bleu']['embeddings_few_shot']:.2f}")
    print(f"  chrF Advanced few-shot: {summary['metrics']['bleu']['chrf_advanced_few_shot']:.2f}")
    
    print("\nchrF++ Scores:")
    print(f"  Zero-shot: {summary['metrics']['chrf']['zero_shot']:.2f}")
    print(f"  chrF Few-shot: {summary['metrics']['chrf']['embeddings_few_shot']:.2f}")
    print(f"  chrF Advanced few-shot: {summary['metrics']['chrf']['chrf_advanced_few_shot']:.2f}")
    
    if 'xcomet' in summary['metrics']:
        print("\nXCOMET Scores:")
        print(f"  Zero-shot: {summary['metrics']['xcomet']['zero_shot']:.4f}")
        print(f"  chrF Few-shot: {summary['metrics']['xcomet']['embeddings_few_shot']:.4f}")
        print(f"  chrF Advanced few-shot: {summary['metrics']['xcomet']['chrf_advanced_few_shot']:.4f}")

if __name__ == "__main__":
    main() 