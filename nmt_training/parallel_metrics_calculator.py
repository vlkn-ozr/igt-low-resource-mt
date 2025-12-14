#!/usr/bin/env python3
import json
import sys
import os
import nltk
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import sacrebleu
from sacrebleu.metrics import CHRF
import copy

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("WARNING: COMET package not available. XCOMET scores will not be calculated.")
    print("To install, run: pip install unbabel-comet")

def main():
    if len(sys.argv) != 3:
        print("Usage: python parallel_metrics_calculator.py <reference_file> <prediction_file>")
        sys.exit(1)
    
    reference_file = sys.argv[1]
    prediction_file = sys.argv[2]
    
    prediction_basename = os.path.basename(prediction_file)
    output_file = f"metrics_scored_{prediction_basename.split('.')[0]}.json"
    
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]
        
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predictions = [line.strip() for line in f if line.strip()]
        
        if len(references) != len(predictions):
            print(f"WARNING: Files have different number of lines: {len(references)} references vs {len(predictions)} predictions")
            min_len = min(len(references), len(predictions))
            references = references[:min_len]
            predictions = predictions[:min_len]
            print(f"Using only the first {min_len} lines from each file")
    
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)
    
    data = {
        "results": []
    }
    
    for i, (reference, prediction) in enumerate(zip(references, predictions)):
        data["results"].append({
            "id": i,
            "reference": reference,
            "prediction": prediction
        })
    
    print("Calculating BLEU scores...")
    calculate_bleu(data)
    
    print("\nCalculating chrF++ scores...")
    calculate_chrf(data)
    
    if COMET_AVAILABLE:
        print("\nCalculating XCOMET scores...")
        try:
            calculate_xcomet(data)
            print("\nXCOMET scores have been calculated and added to each item.")
        except Exception as e:
            print(f"\nError calculating XCOMET scores: {e}")
            print("Failed to calculate XCOMET scores.")
    
    summary = calculate_summary(data)
    
    output_data = {
        'summary': summary,
        'results': data['results']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, indent=2, ensure_ascii=False, fp=f)
    
    print_summary(summary)
    print(f"\nResults saved to {output_file}")

def calculate_bleu(data):
    """Calculate BLEU scores for all translations"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    smoothing = SmoothingFunction().method1
    
    references_tokenized_corpus = []
    predictions_tokenized_corpus = []
    bleu_scores = []
    
    for item in data['results']:
        reference = item['reference']
        prediction = item['prediction']
        
        reference_tokens = word_tokenize(reference.lower())
        references_tokenized_corpus.append([reference_tokens])
        
        prediction_tokens = word_tokenize(prediction.lower())
        predictions_tokenized_corpus.append(prediction_tokens)
        
        bleu = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
        
        item['bleu'] = round(bleu * 100, 2)
    
    weights = (0.25, 0.25, 0.25, 0.25)
    
    corpus_bleu_score = corpus_bleu(references_tokenized_corpus, predictions_tokenized_corpus, 
                                    weights=weights, smoothing_function=smoothing)
    
    data['corpus_bleu'] = round(corpus_bleu_score * 100, 2)

def calculate_chrf(data):
    """Calculate chrF++ scores for all translations"""
    chrf_metric = CHRF(char_order=6, word_order=2, beta=2)
    
    all_chrf_scores = []
    
    for item in data['results']:
        reference = [item['reference']]
        prediction = item['prediction']
        
        chrf_score = chrf_metric.corpus_score([prediction], [reference]).score
        item['chrf'] = round(chrf_score, 2)
        all_chrf_scores.append(chrf_score)
    
    data['avg_chrf'] = round(np.mean(all_chrf_scores), 2)

def calculate_xcomet(data):
    """Calculate XCOMET scores for translations"""
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
    
    xcomet_data = []
    
    for item in data['results']:
        xcomet_data.append({
            "src": "",
            "mt": item['prediction'],
            "ref": item['reference']
        })
    
    batch_size = 6
    
    process_xcomet_batch(model, xcomet_data, data, batch_size, device)

def process_xcomet_batch(model, batch_data, data, batch_size, device):
    """Helper function to process a batch of data with XCOMET model"""
    print(f"Calculating scores for {len(batch_data)} translations...")
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
        if i < len(data['results']):
            score_value = scores[i]
            if isinstance(score_value, list):
                score_value = score_value[0]
            data['results'][i]['xcomet'] = round(float(score_value), 4)
            
            if hasattr(model_output, 'metadata') and hasattr(model_output.metadata, 'error_spans') and i < len(model_output.metadata.error_spans):
                data['results'][i]['error_spans'] = model_output.metadata.error_spans[i]
    
    if hasattr(model_output, 'system_score'):
        data['system_xcomet'] = round(float(model_output.system_score), 4)
    else:
        flat_scores = [s[0] if isinstance(s, list) else s for s in scores]
        data['system_xcomet'] = round(float(np.mean(flat_scores)), 4)

def calculate_summary(data):
    """Calculate summary statistics for all metrics"""
    sample_count = len(data['results'])
    
    summary = {
        'sample_count': sample_count,
        'metrics': {
            'bleu': data.get('corpus_bleu', 0),
            'chrf': data.get('avg_chrf', 0)
        }
    }
    
    if COMET_AVAILABLE:
        summary['metrics']['xcomet'] = data.get('system_xcomet', 0)
    
    return summary

def print_summary(summary):
    """Print a summary of all metrics to the console"""
    print("\n=== METRICS SUMMARY ===")
    print(f"Number of samples: {summary['sample_count']}")
    
    print("\nOverall Scores:")
    print(f"  BLEU: {summary['metrics']['bleu']:.2f}")
    print(f"  chrF++: {summary['metrics']['chrf']:.2f}")
    
    if 'xcomet' in summary['metrics']:
        print(f"  XCOMET: {summary['metrics']['xcomet']:.4f}")

if __name__ == "__main__":
    main() 