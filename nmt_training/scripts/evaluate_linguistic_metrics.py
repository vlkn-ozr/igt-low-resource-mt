#!/usr/bin/env python3
import os
import argparse
import sys
import json
import csv
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import sacrebleu
import spacy
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
 
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate NMT translations using linguistic metrics')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference file (ground truth translations)')
    parser.add_argument('--hypothesis', type=str, required=True,
                        help='Path to hypothesis file (model output)')
    parser.add_argument('--source', type=str, required=False,
                        help='Path to source file (original text)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'logs_rnn_small_multi_100k_plus_1200_sampled_setimes_no_pos_preset_eval_tur', 'fieldwork', 'kara', 'ling_eval'),
                        help='Directory to save evaluation results (default: logs/ling_eval)')
    parser.add_argument('--model_name', type=str, default='nmt_model',
                        help='Name of the model being evaluated (for reporting)')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of random translation samples to include in the report')
    parser.add_argument('--language', type=str, default='en',
                        help='Language code for spaCy model (default: en)')
    parser.add_argument('--detailed_report', action='store_true', default=True,
                        help='Generate a detailed step-by-step linguistic analysis report')
    parser.add_argument('--detailed_samples', type=int, default=20,
                        help='Number of samples to include in the detailed report (default: 3)')
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

def load_spacy_model(language: str = 'en'):
    """Load spaCy model for the specified language."""
    try:
        # Try to load the model
        if language == 'en':
            model_name = 'en_core_web_sm'
        else:
            model_name = f"{language}_core_news_sm"
        
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model {model_name}...")
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
        
        return nlp
    except Exception as e:
        print(f"Error loading spaCy model for language '{language}': {e}")
        print("Falling back to simple tokenization and POS tagging via NLTK.")
        return None

def extract_nouns(text: str, nlp) -> List[str]:
    """Extract all nouns from the text using spaCy."""
    if nlp is None:
        # Fallback to simple word tokenization (not ideal, but better than nothing)
        words = word_tokenize(text.lower())
        return list(set(words))  # Return unique words
    
    doc = nlp(text)
    # Use set to avoid duplicates, then convert back to list
    return list(set([token.text.lower() for token in doc if token.pos_ in ('NOUN', 'PROPN')]))

def extract_verbs(text: str, nlp) -> List[str]:
    """Extract all verbs from the text using spaCy."""
    if nlp is None:
        # Fallback to simple word tokenization (not ideal)
        words = word_tokenize(text.lower())
        return list(set(words))  # Return unique words
    
    doc = nlp(text)
    # Use set to avoid duplicates, then convert back to list
    return list(set([token.text.lower() for token in doc if token.pos_ == 'VERB']))

def extract_subject_verb_pairs(text: str, nlp) -> List[Tuple[str, str]]:
    """Extract subject-verb pairs from the text using spaCy."""
    if nlp is None:
        return []  # Without dependency parsing, we can't identify subject-verb pairs
    
    doc = nlp(text)
    pairs = []
    
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass') and token.head.pos_ == 'VERB':
            subject = token.text.lower()
            verb = token.head.text.lower()
            pairs.append((subject, verb))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    
    return unique_pairs

def extract_verb_tenses(text: str, nlp) -> List[Tuple[str, str]]:
    """Extract verbs and their tenses from the text using spaCy."""
    if nlp is None:
        return []  # Without morphological analysis, we can't identify tenses
    
    doc = nlp(text)
    tenses = []
    
    for token in doc:
        if token.pos_ == 'VERB':
            verb = token.text.lower()
            # Get tense information if available
            tense = token.morph.get('Tense')
            if tense:
                tense = tense[0] if isinstance(tense, list) else tense
                tenses.append((verb, tense))
            else:
                # Try to infer tense from the auxiliary verbs
                aux_verbs = [aux.text.lower() for aux in token.children if aux.dep_ == 'aux']
                if aux_verbs:
                    tenses.append((verb, ' '.join(aux_verbs)))
                else:
                    tenses.append((verb, 'unknown'))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tenses = []
    for tense_pair in tenses:
        if tense_pair[0] not in seen:  # Only check verb, as a verb should have only one tense
            seen.add(tense_pair[0])
            unique_tenses.append(tense_pair)
    
    return unique_tenses

def calculate_unique_word_ratio(text: str, nlp=None) -> float:
    """Calculate the ratio of unique words to total words in the text."""
    # Use spaCy's tokenization if available
    if nlp is not None:
        doc = nlp(text.lower())
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
    else:
        # Fallback to simple split by whitespace
        words = text.lower().split()
    
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def calculate_noun_match_accuracy(ref_nouns: List[str], hyp_nouns: List[str]) -> float:
    """Calculate the percentage of correctly predicted nouns."""
    if not ref_nouns:
        return 1.0 if not hyp_nouns else 0.0
    
    # Count matches without duplicates
    matches = sum(1 for noun in set(hyp_nouns) if noun in set(ref_nouns))
    return matches / len(set(ref_nouns)) if set(ref_nouns) else 0.0

def calculate_verb_match_accuracy(ref_verbs: List[str], hyp_verbs: List[str]) -> float:
    """Calculate the percentage of correctly predicted verbs."""
    if not ref_verbs:
        return 1.0 if not hyp_verbs else 0.0
    
    # Count matches without duplicates
    matches = sum(1 for verb in set(hyp_verbs) if verb in set(ref_verbs))
    return matches / len(set(ref_verbs)) if set(ref_verbs) else 0.0

def calculate_subject_verb_agreement_accuracy(ref_pairs: List[Tuple[str, str]], hyp_pairs: List[Tuple[str, str]]) -> float:
    """Calculate the percentage of correctly predicted subject-verb agreements."""
    if not ref_pairs:
        return 1.0 if not hyp_pairs else 0.0
    
    # Use sets to ensure unique pairs
    ref_pair_set = set(ref_pairs)
    hyp_pair_set = set(hyp_pairs)
    matches = sum(1 for pair in hyp_pair_set if pair in ref_pair_set)
    return matches / len(ref_pair_set) if ref_pair_set else 0.0

def calculate_tense_match_accuracy(ref_tenses: List[Tuple[str, str]], hyp_tenses: List[Tuple[str, str]]) -> float:
    """Calculate the percentage of correctly predicted verb tenses."""
    if not ref_tenses:
        return 1.0 if not hyp_tenses else 0.0
    
    # Create dictionaries to easily look up tenses for verbs
    ref_tense_dict = {verb: tense for verb, tense in ref_tenses}
    hyp_tense_dict = {verb: tense for verb, tense in hyp_tenses}
    
    # Count matches for verbs that appear in both reference and hypothesis
    matches = 0
    for verb in ref_tense_dict:
        if verb in hyp_tense_dict and ref_tense_dict[verb] == hyp_tense_dict[verb]:
            matches += 1
    
    return matches / len(ref_tense_dict) if len(ref_tense_dict) > 0 else 0.0

def evaluate_translations(references: List[str], hypotheses: List[str], nlp) -> Dict[str, Any]:
    """Evaluate translations using linguistic metrics."""
    results = {
        "noun_match_accuracy": [],
        "verb_match_accuracy": [],
        "subject_verb_agreement_accuracy": [],
        "tense_match_accuracy": [],
        "non_repetition_ratio": [],
        "sentence_bleu": []  # Add sentence-level BLEU scores
    }
    
    for ref, hyp in tqdm(zip(references, hypotheses), total=len(references), desc="Evaluating linguistic metrics"):
        # Extract linguistic features
        ref_nouns = extract_nouns(ref, nlp)
        hyp_nouns = extract_nouns(hyp, nlp)
        
        ref_verbs = extract_verbs(ref, nlp)
        hyp_verbs = extract_verbs(hyp, nlp)
        
        ref_pairs = extract_subject_verb_pairs(ref, nlp)
        hyp_pairs = extract_subject_verb_pairs(hyp, nlp)
        
        ref_tenses = extract_verb_tenses(ref, nlp)
        hyp_tenses = extract_verb_tenses(hyp, nlp)
        
        # Calculate metrics
        noun_acc = calculate_noun_match_accuracy(ref_nouns, hyp_nouns)
        verb_acc = calculate_verb_match_accuracy(ref_verbs, hyp_verbs)
        sv_acc = calculate_subject_verb_agreement_accuracy(ref_pairs, hyp_pairs)
        tense_acc = calculate_tense_match_accuracy(ref_tenses, hyp_tenses)
        unique_ratio = calculate_unique_word_ratio(hyp, nlp)
        
        # Calculate sentence-level BLEU
        try:
            sentence_bleu = sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0  # Normalize to 0-1 scale like other metrics
        except:
            sentence_bleu = 0.0
        
        # Store results (still in 0-1 scale for individual sentences)
        results["noun_match_accuracy"].append(noun_acc)
        results["verb_match_accuracy"].append(verb_acc)
        results["subject_verb_agreement_accuracy"].append(sv_acc)
        results["tense_match_accuracy"].append(tense_acc)
        results["non_repetition_ratio"].append(unique_ratio)
        results["sentence_bleu"].append(sentence_bleu)
    
    # Calculate average metrics and convert to 0-100 scale
    avg_results = {
        "noun_match_accuracy": np.mean(results["noun_match_accuracy"]) * 100,
        "verb_match_accuracy": np.mean(results["verb_match_accuracy"]) * 100,
        "subject_verb_agreement_accuracy": np.mean(results["subject_verb_agreement_accuracy"]) * 100,
        "tense_match_accuracy": np.mean(results["tense_match_accuracy"]) * 100,
        "non_repetition_ratio": np.mean(results["non_repetition_ratio"]) * 100,
        "sentence_bleu": np.mean(results["sentence_bleu"]) * 100  # Use consistent key name
    }
    
    return {"sentence_scores": results, "average_scores": avg_results}

def calculate_bleu(references: List[str], hypotheses: List[str]) -> sacrebleu.metrics.bleu.BLEUScore:
    """Calculate BLEU score."""
    return sacrebleu.corpus_bleu(hypotheses, [references])

def save_results_json(results: Dict[str, Any], output_path: str):
    """Save evaluation results to a JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a serializable copy of the results
    serializable_results = results.copy()
    
    # Convert BLEU score object to a dictionary if present
    if "bleu" in serializable_results:
        bleu_obj = serializable_results["bleu"]
        serializable_results["bleu"] = {
            "score": bleu_obj.score,
            "precisions": bleu_obj.precisions,
            "bp": bleu_obj.bp,
            "sys_len": bleu_obj.sys_len,
            "ref_len": bleu_obj.ref_len
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)

def save_results_csv(results: Dict[str, Any], output_path: str):
    """Save main evaluation metrics to a CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Score"])
        
        for metric, score in results["average_scores"].items():
            writer.writerow([metric, f"{score:.2f}"])
        
        if "bleu" in results:
            writer.writerow(["bleu", f"{results['bleu'].score:.2f}"])

def save_sentence_scores(results: Dict[str, Any], output_path: str):
    """Save per-sentence scores to a CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert scores to 0-100 scale for the CSV
    df = pd.DataFrame({
        k: [v * 100 for v in vals] for k, vals in results["sentence_scores"].items()
    })
    df.to_csv(output_path, index=True, index_label="Sentence ID")

def generate_markdown_report(results: Dict[str, Any], samples: List[Dict[str, Any]], output_path: str):
    """Generate a Markdown report of evaluation results."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# Linguistic Metrics Evaluation Report\n\n")
        f.write(f"**Model:** {results['model_name']}\n")
        f.write(f"**Date:** {results['timestamp']}\n")
        f.write(f"**Dataset:** {results['dataset_info']['reference_file']}\n\n")
        
        # Corpus statistics
        f.write(f"## Corpus Statistics\n\n")
        f.write(f"- Number of sentences: {results['dataset_info']['num_sentences']}\n\n")
        
        # Main metrics
        f.write(f"## Evaluation Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        
        for metric, score in results["average_scores"].items():
            metric_name = " ".join(metric.split("_")).title()
            f.write(f"| {metric_name} | {score:.2f} |\n")
        
        if "bleu" in results:
            f.write(f"| BLEU | {results['bleu'].score:.2f} |\n")
        
        f.write("\n")
        
        # Sample translations
        f.write(f"## Sample Translations\n\n")
        for i, sample in enumerate(samples):
            f.write(f"### Sample {i+1}\n\n")
            
            if "source" in sample:
                f.write(f"**Source:** {sample['source']}\n\n")
            
            f.write(f"**Reference:** {sample['reference']}\n\n")
            f.write(f"**Hypothesis:** {sample['hypothesis']}\n\n")
            
            f.write("**Metrics:**\n\n")
            f.write("| Metric | Score |\n")
            f.write("|--------|-------|\n")
            
            for metric in results["average_scores"].keys():
                metric_name = " ".join(metric.split("_")).title()
                if metric in sample:
                    # Convert to 0-100 scale for display
                    score = sample[metric] * 100
                    f.write(f"| {metric_name} | {score:.2f} |\n")
            
            # Include corpus-level BLEU if available
            if "bleu" in results and "bleu" not in sample and "sentence_bleu" not in sample:
                f.write(f"| BLEU | {results['bleu'].score:.2f} |\n")
            
            f.write("\n")

def generate_detailed_report(references: List[str], hypotheses: List[str], 
                            nlp, sample_indices: List[int], output_path: str):
    """
    Generate a detailed report showing step-by-step how each metric is calculated
    for specific sample sentences.
    
    Args:
        references: List of reference translations
        hypotheses: List of hypothesis translations
        nlp: spaCy language model
        sample_indices: Indices of sentences to include in the report
        output_path: Path to save the detailed report
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Linguistic Metrics Evaluation: Step-by-Step Analysis\n\n")
        f.write("This report shows the detailed linguistic analysis process for selected translation pairs.\n\n")
        
        # Initialize lists to store sample metrics for the summary
        all_noun_acc = []
        all_verb_acc = []
        all_sv_acc = []
        all_tense_acc = []
        all_unique_ratio = []
        all_bleu = []
        
        # Process each sample and collect metrics
        sample_metrics = []
        for idx, sample_idx in enumerate(sample_indices):
            ref = references[sample_idx]
            hyp = hypotheses[sample_idx]
            
            # Extract all linguistic features - these are already unique items based on the updated extraction functions
            ref_nouns = extract_nouns(ref, nlp)
            hyp_nouns = extract_nouns(hyp, nlp)
            
            ref_verbs = extract_verbs(ref, nlp)
            hyp_verbs = extract_verbs(hyp, nlp)
            
            ref_pairs = extract_subject_verb_pairs(ref, nlp)
            hyp_pairs = extract_subject_verb_pairs(hyp, nlp)
            
            ref_tenses = extract_verb_tenses(ref, nlp)
            hyp_tenses = extract_verb_tenses(hyp, nlp)
            
            # Calculate metrics
            noun_acc = calculate_noun_match_accuracy(ref_nouns, hyp_nouns)
            verb_acc = calculate_verb_match_accuracy(ref_verbs, hyp_verbs)
            sv_acc = calculate_subject_verb_agreement_accuracy(ref_pairs, hyp_pairs)
            tense_acc = calculate_tense_match_accuracy(ref_tenses, hyp_tenses)
            unique_ratio = calculate_unique_word_ratio(hyp, nlp)
            
            # Calculate sentence-level BLEU
            try:
                bleu_score = sacrebleu.sentence_bleu(hyp, [ref]).score
            except:
                bleu_score = 0.0
            
            # Append metrics to lists
            all_noun_acc.append(noun_acc)
            all_verb_acc.append(verb_acc)
            all_sv_acc.append(sv_acc)
            all_tense_acc.append(tense_acc)
            all_unique_ratio.append(unique_ratio)
            all_bleu.append(bleu_score)
            
            # Store metrics for this sample
            sample_metrics.append({
                "idx": idx + 1,
                "ref": ref,
                "hyp": hyp,
                "noun_acc": noun_acc,
                "verb_acc": verb_acc,
                "sv_acc": sv_acc,
                "tense_acc": tense_acc,
                "unique_ratio": unique_ratio,
                "bleu": bleu_score,
                "ref_nouns": ref_nouns,
                "hyp_nouns": hyp_nouns,
                "ref_verbs": ref_verbs,
                "hyp_verbs": hyp_verbs,
                "ref_pairs": ref_pairs,
                "hyp_pairs": hyp_pairs,
                "ref_tenses": ref_tenses,
                "hyp_tenses": hyp_tenses
            })
        
        # Calculate averages
        avg_noun_acc = np.mean(all_noun_acc) * 100 if all_noun_acc else 0
        avg_verb_acc = np.mean(all_verb_acc) * 100 if all_verb_acc else 0
        avg_sv_acc = np.mean(all_sv_acc) * 100 if all_sv_acc else 0
        avg_tense_acc = np.mean(all_tense_acc) * 100 if all_tense_acc else 0
        avg_unique_ratio = np.mean(all_unique_ratio) * 100 if all_unique_ratio else 0
        avg_bleu = np.mean(all_bleu) if all_bleu else 0
        
        # Write summary metrics table at the beginning
        f.write("## Summary Metrics for Sample Set\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Noun Match Accuracy | {avg_noun_acc:.2f} |\n")
        f.write(f"| Verb Match Accuracy | {avg_verb_acc:.2f} |\n")
        f.write(f"| Subject Verb Agreement Accuracy | {avg_sv_acc:.2f} |\n")
        f.write(f"| Tense Match Accuracy | {avg_tense_acc:.2f} |\n")
        f.write(f"| Non Repetition Ratio | {avg_unique_ratio:.2f} |\n")
        f.write(f"| BLEU | {avg_bleu:.2f} |\n\n")
        
        # Now process each sample in detail
        for sample in sample_metrics:
            idx = sample["idx"]
            ref = sample["ref"]
            hyp = sample["hyp"]
            noun_acc = sample["noun_acc"]
            verb_acc = sample["verb_acc"]
            sv_acc = sample["sv_acc"]
            tense_acc = sample["tense_acc"]
            unique_ratio = sample["unique_ratio"]
            bleu = sample["bleu"]
            ref_nouns = sample["ref_nouns"]
            hyp_nouns = sample["hyp_nouns"]
            ref_verbs = sample["ref_verbs"]
            hyp_verbs = sample["hyp_verbs"]
            ref_pairs = sample["ref_pairs"]
            hyp_pairs = sample["hyp_pairs"]
            ref_tenses = sample["ref_tenses"]
            hyp_tenses = sample["hyp_tenses"]
            
            f.write(f"## Sample {idx}\n\n")
            f.write(f"**Reference:** \"{ref}\"\n\n")
            f.write(f"**Hypothesis:** \"{hyp}\"\n\n")
            
            # Add metrics summary table for this sample
            f.write("**Metrics:**\n\n")
            f.write("| Metric | Score |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Noun Match Accuracy | {noun_acc*100:.2f} |\n")
            f.write(f"| Verb Match Accuracy | {verb_acc*100:.2f} |\n")
            f.write(f"| Subject Verb Agreement Accuracy | {sv_acc*100:.2f} |\n")
            f.write(f"| Tense Match Accuracy | {tense_acc*100:.2f} |\n")
            f.write(f"| Non Repetition Ratio | {unique_ratio*100:.2f} |\n")
            f.write(f"| BLEU | {bleu:.2f} |\n\n")
            
            # 1. Noun Match Analysis
            f.write("### 1. Noun Match Accuracy\n\n")
            f.write("#### Step 1: Extract nouns from both sentences\n\n")
            
            f.write("**Reference Nouns:**\n")
            for noun in ref_nouns:
                f.write(f"- {noun}\n")
            f.write("\n")
            
            f.write("**Hypothesis Nouns:**\n")
            for noun in hyp_nouns:
                f.write(f"- {noun}\n")
            f.write("\n")
            
            f.write("#### Step 2: Identify matching nouns\n\n")
            
            matching_nouns = [noun for noun in hyp_nouns if noun in ref_nouns]
            f.write("**Matching nouns:**\n")
            if matching_nouns:
                for noun in matching_nouns:
                    f.write(f"- {noun}\n")
            else:
                f.write("- No matching nouns found\n")
            f.write("\n")
            
            f.write("#### Step 3: Calculate matching percentage\n\n")
            f.write(f"- Matching nouns: {len(matching_nouns)}\n")
            f.write(f"- Total reference nouns: {len(ref_nouns)}\n")
            f.write(f"- Calculation: {len(matching_nouns)}/{len(ref_nouns) if len(ref_nouns) > 0 else 1} = {noun_acc:.4f} = **{noun_acc*100:.2f}%**\n\n")
            
            # 2. Verb Match Analysis
            f.write("### 2. Verb Match Accuracy\n\n")
            f.write("#### Step 1: Extract verbs from both sentences\n\n")
            
            f.write("**Reference Verbs:**\n")
            for verb in ref_verbs:
                f.write(f"- {verb}\n")
            f.write("\n")
            
            f.write("**Hypothesis Verbs:**\n")
            for verb in hyp_verbs:
                f.write(f"- {verb}\n")
            f.write("\n")
            
            f.write("#### Step 2: Identify matching verbs\n\n")
            
            matching_verbs = [verb for verb in hyp_verbs if verb in ref_verbs]
            f.write("**Matching verbs:**\n")
            if matching_verbs:
                for verb in matching_verbs:
                    f.write(f"- {verb}\n")
            else:
                f.write("- No matching verbs found\n")
            f.write("\n")
            
            f.write("#### Step 3: Calculate matching percentage\n\n")
            f.write(f"- Matching verbs: {len(matching_verbs)}\n")
            f.write(f"- Total reference verbs: {len(ref_verbs)}\n")
            f.write(f"- Calculation: {len(matching_verbs)}/{len(ref_verbs) if len(ref_verbs) > 0 else 1} = {verb_acc:.4f} = **{verb_acc*100:.2f}%**\n\n")
            
            # 3. Subject-Verb Agreement Analysis
            f.write("### 3. Subject-Verb Agreement Accuracy\n\n")
            f.write("#### Step 1: Extract subject-verb pairs\n\n")
            
            f.write("**Reference Subject-Verb Pairs:**\n")
            if ref_pairs:
                for subject, verb in ref_pairs:
                    f.write(f"- ({subject}, {verb})\n")
            else:
                f.write("- No subject-verb pairs found\n")
            f.write("\n")
            
            f.write("**Hypothesis Subject-Verb Pairs:**\n")
            if hyp_pairs:
                for subject, verb in hyp_pairs:
                    f.write(f"- ({subject}, {verb})\n")
            else:
                f.write("- No subject-verb pairs found\n")
            f.write("\n")
            
            f.write("#### Step 2: Identify matching pairs\n\n")
            
            matching_pairs = [pair for pair in hyp_pairs if pair in ref_pairs]
            f.write("**Matching pairs:**\n")
            if matching_pairs:
                for subject, verb in matching_pairs:
                    f.write(f"- ({subject}, {verb})\n")
            else:
                f.write("- No matching pairs found\n")
            f.write("\n")
            
            f.write("#### Step 3: Calculate matching percentage\n\n")
            f.write(f"- Matching pairs: {len(matching_pairs)}\n")
            f.write(f"- Total reference pairs: {len(ref_pairs)}\n")
            f.write(f"- Calculation: {len(matching_pairs)}/{len(ref_pairs) if len(ref_pairs) > 0 else 1} = {sv_acc:.4f} = **{sv_acc*100:.2f}%**\n\n")
            
            # 4. Tense Match Analysis
            f.write("### 4. Tense Match Accuracy\n\n")
            f.write("#### Step 1: Extract verbs with tense information\n\n")
            
            f.write("**Reference Verb Tenses:**\n")
            if ref_tenses:
                for verb, tense in ref_tenses:
                    f.write(f"- ({verb}, {tense})\n")
            else:
                f.write("- No verb tense information found\n")
            f.write("\n")
            
            f.write("**Hypothesis Verb Tenses:**\n")
            if hyp_tenses:
                for verb, tense in hyp_tenses:
                    f.write(f"- ({verb}, {tense})\n")
            else:
                f.write("- No verb tense information found\n")
            f.write("\n")
            
            f.write("#### Step 2: Identify matching tenses\n\n")
            
            # Create dictionaries to easily look up tenses for verbs
            ref_tense_dict = {verb: tense for verb, tense in ref_tenses}
            hyp_tense_dict = {verb: tense for verb, tense in hyp_tenses}
            
            # Count matches for verbs that appear in both reference and hypothesis
            matching_tenses = []
            for verb in ref_tense_dict:
                if verb in hyp_tense_dict and ref_tense_dict[verb] == hyp_tense_dict[verb]:
                    matching_tenses.append((verb, ref_tense_dict[verb]))
            
            f.write("**Matching tenses:**\n")
            if matching_tenses:
                for verb, tense in matching_tenses:
                    f.write(f"- ({verb}, {tense})\n")
            else:
                f.write("- No matching tenses found\n")
            f.write("\n")
            
            f.write("#### Step 3: Calculate matching percentage\n\n")
            f.write(f"- Matching tenses: {len(matching_tenses)}\n")
            f.write(f"- Total reference tenses: {len(ref_tenses)}\n")
            f.write(f"- Calculation: {len(matching_tenses)}/{len(ref_tenses) if len(ref_tenses) > 0 else 1} = {tense_acc:.4f} = **{tense_acc*100:.2f}%**\n\n")
            
            # 5. Non-Repetition Ratio Analysis
            f.write("### 5. Non-Repetition Ratio\n\n")
            f.write("#### Step 1: Analyze word distribution\n\n")
            
            # Get the tokenized words
            if nlp is not None:
                doc = nlp(hyp.lower())
                words = [token.text for token in doc if not token.is_punct and not token.is_space]
            else:
                words = hyp.lower().split()
            
            # Count word occurrences
            word_counts = Counter(words)
            total_words = len(words)
            unique_words = len(word_counts)
            repeated_words = [(word, count) for word, count in word_counts.items() if count > 1]
            
            f.write("**Hypothesis Word Count:**\n")
            f.write(f"- Total words: {total_words}\n")
            f.write(f"- Unique words: {unique_words}\n")
            f.write("- Repeated words:\n")
            if repeated_words:
                for word, count in repeated_words:
                    f.write(f"  - \"{word}\" (appears {count} times)\n")
            else:
                f.write("  - No repeated words found\n")
            f.write("\n")
            
            f.write("#### Step 2: Calculate unique word ratio\n\n")
            f.write(f"- Unique words / Total words: {unique_words}/{total_words} = {unique_ratio:.4f} = **{unique_ratio*100:.2f}%**\n\n")
            
            # Calculate sentence-level BLEU
            try:
                bleu_score = sacrebleu.sentence_bleu(hyp, [ref])
                
                f.write("### BLEU Score Analysis\n\n")
                f.write(f"- BLEU score: **{bleu_score.score:.2f}**\n")
                f.write(f"- Precisions: {[f'{p:.2f}%' for p in bleu_score.precisions]}\n")
                f.write(f"- Brevity penalty: {bleu_score.bp:.2f}\n\n")
            except:
                f.write("### BLEU Score Analysis\n\n")
                f.write("- BLEU score calculation failed\n\n")
            
            f.write("---\n\n")
        
        # Add summary interpretation
        f.write("## Interpretation Guide\n\n")
        f.write("The metrics above provide a multi-dimensional evaluation of translation quality:\n\n")
        f.write("1. **Noun Match Accuracy** measures how well the translation preserves key objects, concepts, and entities from the source text.\n\n")
        f.write("2. **Verb Match Accuracy** measures how well the translation preserves actions and states from the source text.\n\n")
        f.write("3. **Subject-Verb Agreement Accuracy** measures grammatical correctness, ensuring subjects are paired with appropriate verb forms.\n\n")
        f.write("4. **Tense Match Accuracy** measures preservation of temporal information, ensuring the timeframe is correctly conveyed.\n\n")
        f.write("5. **Non-Repetition Ratio** detects unnatural repetitions that are a common failure mode in neural machine translation.\n\n")
        f.write("In combination, these metrics provide a more nuanced view of translation quality than BLEU alone can offer.\n")

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read input files
    references = read_file(args.reference)
    hypotheses = read_file(args.hypothesis)
    sources = read_file(args.source) if args.source else None
    
    # Check if the number of lines match
    if len(references) != len(hypotheses):
        print(f"Warning: Number of references ({len(references)}) does not match number of hypotheses ({len(hypotheses)})")
        # Truncate to the shorter length
        min_len = min(len(references), len(hypotheses))
        references = references[:min_len]
        hypotheses = hypotheses[:min_len]
        if sources:
            sources = sources[:min_len]
    
    # Load spaCy model
    print(f"Loading language model for {args.language}...")
    nlp = load_spacy_model(args.language)
    
    # Evaluate translations
    print("Evaluating linguistic metrics...")
    linguistic_results = evaluate_translations(references, hypotheses, nlp)
    
    # Always calculate corpus-level BLEU score
    print("Calculating corpus-level BLEU score...")
    bleu_score = calculate_bleu(references, hypotheses)
    
    # Prepare results
    results = {
        "model_name": args.model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "reference_file": os.path.basename(args.reference),
            "hypothesis_file": os.path.basename(args.hypothesis),
            "source_file": os.path.basename(args.source) if args.source else None,
            "num_sentences": len(references)
        },
        "average_scores": linguistic_results["average_scores"],
        "sentence_scores": linguistic_results["sentence_scores"],
        "bleu": bleu_score  # Always include BLEU score
    }
    
    # Print summary results
    print("\n===== Linguistic Metrics Evaluation Results =====\n")
    for metric, score in results["average_scores"].items():
        print(f"{metric.replace('_', ' ').title()}: {score:.2f}")
    
    print(f"Corpus BLEU score: {bleu_score.score:.2f}")
    
    print("\n")
    
    # Select random samples for the report
    num_samples = min(args.num_samples, len(references))
    sample_indices = random.sample(range(len(references)), num_samples)
    
    samples = []
    for idx in sample_indices:
        sample = {
            "reference": references[idx],
            "hypothesis": hypotheses[idx]
        }
        
        if sources:
            sample["source"] = sources[idx]
        
        # Add metrics for this sample (keeping original 0-1 scale for internal calculations)
        for metric in results["sentence_scores"]:
            sample[metric] = results["sentence_scores"][metric][idx]
        
        # No need to calculate sentence-level BLEU again as it's already in results["sentence_scores"]
        
        samples.append(sample)
    
    # Print some example translations
    print("===== Sample Translations =====\n")
    for i, idx in enumerate(sample_indices[:5]):  # Show only first 5 in console
        print(f"Example {i+1}:")
        if sources:
            print(f"Source: {sources[idx]}")
        print(f"Reference: {references[idx]}")
        print(f"Hypothesis: {hypotheses[idx]}")
        
        for metric in results["sentence_scores"]:
            # Convert to 0-100 scale for display
            score = results["sentence_scores"][metric][idx] * 100
            print(f"{metric.replace('_', ' ').title()}: {score:.2f}")
        
        print()
    
    # Save results
    json_path = os.path.join(args.output_dir, "linguistic_evaluation_results.json")
    csv_path = os.path.join(args.output_dir, "linguistic_evaluation_metrics.csv")
    sentence_scores_path = os.path.join(args.output_dir, "linguistic_sentence_scores.csv")
    markdown_path = os.path.join(args.output_dir, "linguistic_evaluation_report.md")
    detailed_report_path = os.path.join(args.output_dir, "linguistic_evaluation_detailed.md")
    
    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    save_sentence_scores(results, sentence_scores_path)
    
    # Generate markdown report
    generate_markdown_report(results, samples, markdown_path)
    
    # Generate detailed step-by-step report if requested
    if args.detailed_report:
        print("Generating detailed linguistic analysis report...")
        detailed_samples = min(args.detailed_samples, len(sample_indices))
        generate_detailed_report(references, hypotheses, nlp, sample_indices[:detailed_samples], detailed_report_path)
        print(f"Detailed analysis report: {detailed_report_path}")
    
    print(f"Evaluation results saved to {args.output_dir}")
    print(f"Full report: {markdown_path}")

if __name__ == '__main__':
    main() 