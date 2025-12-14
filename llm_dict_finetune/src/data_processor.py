#!/usr/bin/env python3
"""
Data processor for Turkish-English word alignment training data.
Processes parallel sentences and dictionary to create training examples.
"""

import os
import re
import json
import random
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

class DataProcessor:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.turkish_sentences = []
        self.english_sentences = []
        self.dictionary = {}
        self.lemma_dict = {}
        self.turkish_lemmas = []
        
    def load_parallel_data(self):
        """Load parallel Turkish-English sentences."""
        print("Loading parallel data...")
        
        turkish_file = os.path.join(self.data_dir, "parallel_data_40k", "merged_transcriptions_40k.txt")
        with open(turkish_file, 'r', encoding='utf-8') as f:
            self.turkish_sentences = [line.strip() for line in f if line.strip()]
        
        english_file = os.path.join(self.data_dir, "parallel_data_40k", "merged_translations_40k.txt")
        with open(english_file, 'r', encoding='utf-8') as f:
            self.english_sentences = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.turkish_sentences)} Turkish sentences")
        print(f"Loaded {len(self.english_sentences)} English sentences")
        
        min_len = min(len(self.turkish_sentences), len(self.english_sentences))
        self.turkish_sentences = self.turkish_sentences[:min_len]
        self.english_sentences = self.english_sentences[:min_len]
        
    def load_dictionary(self):
        """Load Turkish-English dictionary and English lemmatization dictionary."""
        print("Loading dictionary...")
        
        dict_file = os.path.join(self.data_dir, "dict_40k.txt")
        with open(dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ' -> ' in line:
                    turkish, english = line.strip().split(' -> ', 1)
                    self.dictionary[turkish.strip()] = english.strip()
        
        print(f"Loaded {len(self.dictionary)} Turkish-English dictionary entries")
        
        self.english_lemma_dict = {}
        lemma_dict_file = os.path.join(self.data_dir, "lemmatized_dict.txt")
        with open(lemma_dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ' -> ' in line:
                    inflected, lemma = line.strip().split(' -> ', 1)
                    if inflected.strip() != lemma.strip() and len(inflected.strip()) > 1:
                        self.english_lemma_dict[inflected.strip().lower()] = lemma.strip().lower()
        
        print(f"Loaded {len(self.english_lemma_dict)} English lemmatization entries")
        
    def load_lemma_data(self):
        """Load Turkish lemma data and create inflected->lemma mapping."""
        print("Loading lemma data...")
        
        lemma_file = os.path.join(self.data_dir, "glosslm_lemma.txt")
        with open(lemma_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            self.turkish_lemmas = content.split('\n')
        
        self.inflected_to_lemma = {}
        
        min_len = min(len(self.turkish_lemmas), len(self.turkish_sentences))
        for i in range(min_len):
            if i < len(self.turkish_lemmas) and i < len(self.turkish_sentences):
                inflected_tokens = self.tokenize_simple(self.turkish_sentences[i])
                lemma_tokens = self.turkish_lemmas[i].split()
                
                if len(inflected_tokens) == len(lemma_tokens):
                    for inf_token, lemma_token in zip(inflected_tokens, lemma_tokens):
                        if inf_token != lemma_token:
                            self.inflected_to_lemma[inf_token] = lemma_token
        
        print(f"Loaded {len(self.turkish_lemmas)} lemma sentences")
        print(f"Created {len(self.inflected_to_lemma)} inflected->lemma mappings")
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text.strip())
        return text
        
    def tokenize_simple(self, text: str) -> List[str]:
        """Simple tokenization for Turkish and English."""
        tokens = re.findall(r'\b\w+\b|\d+', text.lower())
        return tokens
        
    def get_lemma(self, word: str) -> str:
        """Get lemma form of a Turkish word, fallback to original if not found."""
        return self.inflected_to_lemma.get(word.lower(), word.lower())
        
    def get_english_lemma(self, word: str) -> str:
        """Get English lemma using the lemmatized dictionary, fallback to simple rules."""
        word_lower = word.lower()
        
        if word_lower in self.english_lemma_dict:
            return self.english_lemma_dict[word_lower]
        
        if word_lower.endswith('s') and len(word_lower) > 3 and not word_lower.endswith('ss'):
            return word_lower[:-1]
        elif word_lower.endswith('ed') and len(word_lower) > 4:
            return word_lower[:-2]
        elif word_lower.endswith('ing') and len(word_lower) > 5:
            return word_lower[:-3]
        
        return word_lower
        
    def find_alignments(self, turkish_tokens: List[str], english_tokens: List[str]) -> List[Tuple[str, str]]:
        """Find word alignments using the dictionary and return lemma forms."""
        alignments = []
        
        for tr_token in turkish_tokens:
            tr_lemma = self.get_lemma(tr_token)
            
            if tr_token in self.dictionary:
                eng_word = self.dictionary[tr_token]
                if eng_word.lower() in [t.lower() for t in english_tokens]:
                    eng_lemma = self.get_english_lemma(eng_word)
                    alignments.append((tr_lemma, eng_lemma))
                    continue
            
            # Direct lookup with lemma
            if tr_lemma in self.dictionary:
                eng_word = self.dictionary[tr_lemma]
                if eng_word.lower() in [t.lower() for t in english_tokens]:
                    eng_lemma = self.get_english_lemma(eng_word)
                    alignments.append((tr_lemma, eng_lemma))
                    continue
            
            for dict_tr, dict_en in self.dictionary.items():
                if (tr_token.startswith(dict_tr) or dict_tr.startswith(tr_token)) and len(dict_tr) > 2:
                    if dict_en.lower() in [t.lower() for t in english_tokens]:
                        eng_lemma = self.get_english_lemma(dict_en)
                        alignments.append((tr_lemma, eng_lemma))
                        break
        
        return alignments
        
    def find_alignments_with_lemmas(self, turkish_tokens: List[str], turkish_lemma_tokens: List[str], english_tokens: List[str]) -> List[Tuple[str, str]]:
        """Find word alignments using lemmatized Turkish tokens and return lemma forms."""
        alignments = []
        
        token_to_lemma = {}
        if len(turkish_tokens) == len(turkish_lemma_tokens):
            for inflected, lemma in zip(turkish_tokens, turkish_lemma_tokens):
                token_to_lemma[inflected] = lemma
        
        for i, tr_token in enumerate(turkish_tokens):
            if tr_token in token_to_lemma:
                tr_lemma = token_to_lemma[tr_token]
            elif i < len(turkish_lemma_tokens):
                tr_lemma = turkish_lemma_tokens[i]
            else:
                tr_lemma = self.get_lemma(tr_token)
            
            if tr_token in self.dictionary:
                eng_word = self.dictionary[tr_token]
                if eng_word.lower() in [t.lower() for t in english_tokens]:
                    eng_lemma = self.get_english_lemma(eng_word)
                    alignments.append((tr_lemma, eng_lemma))
                    continue
            
            for dict_tr, dict_en in self.dictionary.items():
                if (tr_token.startswith(dict_tr) or dict_tr.startswith(tr_token)) and len(dict_tr) > 2:
                    if dict_en.lower() in [t.lower() for t in english_tokens]:
                        eng_lemma = self.get_english_lemma(dict_en)
                        alignments.append((tr_lemma, eng_lemma))
                        break
        
        return alignments
        
    def create_training_example(self, turkish_sent: str, english_sent: str, sentence_idx: int = None) -> Dict:
        """Create a training example in the required format."""
        turkish_clean = self.clean_text(turkish_sent)
        english_clean = self.clean_text(english_sent)
        
        turkish_tokens = self.tokenize_simple(turkish_clean)
        english_tokens = self.tokenize_simple(english_clean)
        
        # Get the corresponding lemmatized Turkish sentence if available
        turkish_lemma_sent = None
        if sentence_idx is not None and sentence_idx < len(self.turkish_lemmas):
            turkish_lemma_sent = self.turkish_lemmas[sentence_idx].strip()
            turkish_lemma_tokens = turkish_lemma_sent.split()
        else:
            turkish_lemma_tokens = [self.get_lemma(token) for token in turkish_tokens]
        
        alignments = self.find_alignments_with_lemmas(turkish_tokens, turkish_lemma_tokens, english_tokens)
        
        if len(alignments) < 2:  # Filter out examples with too few alignments
            return None
            
        # Format alignments as required
        alignment_text = "\n".join([f"{tr} - {en}" for tr, en in alignments])
        
        # Create the prompt with explicit instructions about lemma forms and format
        prompt = f"""### Instruction:
Generate word-level alignments between these two sentences, matching each word in the source language to its corresponding word(s) in the target language. 

IMPORTANT REQUIREMENTS:
1. Use ONLY the lemma (dictionary/root form) of each word, NOT inflected forms
2. Format each alignment as: source_lemma - target_lemma
3. Put each alignment on a separate line
4. Do not include punctuation, numbers, or function words unless they have clear semantic meaning
5. Ensure both words in each pair are in their base dictionary forms
6. Wrap all alignments between <alignments> and </alignments> tags

### Source sentence (Turkish):
{turkish_clean}

### Target sentence (English):
{english_clean}

### Word alignments:
<alignments>
{alignment_text}
</alignments>"""

        return {
            "text": prompt,
            "turkish": turkish_clean,
            "english": english_clean,
            "alignments": alignments,
            "num_alignments": len(alignments)
        }
        
    def process_all_data(self, max_examples: int = None) -> List[Dict]:
        """Process all parallel data to create training examples."""
        print("Processing training examples...")
        
        training_examples = []
        
        pairs = list(zip(self.turkish_sentences, self.english_sentences))
        if max_examples:
            pairs = pairs[:max_examples]
            
        for i, (turkish_sent, english_sent) in enumerate(tqdm(pairs, desc="Creating examples")):
            example = self.create_training_example(turkish_sent, english_sent, sentence_idx=i)
            if example:
                training_examples.append(example)
                
        print(f"Created {len(training_examples)} training examples")
        return training_examples
        
    def save_training_data(self, examples: List[Dict], output_file: str = "training_data.jsonl"):
        """Save training examples to JSONL format."""
        output_path = os.path.join(self.data_dir, "outputs", output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"Saved {len(examples)} examples to {output_path}")
        
    def get_statistics(self, examples: List[Dict]):
        """Print statistics about the training data."""
        if not examples:
            return
            
        num_alignments = [ex["num_alignments"] for ex in examples]
        
        print("\n=== Training Data Statistics ===")
        print(f"Total examples: {len(examples)}")
        print(f"Average alignments per example: {sum(num_alignments) / len(num_alignments):.2f}")
        print(f"Min alignments: {min(num_alignments)}")
        print(f"Max alignments: {max(num_alignments)}")
        
        # Show a sample
        print("\n=== Sample Training Example ===")
        print(examples[0]["text"])

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Turkish-English parallel data for training")
    parser.add_argument("--max_examples", type=int, default=5000,
                       help="Maximum number of training examples to process")
    parser.add_argument("--output_file", type=str, default="training_data.jsonl",
                       help="Output file name for training data")
    
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    # Load all data
    processor.load_parallel_data()
    processor.load_dictionary()
    processor.load_lemma_data()
    
    # Process training examples
    examples = processor.process_all_data(max_examples=args.max_examples)
    
    # Save training data
    processor.save_training_data(examples, args.output_file)
    
    # Print statistics
    processor.get_statistics(examples)

if __name__ == "__main__":
    main() 