#!/usr/bin/env python3
"""
Demo script to test alignment inference pipeline using base Qwen model.
This can be used to test the pipeline before running with a fine-tuned model.
"""

import os
import json
import torch
import argparse
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

class DemoAlignmentInference:
    def __init__(self, output_dir="demo_alignment_outputs"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/prompts", exist_ok=True)
        os.makedirs(f"{output_dir}/outputs", exist_ok=True)
        os.makedirs(f"{output_dir}/alignments", exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.load_base_model()
        
        self.dictionary = defaultdict(Counter)
        self.all_alignments = []
        self.processed_pairs = []
        
    def load_base_model(self):
        """Load the base Qwen model for demo purposes."""
        print("Loading base Qwen model for demo...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        print("✓ Base model loaded successfully")
    
    def create_prompt(self, turkish_sentence, english_sentence):
        """Create alignment prompt for the model."""
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
{turkish_sentence}

### Target sentence (English):
{english_sentence}

### Word alignments:
<alignments>"""
        return prompt
    
    def clean_alignments(self, text):
        """Clean up generated alignments text."""
        text = text.strip()
        
        if "</alignments>" in text:
            text = text.split("</alignments>")[0].strip()
        
        if text.startswith("<alignments>"):
            text = text[len("<alignments>"):].strip()
        
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if (line.startswith('[') or '[ - the' in line or 
                any(stop_word in line.lower() for stop_word in 
                    ['instruction:', 'you are', '###', 'source sentence', 'target sentence'])):
                break
                
            if ' - ' in line and len(line.split(' - ')) == 2:
                parts = line.split(' - ')
                if len(parts[0].strip()) > 0 and len(parts[1].strip()) > 0:
                    clean_lines.append(line)
        
        return clean_lines
    
    def generate_alignment(self, turkish_sentence, english_sentence, pair_id):
        """Generate word alignment for a sentence pair."""
        prompt = self.create_prompt(turkish_sentence, english_sentence)
        
        prompt_file = f"{self.output_dir}/prompts/prompt_{pair_id:04d}.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_output[len(prompt):].strip()
        
        output_file = f"{self.output_dir}/outputs/output_{pair_id:04d}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== RAW OUTPUT ===\n{generated_part}\n\n")
            f.write(f"=== FULL OUTPUT ===\n{full_output}")
        
        clean_lines = self.clean_alignments(generated_part)
        
        alignments = []
        for line in clean_lines:
            if ' - ' in line:
                turkish_word, english_word = line.split(' - ', 1)
                turkish_word = turkish_word.strip()
                english_word = english_word.strip()
                
                if turkish_word and english_word:
                    alignments.append((turkish_word, english_word))
                    self.dictionary[turkish_word][english_word] += 1
        
        alignment_file = f"{self.output_dir}/alignments/alignment_{pair_id:04d}.json"
        alignment_data = {
            "pair_id": pair_id,
            "turkish_sentence": turkish_sentence,
            "english_sentence": english_sentence,
            "alignments": alignments,
            "raw_output": generated_part
        }
        
        with open(alignment_file, 'w', encoding='utf-8') as f:
            json.dump(alignment_data, f, ensure_ascii=False, indent=2)
        
        self.all_alignments.extend(alignments)
        self.processed_pairs.append(alignment_data)
        
        return alignments
    
    def run_demo(self, num_pairs=10):
        """Run demo with sample Turkish-English pairs."""
        print(f"Running demo with {num_pairs} sample pairs...")
        
        demo_pairs = [
            ("Kitap okudum.", "I read a book."),
            ("Ev büyük.", "The house is big."),
            ("Su içiyorum.", "I am drinking water."),
            ("Araba kırmızı.", "The car is red."),
            ("Okula gidiyorum.", "I am going to school."),
            ("Yemek yedim.", "I ate food."),
            ("Müzik dinliyorum.", "I am listening to music."),
            ("Film izledik.", "We watched a movie."),
            ("Bahçede çiçekler var.", "There are flowers in the garden."),
            ("Hava güzel.", "The weather is nice.")
        ]
        
        pairs_to_process = demo_pairs[:num_pairs]
        
        for i, (turkish, english) in enumerate(pairs_to_process):
            print(f"\nProcessing demo pair {i+1}/{len(pairs_to_process)}:")
            print(f"  Turkish: {turkish}")
            print(f"  English: {english}")
            
            try:
                alignments = self.generate_alignment(turkish, english, i+1)
                print(f"  → Generated {len(alignments)} alignments:")
                
                for tr, en in alignments:
                    print(f"    {tr} → {en}")
                    
            except Exception as e:
                print(f"  ✗ Error processing pair {i+1}: {e}")
                continue
        
        return self.build_demo_dictionary()
    
    def build_demo_dictionary(self):
        """Build dictionary from demo results."""
        print("\nBuilding demo dictionary...")
        
        dictionaries = {
            "full_with_counts": {},
            "best_translation": {},
            "all_translations": {},
            "statistics": {}
        }
        
        total_alignments = len(self.all_alignments)
        unique_turkish_words = len(self.dictionary)
        
        for turkish_word, english_counts in self.dictionary.items():
            dictionaries["full_with_counts"][turkish_word] = dict(english_counts)
            
            best_english = max(english_counts.items(), key=lambda x: x[1])
            dictionaries["best_translation"][turkish_word] = best_english[0]
            
            all_translations = sorted(english_counts.items(), key=lambda x: x[1], reverse=True)
            dictionaries["all_translations"][turkish_word] = [en for en, _ in all_translations]
        
        dictionaries["statistics"] = {
            "total_sentence_pairs": len(self.processed_pairs),
            "total_alignments": total_alignments,
            "unique_turkish_words": unique_turkish_words,
            "dictionary_entries": len(dictionaries["best_translation"]),
            "average_alignments_per_pair": total_alignments / len(self.processed_pairs) if self.processed_pairs else 0,
            "timestamp": self.timestamp,
            "model_type": "base_qwen_demo"
        }
        
        dict_file = f"{self.output_dir}/demo_dictionary_{self.timestamp}.json"
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(dictionaries, f, ensure_ascii=False, indent=2)
        
        simple_dict_file = f"{self.output_dir}/demo_dictionary_simple_{self.timestamp}.txt"
        with open(simple_dict_file, 'w', encoding='utf-8') as f:
            f.write("# Turkish-English Demo Dictionary\n")
            f.write(f"# Generated: {self.timestamp}\n")
            f.write(f"# Model: Base Qwen2.5-7B-Instruct (Demo)\n")
            f.write(f"# Total entries: {len(dictionaries['best_translation'])}\n\n")
            
            for turkish, english in sorted(dictionaries["best_translation"].items()):
                f.write(f"{turkish} → {english}\n")
        
        print(f"✓ Demo results saved to {self.output_dir}/")
        print(f"  - Dictionary: {dict_file}")
        print(f"  - Simple format: {simple_dict_file}")
        
        return dictionaries

def main():
    parser = argparse.ArgumentParser(description="Demo Turkish-English dictionary builder")
    parser.add_argument("--num_pairs", type=int, default=10,
                       help="Number of demo pairs to process")
    parser.add_argument("--output_dir", type=str, default="demo_alignment_outputs",
                       help="Output directory for demo results")
    
    args = parser.parse_args()
    
    print("=== TURKISH-ENGLISH DICTIONARY BUILDER DEMO ===")
    print(f"Processing {args.num_pairs} demo pairs")
    print(f"Output directory: {args.output_dir}")
    print("Note: This demo uses the base Qwen model, not a fine-tuned alignment model")
    print()
    
    demo = DemoAlignmentInference(args.output_dir)
    
    results = demo.run_demo(args.num_pairs)
    
    print("\n=== DEMO RESULTS ===")
    stats = results["statistics"]
    print(f"Processed {stats['total_sentence_pairs']} sentence pairs")
    print(f"Generated {stats['total_alignments']} total alignments")
    print(f"Created dictionary with {stats['dictionary_entries']} entries")
    print(f"Average {stats['average_alignments_per_pair']:.1f} alignments per sentence pair")
    
    print("\n=== DEMO DICTIONARY ENTRIES ===")
    for turkish, english in results["best_translation"].items():
        print(f"{turkish} → {english}")
    
    print(f"\n✓ Demo complete! Check {args.output_dir}/ for all results.")
    print("\nTo run with a fine-tuned model, use: python inference_alignment_dictionary.py --model_path <path_to_model>")

if __name__ == "__main__":
    main() 