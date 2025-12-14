#!/usr/bin/env python3
"""
Batch inference script for processing 1000 Turkish-English pairs efficiently.
Includes memory management, progress tracking, and robust error handling.
"""

import os
import json
import torch
import argparse
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
import gc
from tqdm import tqdm
warnings.filterwarnings("ignore")

class BatchAlignmentInference:
    def __init__(self, model_path, output_dir="batch_alignment_outputs", batch_size=10):
        self.model_path = model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/prompts", exist_ok=True)
        os.makedirs(f"{output_dir}/outputs", exist_ok=True)
        os.makedirs(f"{output_dir}/alignments", exist_ok=True)
        os.makedirs(f"{output_dir}/batches", exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.load_model()
        
        self.dictionary = defaultdict(Counter)
        self.all_alignments = []
        self.processed_pairs = []
        self.failed_pairs = []
        
    def load_model(self):
        """Load the fine-tuned alignment model."""
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        peft_model_path = os.path.join(self.model_path, "peft_model")
        self.model = PeftModel.from_pretrained(base_model, peft_model_path)
        self.model.eval()
        print("✓ Model loaded successfully")
    
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
    
    def process_batch(self, batch_pairs, batch_id):
        """Process a batch of sentence pairs."""
        batch_results = []
        
        for i, (turkish, english) in enumerate(batch_pairs):
            pair_id = batch_id * self.batch_size + i + 1
            
            try:
                prompt = self.create_prompt(turkish, english)
                
                prompt_file = f"{self.output_dir}/prompts/prompt_{pair_id:04d}.txt"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
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
                    "batch_id": batch_id,
                    "turkish_sentence": turkish,
                    "english_sentence": english,
                    "alignments": alignments,
                    "raw_output": generated_part,
                    "num_alignments": len(alignments)
                }
                
                with open(alignment_file, 'w', encoding='utf-8') as f:
                    json.dump(alignment_data, f, ensure_ascii=False, indent=2)
                
                self.all_alignments.extend(alignments)
                batch_results.append(alignment_data)
                
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                error_data = {
                    "pair_id": pair_id,
                    "batch_id": batch_id,
                    "turkish_sentence": turkish,
                    "english_sentence": english,
                    "error": str(e)
                }
                self.failed_pairs.append(error_data)
                print(f"  ✗ Error processing pair {pair_id}: {e}")
                continue
        
        batch_file = f"{self.output_dir}/batches/batch_{batch_id:03d}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        return batch_results
    
    def load_parallel_data(self, turkish_file, english_file, max_pairs=1000):
        """Load parallel Turkish-English data."""
        print(f"Loading parallel data (max {max_pairs} pairs)...")
        
        turkish_sentences = []
        english_sentences = []
        
        with open(turkish_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_pairs:
                    break
                line = line.strip()
                if line:
                    turkish_sentences.append(line)
        
        with open(english_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_pairs:
                    break
                line = line.strip()
                if line:
                    english_sentences.append(line)
        
        min_length = min(len(turkish_sentences), len(english_sentences))
        
        pairs = []
        for i in range(min_length):
            if turkish_sentences[i] and english_sentences[i]:
                pairs.append((turkish_sentences[i], english_sentences[i]))
        
        print(f"✓ Loaded {len(pairs)} sentence pairs")
        return pairs[:max_pairs]
    
    def process_all_pairs(self, pairs):
        """Process all sentence pairs in batches."""
        print(f"Processing {len(pairs)} sentence pairs in batches of {self.batch_size}...")
        
        batches = [pairs[i:i + self.batch_size] for i in range(0, len(pairs), self.batch_size)]
        
        for batch_id, batch_pairs in enumerate(tqdm(batches, desc="Processing batches")):
            print(f"\nProcessing batch {batch_id + 1}/{len(batches)} ({len(batch_pairs)} pairs)")
            
            batch_results = self.process_batch(batch_pairs, batch_id)
            self.processed_pairs.extend(batch_results)
            
            if (batch_id + 1) % 10 == 0:
                self.save_intermediate_results()
                print(f"  ✓ Saved intermediate results after batch {batch_id + 1}")
            
            gc.collect()
            torch.cuda.empty_cache()
    
    def save_intermediate_results(self):
        """Save intermediate results."""
        dict_file = f"{self.output_dir}/dictionary_intermediate_{self.timestamp}.json"
        dict_data = {}
        for turkish_word, english_counts in self.dictionary.items():
            dict_data[turkish_word] = dict(english_counts)
        
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(dict_data, f, ensure_ascii=False, indent=2)
        
        if self.failed_pairs:
            failed_file = f"{self.output_dir}/failed_pairs_{self.timestamp}.json"
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_pairs, f, ensure_ascii=False, indent=2)
        
        progress_file = f"{self.output_dir}/progress_{self.timestamp}.json"
        progress_data = {
            "processed_pairs": len(self.processed_pairs),
            "failed_pairs": len(self.failed_pairs),
            "total_alignments": len(self.all_alignments),
            "unique_turkish_words": len(self.dictionary),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    def build_final_dictionary(self, min_frequency=1):
        """Build the final dictionary with frequency filtering."""
        print(f"Building final dictionary (min frequency: {min_frequency})...")
        
        dictionaries = {
            "full_with_counts": {},
            "best_translation": {},
            "all_translations": {},
            "statistics": {},
            "quality_metrics": {}
        }
        
        total_alignments = len(self.all_alignments)
        unique_turkish_words = len(self.dictionary)
        
        for turkish_word, english_counts in self.dictionary.items():
            filtered_counts = {en: count for en, count in english_counts.items() 
                             if count >= min_frequency}
            
            if not filtered_counts:
                continue
            
            dictionaries["full_with_counts"][turkish_word] = filtered_counts
            
            best_english = max(filtered_counts.items(), key=lambda x: x[1])
            dictionaries["best_translation"][turkish_word] = best_english[0]
            
            all_translations = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
            dictionaries["all_translations"][turkish_word] = [en for en, _ in all_translations]
        
        alignment_counts = Counter(self.all_alignments)
        most_common_alignments = alignment_counts.most_common(50)
        
        dictionaries["quality_metrics"] = {
            "most_frequent_alignments": most_common_alignments,
            "alignment_frequency_distribution": dict(Counter(alignment_counts.values())),
            "words_with_multiple_translations": sum(1 for counts in dictionaries["full_with_counts"].values() if len(counts) > 1),
            "average_translations_per_word": sum(len(counts) for counts in dictionaries["full_with_counts"].values()) / len(dictionaries["full_with_counts"]) if dictionaries["full_with_counts"] else 0
        }
        
        dictionaries["statistics"] = {
            "total_sentence_pairs": len(self.processed_pairs),
            "failed_pairs": len(self.failed_pairs),
            "success_rate": len(self.processed_pairs) / (len(self.processed_pairs) + len(self.failed_pairs)) * 100 if (len(self.processed_pairs) + len(self.failed_pairs)) > 0 else 0,
            "total_alignments": total_alignments,
            "unique_turkish_words": unique_turkish_words,
            "dictionary_entries": len(dictionaries["best_translation"]),
            "average_alignments_per_pair": total_alignments / len(self.processed_pairs) if self.processed_pairs else 0,
            "timestamp": self.timestamp,
            "batch_size": self.batch_size
        }
        
        return dictionaries
    
    def save_final_results(self, dictionaries):
        """Save all final results."""
        print("Saving final results...")
        
        dict_file = f"{self.output_dir}/turkish_english_dictionary_{self.timestamp}.json"
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(dictionaries, f, ensure_ascii=False, indent=2)
        
        simple_dict_file = f"{self.output_dir}/dictionary_simple_{self.timestamp}.txt"
        with open(simple_dict_file, 'w', encoding='utf-8') as f:
            f.write("# Turkish-English Dictionary\n")
            f.write(f"# Generated: {self.timestamp}\n")
            f.write(f"# Total entries: {len(dictionaries['best_translation'])}\n")
            f.write(f"# Success rate: {dictionaries['statistics']['success_rate']:.1f}%\n\n")
            
            for turkish, english in sorted(dictionaries["best_translation"].items()):
                f.write(f"{turkish} → {english}\n")
        
        csv_file = f"{self.output_dir}/dictionary_detailed_{self.timestamp}.csv"
        rows = []
        for turkish, english_counts in dictionaries["full_with_counts"].items():
            for english, count in english_counts.items():
                rows.append({
                    "turkish": turkish,
                    "english": english,
                    "frequency": count,
                    "is_primary": english == dictionaries["best_translation"].get(turkish, ""),
                    "confidence": count / sum(english_counts.values())
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        stats_file = f"{self.output_dir}/comprehensive_statistics_{self.timestamp}.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=== BATCH ALIGNMENT INFERENCE STATISTICS ===\n\n")
            
            f.write("BASIC STATISTICS:\n")
            for key, value in dictionaries["statistics"].items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nQUALITY METRICS:\n")
            for key, value in dictionaries["quality_metrics"].items():
                if key != "most_frequent_alignments":
                    f.write(f"  {key}: {value}\n")
            
            f.write(f"\nTOP 20 MOST FREQUENT ALIGNMENTS:\n")
            for (tr, en), count in dictionaries["quality_metrics"]["most_frequent_alignments"][:20]:
                f.write(f"  {tr} → {en}: {count} times\n")
            
            if self.failed_pairs:
                f.write(f"\nFAILED PAIRS SUMMARY:\n")
                f.write(f"  Total failed: {len(self.failed_pairs)}\n")
                error_types = Counter(fp.get("error", "Unknown") for fp in self.failed_pairs)
                for error, count in error_types.most_common():
                    f.write(f"  {error}: {count} times\n")
        
        print(f"✓ Results saved to {self.output_dir}/")
        print(f"  - Dictionary: {dict_file}")
        print(f"  - Simple format: {simple_dict_file}")
        print(f"  - CSV format: {csv_file}")
        print(f"  - Statistics: {stats_file}")
        
        return dictionaries

def main():
    parser = argparse.ArgumentParser(description="Batch Turkish-English dictionary builder")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned alignment model")
    parser.add_argument("--turkish_file", type=str,
                       default="parallel_data_40k/test_transcription.txt",
                       help="Path to Turkish sentences file")
    parser.add_argument("--english_file", type=str,
                       default="parallel_data_40k/test_translation.txt",
                       help="Path to English sentences file")
    parser.add_argument("--max_pairs", type=int, default=1000,
                       help="Maximum number of sentence pairs to process")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, default="batch_alignment_outputs",
                       help="Output directory for results")
    parser.add_argument("--min_frequency", type=int, default=1,
                       help="Minimum frequency for dictionary entries")
    
    args = parser.parse_args()
    
    print("=== BATCH TURKISH-ENGLISH DICTIONARY BUILDER ===")
    print(f"Model: {args.model_path}")
    print(f"Turkish file: {args.turkish_file}")
    print(f"English file: {args.english_file}")
    print(f"Max pairs: {args.max_pairs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    inference = BatchAlignmentInference(args.model_path, args.output_dir, args.batch_size)
    
    pairs = inference.load_parallel_data(args.turkish_file, args.english_file, args.max_pairs)
    
    inference.process_all_pairs(pairs)
    
    dictionaries = inference.build_final_dictionary(args.min_frequency)
    
    final_results = inference.save_final_results(dictionaries)
    
    print("\n=== FINAL STATISTICS ===")
    stats = final_results["statistics"]
    print(f"Processed {stats['total_sentence_pairs']} sentence pairs")
    print(f"Failed {stats['failed_pairs']} pairs ({100-stats['success_rate']:.1f}%)")
    print(f"Generated {stats['total_alignments']} total alignments")
    print(f"Created dictionary with {stats['dictionary_entries']} entries")
    print(f"Average {stats['average_alignments_per_pair']:.1f} alignments per sentence pair")
    
    print("\n=== SAMPLE DICTIONARY ENTRIES ===")
    sample_entries = list(final_results["best_translation"].items())[:10]
    for turkish, english in sample_entries:
        print(f"{turkish} → {english}")
    
    print(f"\n✓ Batch processing complete! Check {args.output_dir}/ for all results.")

if __name__ == "__main__":
    main() 