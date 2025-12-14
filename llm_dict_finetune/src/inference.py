#!/usr/bin/env python3
"""
Simple clean inference script - basic generation with post-processing cleanup.
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def clean_alignments(text):
    """Clean up generated alignments text."""
    text = text.strip()
    
    lines = text.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('[') or '[ - the' in line:
            break
            
        if any(stop_word in line.lower() for stop_word in ['instruction:', 'you are', '###', 'source sentence', 'target sentence']):
            break
            
        if ' - ' in line and len(line.split(' - ')) == 2:
            parts = line.split(' - ')
            if len(parts[0].strip()) > 0 and len(parts[1].strip()) > 0:
                clean_lines.append(line)
    
    return clean_lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--turkish", type=str, required=True)
    parser.add_argument("--english", type=str, required=True)
    args = parser.parse_args()
    
    print("=== SIMPLE CLEAN INFERENCE ===")
    print(f"Turkish: {args.turkish}")
    print(f"English: {args.english}")
    print()
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    peft_model_path = os.path.join(args.model_path, "peft_model")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.eval()
    print("✓ Model loaded")
    
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
{args.turkish}

### Target sentence (English):
{args.english}

### Word alignments:
<alignments>"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("Generating alignments...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Conservative limit
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = full_output[len(prompt):].strip()
    
    print("Raw generated text:")
    print(f"'{generated_part}'")
    print()
    
    if "</alignments>" in generated_part:
        alignments_text = generated_part.split("</alignments>")[0].strip()
    else:
        alignments_text = generated_part
    
    if alignments_text.startswith("<alignments>"):
        alignments_text = alignments_text[len("<alignments>"):].strip()
    
    clean_lines = clean_alignments(alignments_text)
    
    print("=== CLEANED ALIGNMENTS ===")
    valid_alignments = []
    
    for line in clean_lines:
        if ' - ' in line:
            turkish_word, english_word = line.split(' - ', 1)
            turkish_word = turkish_word.strip()
            english_word = english_word.strip()
            
            if turkish_word and english_word:
                valid_alignments.append((turkish_word, english_word))
                print(f"✓ {turkish_word} → {english_word}")
    
    print(f"\nFound {len(valid_alignments)} valid alignment pairs")
    
    print("\n=== QUICK TEST ===")
    test_turkish = "Kitap okudum."
    test_english = "I read a book."
    
    test_prompt = prompt.replace(args.turkish, test_turkish).replace(args.english, test_english)
    test_inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=2048)
    test_inputs = {k: v.to(model.device) for k, v in test_inputs.items()}
    
    with torch.no_grad():
        test_outputs = model.generate(
            **test_inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    test_result = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
    test_generated = test_result[len(test_prompt):].strip()
    
    if "</alignments>" in test_generated:
        test_generated = test_generated.split("</alignments>")[0].strip()
    
    print(f"Test '{test_turkish}' → '{test_english}':")
    print(f"Result: {test_generated}")
    
    return valid_alignments

if __name__ == "__main__":
    main() 