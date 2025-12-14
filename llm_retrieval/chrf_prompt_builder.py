#!/usr/bin/env python3
"""
Improved utility for demonstrating prompt building with chrF++ retrieval.
This script correctly handles prompt templates that already contain examples
and saves the result to a file with dynamic naming based on the input.
"""

import json
import sys
import os
import re
import datetime
from chrf_retriever import ChrfRetriever, load_from_eval_files
import random

def read_template(template_file):
    """Read a prompt template from file."""
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading template file {template_file}: {e}")
        return None

def build_prompt_with_input(template, input_text):
    """Build a prompt by replacing the input_text placeholder in the template."""
    prompt = template.replace("{input_text}", input_text)
    return prompt

def generate_custom_few_shot_prompt(template_file, input_text, chrf_examples):
    """Generate a custom few-shot prompt by replacing the standard examples with retrieved examples."""
    template = read_template(template_file)
    if not template:
        return None
    
    lines = template.split('\n')
    
    before_examples = []
    after_examples = []
    in_examples_section = False
    example_section_end = False
    
    for line in lines:
        if "Here are some examples" in line:
            in_examples_section = True
            before_examples.append(line)
        elif "Now translate" in line:
            example_section_end = True
            after_examples.append(line)
        elif not in_examples_section:
            before_examples.append(line)
        elif example_section_end:
            after_examples.append(line)
    
    examples_text = "\n\n"
    for i, example in enumerate(chrf_examples):
        examples_text += f"Example {i+1}:\n"
        examples_text += f"Input: {example['input']}\n"
        examples_text += f"<translation>{example['expected']}</translation>\n\n"
    
    custom_template = "\n".join(before_examples) + examples_text + "\n".join(after_examples)
    
    prompt = custom_template.replace("{input_text}", input_text)
    
    return prompt

def generate_output_filename(input_text, template_type="few_shot", output_dir="prompts"):
    """Generate a dynamic output filename based on the input text and template type."""
    clean_input = re.sub(r'[^\w\s-]', '', input_text)
    words = clean_input.split()
    filename_base = "_".join(words[:3])
    
    if not filename_base:
        filename_base = "prompt"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return os.path.join(output_dir, f"chrf_{template_type}_{filename_base}_{timestamp}.txt")

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python chrf_prompt_builder.py <input_text> [output_file] [--template advanced|few_shot]")
        print("Example: python chrf_prompt_builder.py \"EU-NOM-PROP-ABBR and-CNJ-COO Serbia-NOM-PROP a-DET-INDEF agreement-NOM on-V-PST-3S\" [output_file.txt] [--template advanced]")
        print("\nOptions:")
        print("  --template: Specify the prompt template to use (default: few_shot)")
        print("    - 'few_shot': Use the standard few-shot prompt template")
        print("    - 'advanced': Use the advanced few-shot prompt template")
        print("\nIf output_file is not specified, a dynamic filename will be generated in the 'prompts' directory.")
        return 1
    
    input_text = sys.argv[1]
    output_file = None
    template_type = "few_shot"
    
    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "--template" and i + 1 < len(sys.argv):
            template_value = sys.argv[i + 1].lower()
            if template_value in ["advanced", "few_shot"]:
                template_type = template_value
            else:
                print(f"Invalid template type: {template_value}. Using default: few_shot")
        elif not arg.startswith("--") and output_file is None:
            output_file = arg
    
    if output_file is None:
        output_file = generate_output_filename(input_text, template_type)
    
    if template_type == "advanced":
        template_file = 'advanced_few_shot_prompt.txt'
    else:
        template_file = 'few_shot_prompt.txt'
    
    print(f"Using template: {template_file}")
    
    try:
        with open('examples.json', 'r', encoding='utf-8') as f:
            examples_exist = True
    except FileNotFoundError:
        examples_exist = False
    
    if not examples_exist:
        print("Generating examples file from eval_gloss.txt and eval_translation.txt...")
        examples = load_from_eval_files()
        if not examples:
            print("Error: Could not load examples.")
            return 1
    
    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' not found.")
        return 1
    
    print(f"Building prompt for input: {input_text}")
    
    retriever = ChrfRetriever('examples.json', n_examples=20)
    retrieved_examples = retriever.retrieve(input_text)
    
    chrf_examples = [ex for ex in retrieved_examples if ex['input'].lower() != input_text.lower()]
    
    if len(chrf_examples) < 3:
        try:
            print("Need more examples after filtering duplicates, calculating similarity for all examples...")
            with open('examples.json', 'r', encoding='utf-8') as f:
                all_examples = json.load(f)
            
            scored_examples = []
            for example in all_examples:
                if example['input'].lower() == input_text.lower():
                    continue
                    
                if example in chrf_examples:
                    continue
                    
                score = retriever._compute_chrf_score(input_text, example['input'])
                scored_examples.append((score, example))
            
            scored_examples.sort(reverse=True, key=lambda x: x[0])
            
            remaining_needed = 3 - len(chrf_examples)
            additional_examples = [example for _, example in scored_examples[:remaining_needed]]
            
            if additional_examples:
                print(f"Found {len(additional_examples)} additional similar examples.")
                chrf_examples.extend(additional_examples)
        except Exception as e:
            print(f"Error finding additional examples: {e}")
    
    chrf_examples = chrf_examples[:3]
    
    print("\nRetrieved examples with chrF++ scores:")
    for i, example in enumerate(chrf_examples):
        score = retriever._compute_chrf_score(input_text, example['input'])
        print(f"  Example {i+1} (similarity score: {score:.4f}):")
        print(f"    Input: {example['input']}")
        print(f"    Expected: {example['expected']}")
    
    custom_prompt = generate_custom_few_shot_prompt(template_file, input_text, chrf_examples)
    
    if not custom_prompt:
        print("Error generating custom prompt, falling back to standard prompt")
        standard_template = read_template(template_file)
        custom_prompt = build_prompt_with_input(standard_template, input_text)
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(custom_prompt)
        
        print(f"\nPrompt saved to: {output_file}")
    except Exception as e:
        print(f"Error saving prompt to file: {e}")
        print("\n=== PROMPT WITH CHRF++ RETRIEVED EXAMPLES ===\n")
        print(custom_prompt)
        print("\n=== END OF PROMPT ===\n")
        return 1
    
    print("\n=== PROMPT WITH CHRF++ RETRIEVED EXAMPLES ===\n")
    print(custom_prompt)
    print("\n=== END OF PROMPT ===\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 