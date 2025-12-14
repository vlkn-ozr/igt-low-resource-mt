#!/usr/bin/env python3

import json
import re
import sys
import os
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract translations and restructure evaluation data from JSON file')
    parser.add_argument('input_file', help='Input JSON file to process')
    parser.add_argument('--zero-shot', action='store_true', help='Process zero_shot field')
    parser.add_argument('--embeddings-few-shot', action='store_true', help='Process embeddings_few_shot field')
    parser.add_argument('--chrf-advanced-few-shot', action='store_true', help='Process chrf_advanced_few_shot field')
    parser.add_argument('--extract-only', action='store_true', help='Only perform extraction, skip restructuring')
    parser.add_argument('--restructure-only', action='store_true', help='Only perform restructuring, skip extraction')
    parser.add_argument('--outdir', type=str, default=None, help='Directory to save output files (defaults to input file\'s directory)')
    
    args = parser.parse_args()
    
    # If no specific fields are specified, process all by default
    if not any([args.zero_shot, args.embeddings_few_shot, args.chrf_advanced_few_shot]):
        fields_to_process = ["zero_shot", "embeddings_few_shot", "chrf_advanced_few_shot"]
        include_fields = {"zero_shot": True, "embeddings_few_shot": True, "chrf_advanced_few_shot": True}
    else:
        fields_to_process = []
        if args.zero_shot:
            fields_to_process.append("zero_shot")
        if args.embeddings_few_shot:
            fields_to_process.append("embeddings_few_shot")
        if args.chrf_advanced_few_shot:
            fields_to_process.append("chrf_advanced_few_shot")
        
        include_fields = {
            "zero_shot": args.zero_shot,
            "embeddings_few_shot": args.embeddings_few_shot,
            "chrf_advanced_few_shot": args.chrf_advanced_few_shot
        }
    
    input_file = args.input_file
    
    # Determine output directory. If --outdir is not provided, default to the input file's directory
    output_dir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(input_file))

    # Ensure the output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create intermediate and final output filenames in the chosen directory
    base_name_no_dir = os.path.splitext(os.path.basename(input_file))[0]
    cleaned_file = os.path.join(output_dir, f"{base_name_no_dir}_cleaned.json")
    final_output_file = os.path.join(output_dir, f"restructured_{base_name_no_dir}.json")
    
    print(f"Processing {input_file}...")
    print(f"Fields to process: {', '.join(fields_to_process)}")
    
    # Read the input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
            data = json.loads(file_content)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file.")
        sys.exit(1)
    
    # Step 1: Extract translations (unless --restructure-only is specified)
    if not args.restructure_only:
        print("\nStep 1: Extracting translations...")
        extract_translations(data, fields_to_process)
        
        # Save cleaned data to intermediate file
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            json_str = json_str.replace('\\u2019', "'")  # Right single quotation mark
            f.write(json_str)
        
        print(f"Extraction complete. Cleaned data saved to {cleaned_file}")
        
        if args.extract_only:
            print("Extraction-only mode. Stopping here.")
            return
    else:
        # If restructure-only, use the input file as-is
        cleaned_file = input_file
    
    # Step 2: Restructure data (unless --extract-only is specified)
    if not args.extract_only:
        print("\nStep 2: Restructuring data...")
        
        # Read the cleaned data (or original data if restructure-only)
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            cleaned_data = json.load(f)
        
        restructured_data = restructure_data(cleaned_data, include_fields)
        
        # Save restructured data
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump({"results": restructured_data}, f, ensure_ascii=False, indent=2)
        
        print(f"Restructuring complete. Final results saved to {final_output_file}")
        print(f"Created {len(restructured_data)} restructured entries")
        
        # Clean up intermediate file if we created it
        if not args.restructure_only and os.path.exists(cleaned_file) and cleaned_file != input_file:
            os.remove(cleaned_file)
            print(f"Cleaned up intermediate file: {cleaned_file}")

def extract_translations(data, fields_to_process):
    """Extract the first translation tag content and clean it"""
    
    def extract_first_translation(text):
        match = re.search(r'<translation>(.*?)</translation>', text, re.DOTALL)
        
        if not match:
            match = re.search(r'<translation>(.*?)<translation>', text, re.DOTALL)
        
        if match:
            extracted = match.group(1).strip()
            
            first_line = extracted.split('\n')[0].strip()
            
            if not first_line and len(extracted.split('\n')) > 1:
                first_line = extracted.split('\n')[1].strip()
            
            first_line = re.sub(r'<.*?>', '', first_line)
            
            return first_line
            
        return text
    
    for setting in fields_to_process:
        if setting in data:
            processed_count = 0
            for entry in data[setting]:
                if "generated" in entry:
                    entry["generated"] = extract_first_translation(entry["generated"])
                    processed_count += 1
            print(f"Processed {processed_count} entries in {setting}")
        else:
            print(f"Warning: Field '{setting}' not found in input data")

def restructure_data(data, include_fields):
    """Restructure the data into a unified format"""
    
    def clean_generated(text):
        translation_pattern = r'<translation>(.*?)</translation>'
        matches = re.findall(translation_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        else:
            return text.replace('<translation>', '').replace('</translation>', '').strip()
    
    zero_shot = data.get('zero_shot', []) if include_fields["zero_shot"] else []
    embeddings_few_shot = data.get('embeddings_few_shot', []) if include_fields["embeddings_few_shot"] else []
    chrf_advanced_few_shot = data.get('chrf_advanced_few_shot', []) if include_fields["chrf_advanced_few_shot"] else []
    
    restructured = []
    
    max_length = 0
    if include_fields["zero_shot"] and zero_shot:
        max_length = max(max_length, len(zero_shot))
    if include_fields["embeddings_few_shot"] and embeddings_few_shot:
        max_length = max(max_length, len(embeddings_few_shot))
    if include_fields["chrf_advanced_few_shot"] and chrf_advanced_few_shot:
        max_length = max(max_length, len(chrf_advanced_few_shot))
    
    if max_length == 0 and zero_shot:
        max_length = len(zero_shot)
        include_fields["zero_shot"] = True
    
    for i in range(max_length):
        entry = {}
        
        if include_fields["zero_shot"] and i < len(zero_shot):
            entry["input"] = zero_shot[i].get("input", "")
            entry["expected"] = zero_shot[i].get("expected", "")
        elif include_fields["embeddings_few_shot"] and i < len(embeddings_few_shot):
            entry["input"] = embeddings_few_shot[i].get("input", "")
            entry["expected"] = embeddings_few_shot[i].get("expected", "")
        elif include_fields["chrf_advanced_few_shot"] and i < len(chrf_advanced_few_shot):
            entry["input"] = chrf_advanced_few_shot[i].get("input", "")
            entry["expected"] = chrf_advanced_few_shot[i].get("expected", "")
        
        if include_fields["zero_shot"]:
            entry["zero_shot"] = clean_generated(zero_shot[i]["generated"]) if i < len(zero_shot) else ""
        if include_fields["embeddings_few_shot"]:
            entry["embeddings_few_shot"] = clean_generated(embeddings_few_shot[i]["generated"]) if i < len(embeddings_few_shot) else ""
        if include_fields["chrf_advanced_few_shot"]:
            entry["chrf_advanced_few_shot"] = clean_generated(chrf_advanced_few_shot[i]["generated"]) if i < len(chrf_advanced_few_shot) else ""
        
        restructured.append(entry)
    
    return restructured

if __name__ == "__main__":
    main() 