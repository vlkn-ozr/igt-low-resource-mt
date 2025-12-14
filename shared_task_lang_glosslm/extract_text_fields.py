#!/usr/bin/env python3
"""Extract transcription, gloss, and translation fields to separate text files."""

import pandas as pd
import os
import glob

output_dir = 'text_fields'
os.makedirs(output_dir, exist_ok=True)

languages = ['usp']
splits = ['train', 'eval', 'test']
fields = ['transcription', 'glosses', 'translation']

print("Extracting transcription, gloss, and translation fields to separate txt files...")

for lang in languages:
    print(f"\nProcessing {lang}:")
    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)
    
    for split in splits:
        input_file = f"unsegmented_data_strict/{lang}/{split}.csv"
        
        if not os.path.exists(input_file):
            print(f"  Skipping {split} (file not found)")
            continue
            
        try:
            df = pd.read_csv(input_file)
            print(f"  Processing {split} split ({len(df)} samples)")
            
            for field in fields:
                if field in df.columns:
                    output_file = os.path.join(lang_dir, f"{split}.{field}.txt")
                    
                    field_values = df[field].fillna("").astype(str)
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(field_values))
                    
                    print(f"    Saved {len(field_values)} {field} entries to {output_file}")
                else:
                    print(f"    Field '{field}' not found in {split}")
                    
        except Exception as e:
            print(f"  Error processing {input_file}: {str(e)}")

print("\nExtraction completed!")

print("\nSummary of extracted files:")
for lang in languages:
    print(f"\n{lang.upper()}:")
    lang_files = glob.glob(f"{output_dir}/{lang}/*.txt")
    for file in sorted(lang_files):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            file_size = os.path.getsize(file)
            print(f"  {os.path.basename(file):<25} {line_count} lines ({file_size/1024:.1f} KB)")
        except Exception as e:
            print(f"  {os.path.basename(file):<25} Error: {str(e)}") 