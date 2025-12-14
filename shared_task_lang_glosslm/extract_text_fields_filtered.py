#!/usr/bin/env python3
"""Extract text fields while filtering out entries with empty glosses or translations."""

import pandas as pd
import os
import glob

output_dir = 'text_fields_filtered'
os.makedirs(output_dir, exist_ok=True)

languages = ['usp']
splits = ['train', 'eval', 'test']
fields = ['transcription', 'glosses', 'translation']

print("Extracting transcription, gloss, and translation fields to separate txt files...")
print("Filtering out entries where either glosses or translation is empty")

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
            total_samples = len(df)
            
            df_filtered = df.dropna(subset=['glosses', 'translation'])
            
            df_filtered = df_filtered[(df_filtered['glosses'] != '') & (df_filtered['translation'] != '')]
            
            filtered_count = total_samples - len(df_filtered)
            print(f"  Processing {split} split ({len(df_filtered)}/{total_samples} samples, {filtered_count} removed)")
            
            for field in fields:
                if field in df_filtered.columns:
                    output_file = os.path.join(lang_dir, f"{split}.{field}.txt")
                    
                    field_values = df_filtered[field].fillna("").astype(str)
                    
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

print("\nFiltering summary:")
for lang in languages:
    train_trans = os.path.join(output_dir, lang, "train.transcription.txt")
    if os.path.exists(train_trans):
        with open(train_trans, 'r', encoding='utf-8') as f:
            filtered_count = sum(1 for _ in f)
        
        orig_train = f"unsegmented_data_strict/{lang}/train.csv"
        if os.path.exists(orig_train):
            orig_count = len(pd.read_csv(orig_train))
            percent_kept = (filtered_count / orig_count) * 100 if orig_count > 0 else 0
            print(f"  {lang.upper()}: Kept {filtered_count}/{orig_count} train samples ({percent_kept:.1f}%)") 