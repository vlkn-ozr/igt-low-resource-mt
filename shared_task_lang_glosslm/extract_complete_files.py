#!/usr/bin/env python3
"""Extract combined text files from unsegmented data with all splits merged."""

import pandas as pd
import os
import glob

output_dir = 'text_fields_complete'
os.makedirs(output_dir, exist_ok=True)

languages = ['usp']
splits = ['train', 'eval', 'test']
fields = ['transcription', 'glosses', 'translation']

print("Extracting combined text files (all splits in one file)...")
print("1. Complete dataset with all entries")
print("2. Filtered dataset without empty glosses/translations")

print("\n--- COMPLETE DATASET ---")
complete_dir = os.path.join(output_dir, 'complete')
os.makedirs(complete_dir, exist_ok=True)

for lang in languages:
    print(f"\nProcessing {lang}:")
    
    all_data = pd.DataFrame()
    
    for split in splits:
        input_file = f"unsegmented_data_strict/{lang}/{split}.csv"
        
        if not os.path.exists(input_file):
            print(f"  Skipping {split} (file not found)")
            continue
            
        try:
            df = pd.read_csv(input_file)
            print(f"  Adding {split} data: {len(df)} samples")
            
            df['split'] = split
            
            all_data = pd.concat([all_data, df], ignore_index=True)
            
        except Exception as e:
            print(f"  Error processing {input_file}: {str(e)}")
    
    for field in fields:
        if field in all_data.columns:
            output_file = os.path.join(complete_dir, f"{lang}.{field}.txt")
            
            field_values = all_data[field].fillna("").astype(str)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(field_values))
            
            print(f"  Saved {len(field_values)} combined {field} entries to {output_file}")

print("\n--- FILTERED DATASET ---")
filtered_dir = os.path.join(output_dir, 'filtered')
os.makedirs(filtered_dir, exist_ok=True)

for lang in languages:
    print(f"\nProcessing {lang}:")
    
    all_data = pd.DataFrame()
    
    for split in splits:
        input_file = f"unsegmented_data_strict/{lang}/{split}.csv"
        
        if not os.path.exists(input_file):
            print(f"  Skipping {split} (file not found)")
            continue
            
        try:
            df = pd.read_csv(input_file)
            
            df_filtered = df.dropna(subset=['glosses', 'translation'])
            df_filtered = df_filtered[(df_filtered['glosses'] != '') & (df_filtered['translation'] != '')]
            
            print(f"  Adding {split} data: {len(df_filtered)}/{len(df)} samples")
            
            df_filtered['split'] = split
            
            all_data = pd.concat([all_data, df_filtered], ignore_index=True)
            
        except Exception as e:
            print(f"  Error processing {input_file}: {str(e)}")
    
    for field in fields:
        if field in all_data.columns:
            output_file = os.path.join(filtered_dir, f"{lang}.{field}.txt")
            
            field_values = all_data[field].fillna("").astype(str)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(field_values))
            
            print(f"  Saved {len(field_values)} filtered {field} entries to {output_file}")

print("\n--- SPECIAL HANDLING FOR NYANGBO ---")
nyb_dir = os.path.join(output_dir, 'nyb_transcription_glosses')
os.makedirs(nyb_dir, exist_ok=True)

nyb_data = pd.DataFrame()

for split in splits:
    input_file = f"unsegmented_data_strict/nyb/{split}.csv"
    
    if not os.path.exists(input_file):
        print(f"  Skipping {split} (file not found)")
        continue
        
    try:
        df = pd.read_csv(input_file)
        
        df_filtered = df.dropna(subset=['glosses'])
        df_filtered = df_filtered[df_filtered['glosses'] != '']
        
        print(f"  Adding {split} data: {len(df_filtered)}/{len(df)} samples")
        
        df_filtered['split'] = split
        
        nyb_data = pd.concat([nyb_data, df_filtered], ignore_index=True)
        
    except Exception as e:
        print(f"  Error processing {input_file}: {str(e)}")

for field in ['transcription', 'glosses']:
    if field in nyb_data.columns:
        output_file = os.path.join(nyb_dir, f"nyb.{field}.txt")
        
        field_values = nyb_data[field].fillna("").astype(str)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(field_values))
        
        print(f"  Saved {len(field_values)} Nyangbo {field} entries to {output_file}")

print("\nExtraction completed!")

print("\nSummary of extracted files:")

print("\nCOMPLETE DATASET:")
complete_files = glob.glob(f"{complete_dir}/*.txt")
for file in sorted(complete_files):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        file_size = os.path.getsize(file)
        print(f"  {os.path.basename(file):<25} {line_count} lines ({file_size/1024:.1f} KB)")
    except Exception as e:
        print(f"  {os.path.basename(file):<25} Error: {str(e)}")

print("\nFILTERED DATASET:")
filtered_files = glob.glob(f"{filtered_dir}/*.txt")
for file in sorted(filtered_files):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        file_size = os.path.getsize(file)
        print(f"  {os.path.basename(file):<25} {line_count} lines ({file_size/1024:.1f} KB)")
    except Exception as e:
        print(f"  {os.path.basename(file):<25} Error: {str(e)}")

print("\nNYANGBO TRANSCRIPTION-GLOSSES PAIRS:")
nyb_files = glob.glob(f"{nyb_dir}/*.txt")
for file in sorted(nyb_files):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        file_size = os.path.getsize(file)
        print(f"  {os.path.basename(file):<25} {line_count} lines ({file_size/1024:.1f} KB)")
    except Exception as e:
        print(f"  {os.path.basename(file):<25} Error: {str(e)}") 