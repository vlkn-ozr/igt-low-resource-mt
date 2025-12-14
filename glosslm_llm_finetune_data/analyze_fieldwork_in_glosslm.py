#!/usr/bin/env python3
"""
Script to analyze fieldwork languages in GlossLM train split using cached parquet file.
"""

import pandas as pd
import sys
from datetime import datetime
from collections import defaultdict

class TeeWriter:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"fieldwork_glosslm_analysis_{timestamp}.txt"

    with open(output_file, 'w') as f:
        sys.stdout = TeeWriter(f)
        
        print("Loading GlossLM train split from cached parquet file...")
        
        try:
            glosslm_train_df = pd.read_parquet('raw_data/glosslm_train.parquet')
            print("GlossLM train split loaded successfully from cached file")
            print(f"Total samples in GlossLM train: {len(glosslm_train_df)}")
            print(f"GlossLM columns: {list(glosslm_train_df.columns)}")
            print(f"First GlossLM example glottocode: {glosslm_train_df.iloc[0]['glottocode']}")
        except Exception as e:
            print(f"Error loading GlossLM train dataset: {e}")
            sys.stdout = sys.__stdout__
            return

        glottocode_to_name = {}
        try:
            with open("fieldwork_iso.txt", "r") as iso_f:
                lines = iso_f.readlines()
                for line in lines[2:]:
                    parts = [part.strip() for part in line.strip().split("|")]
                    if len(parts) >= 3:
                        glottocode = parts[1].strip()
                        language_name = parts[2].strip()
                        glottocode_to_name[glottocode] = language_name
            print(f"Loaded {len(glottocode_to_name)} language mappings from fieldwork_iso.txt")
        except Exception as e:
            print(f"Warning: Could not load fieldwork_iso.txt: {e}")
            print("Will use glottocodes directly")

        fieldwork_glottocodes = set(glottocode_to_name.keys())
        print(f"Total unique fieldwork glottocodes from ISO file: {len(fieldwork_glottocodes)}")

        print("\nAnalyzing GlossLM train split for fieldwork glottocodes...")
        
        glosslm_glottocode_counts = defaultdict(int)
        fieldwork_in_glosslm = defaultdict(int)
        
        glosslm_train_df = glosslm_train_df.dropna(subset=['glottocode'])
        
        for _, row in glosslm_train_df.iterrows():
            glottocode = row['glottocode']
            glosslm_glottocode_counts[glottocode] += 1
            
            if glottocode in fieldwork_glottocodes:
                fieldwork_in_glosslm[glottocode] += 1

        print(f"\nFieldwork languages found in GlossLM train split:")
        print("=" * 80)
        
        total_fieldwork_samples = 0
        glottocodes_found = 0
        
        for glottocode, count in sorted(
            fieldwork_in_glosslm.items(), 
            key=lambda x: (-x[1], x[0])
        ):
            language_name = glottocode_to_name.get(glottocode, glottocode)
            print(f"{language_name} ({glottocode}): {count} samples")
            total_fieldwork_samples += count
            glottocodes_found += 1

        print(f"\nSummary:")
        print(f"Total unique glottocodes in GlossLM train: {len(glosslm_glottocode_counts)}")
        print(f"Total fieldwork glottocodes: {len(fieldwork_glottocodes)}")
        print(f"Fieldwork glottocodes found in GlossLM train: {glottocodes_found}")
        print(f"Total fieldwork samples in GlossLM train: {total_fieldwork_samples}")
        print(f"Percentage of fieldwork glottocodes in GlossLM train: {glottocodes_found/len(fieldwork_glottocodes)*100:.2f}%")
        print(f"Percentage of GlossLM train samples that are fieldwork: {total_fieldwork_samples/len(glosslm_train_df)*100:.2f}%")

        missing_glottocodes = fieldwork_glottocodes - set(fieldwork_in_glosslm.keys())
        if missing_glottocodes:
            print(f"\nFieldwork glottocodes NOT found in GlossLM train ({len(missing_glottocodes)}):")
            for glottocode in sorted(missing_glottocodes):
                language_name = glottocode_to_name.get(glottocode, glottocode)
                print(f"  {language_name} ({glottocode})")

        print(f"\nTop 20 glottocodes in GlossLM train split:")
        print("=" * 50)
        top_glottocodes = sorted(glosslm_glottocode_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for glottocode, count in top_glottocodes:
            language_name = glottocode_to_name.get(glottocode, glottocode)
            is_fieldwork = " (FIELDWORK)" if glottocode in fieldwork_glottocodes else ""
            print(f"{language_name} ({glottocode}): {count} samples{is_fieldwork}")

        print(f"\nSample GlossLM data (first 5 examples):")
        print("=" * 50)
        for i in range(min(5, len(glosslm_train_df))):
            row = glosslm_train_df.iloc[i]
            print(f"Example {i+1}:")
            print(f"  Language: {row.get('language', 'N/A')}")
            print(f"  Glottocode: {row.get('glottocode', 'N/A')}")
            print(f"  Translation: {row.get('translation', 'N/A')[:100]}...")
            print()

    sys.stdout = sys.__stdout__
    print(f"\nAnalysis results have been saved to: {output_file}")

if __name__ == "__main__":
    main() 