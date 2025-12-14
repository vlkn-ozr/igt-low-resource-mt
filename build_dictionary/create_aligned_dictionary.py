#!/usr/bin/env python3
"""
Script to create a simple Turkish-English dictionary from aligned parallel corpus.
"""

import collections
import sys
from pathlib import Path

def create_dictionary(tr_file, en_file, alignment_file, output_file):
    """Create dictionary from aligned parallel corpus."""
    aligned_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    
    with open(tr_file, "r", encoding="utf-8") as f_tr, \
         open(en_file, "r", encoding="utf-8") as f_en, \
         open(alignment_file, "r", encoding="utf-8") as f_align:
        
        for tr_line, en_line, align_line in zip(f_tr, f_en, f_align):
            tr_tokens = tr_line.strip().split()
            en_tokens = en_line.strip().split()
            alignments = align_line.strip().split()
            
            for alignment in alignments:
                try:
                    idx_tr, idx_en = alignment.split("-")
                    idx_tr = int(idx_tr)
                    idx_en = int(idx_en)
                except ValueError:
                    continue
                
                if idx_tr < len(tr_tokens) and idx_en < len(en_tokens):
                    tr_word = tr_tokens[idx_tr].lower()
                    en_word = en_tokens[idx_en].lower()
                    aligned_counts[tr_word][en_word] += 1
    
    final_dictionary = {}
    for tr_word, en_word_dict in aligned_counts.items():
        most_common_en = max(en_word_dict.items(), key=lambda item: item[1])[0]
        final_dictionary[tr_word] = most_common_en
    
    with open(output_file, "w", encoding="utf-8") as f:
        for tr_word, en_word in sorted(final_dictionary.items()):
            f.write(f"{tr_word} -> {en_word}\n")
    
    print(f"Dictionary created with {len(final_dictionary)} entries")
    print(f"Saved to {output_file}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python create_aligned_dictionary.py <tr_file> <en_file> <alignment_file> <output_file>")
        sys.exit(1)
    
    tr_file = sys.argv[1]
    en_file = sys.argv[2]
    alignment_file = sys.argv[3]
    output_file = sys.argv[4]
    
    for file_path in [tr_file, en_file, alignment_file]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    create_dictionary(tr_file, en_file, alignment_file, output_file)

if __name__ == "__main__":
    main()
