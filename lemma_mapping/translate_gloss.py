#!/usr/bin/env python3
"""
Script to translate Turkish glosses to English using a dictionary.
"""

import sys
from pathlib import Path

def load_tr_en_dictionary(dict_path):
    """Load Turkish-English dictionary from file."""
    mapping = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            tr_word, en_word = line.split("->", 1)
            tr_word = tr_word.strip()
            en_word = en_word.strip()
            if tr_word:
                mapping[tr_word] = en_word
    return mapping

def turkish_lower(text):
    """
    Properly convert Turkish text to lowercase, handling special characters like 'İ' -> 'i'
    """
    text = text.replace('İ', 'i')
    return text.lower()

def translate_gloss(gloss_path, tr_en_mapping, output_path):
    """
    Reads the gloss file (where a token is like: 'akşam-NOM'),
    checks if the base (the part before the hyphen) has an English
    translation in the mapping, and if so, replaces the base while keeping
    the grammatical tags.
    Outputs the resulting lines to the output file.
    """
    # Create a lowercase version of the mapping for case-insensitive lookups
    lowercase_mapping = {turkish_lower(k): v for k, v in tr_en_mapping.items()}
    
    with open(gloss_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            tokens = line.strip().split()
            new_tokens = []
            for token in tokens:
                # If there's a hyphen, split into base and tags.
                if "-" in token:
                    parts = token.split("-")
                    base = parts[0]
                    rest = "-".join(parts[1:])
                    
                    # Try exact match first
                    if base in tr_en_mapping:
                        new_token = tr_en_mapping[base] + "-" + rest
                    # Try lowercase version
                    elif turkish_lower(base) in lowercase_mapping:
                        # Preserve original case pattern for the translation if possible
                        en_word = lowercase_mapping[turkish_lower(base)]
                        if base.isupper():
                            en_word = en_word.upper()
                        elif base[0].isupper():
                            en_word = en_word.capitalize()
                        new_token = en_word + "-" + rest
                    else:
                        new_token = token
                else:
                    # If the token does not have a hyphen, try to translate it directly.
                    if token in tr_en_mapping:
                        new_token = tr_en_mapping[token]
                    # Try lowercase version
                    elif turkish_lower(token) in lowercase_mapping:
                        # Preserve original case pattern for the translation if possible
                        en_word = lowercase_mapping[turkish_lower(token)]
                        if token.isupper():
                            en_word = en_word.upper()
                        elif token[0].isupper():
                            en_word = en_word.capitalize()
                        new_token = en_word
                    else:
                        new_token = token
                new_tokens.append(new_token)
            fout.write(" ".join(new_tokens) + "\n")
    
def main():
    if len(sys.argv) < 4:
        print("Usage: python translate_gloss.py <dict_file> <gloss_file> <output_file>")
        sys.exit(1)
    
    dict_path = sys.argv[1]
    gloss_path = sys.argv[2]
    output_path = sys.argv[3]
    
    for file_path in [dict_path, gloss_path]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    tr_en_mapping = load_tr_en_dictionary(dict_path)
    print(f"Loaded {len(tr_en_mapping)} dictionary entries")
    
    translate_gloss(gloss_path, tr_en_mapping, output_path)
    print(f"Translation complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()