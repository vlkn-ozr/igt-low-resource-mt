#!/usr/bin/env python3
"""
Script to analyze dictionary files for duplicate entries.
"""

import sys
from collections import Counter
from pathlib import Path

def load_tr_words(dict_path):
    """Load all Turkish words from the dictionary file."""
    tr_words = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '->' in line:
                tr_word = line.split('->', 1)[0].strip()
                if tr_word:
                    tr_words.append(tr_word)
    return tr_words

def analyze_dictionary(dict_path):
    """Analyze dictionary file for duplicates and other statistics."""
    tr_words = load_tr_words(dict_path)
    counter = Counter(tr_words)
    duplicates = {word: count for word, count in counter.items() if count > 1}
    
    print(f'Dictionary file: {dict_path}')
    print(f'Total lines with entries: {sum(counter.values())}')
    print(f'Unique Turkish words: {len(counter)}')
    print(f'Duplicate Turkish words: {len(duplicates)}')
    print(f'Difference (duplicates count): {sum(counter.values()) - len(counter)}')
    
    if duplicates:
        print('\nTop 20 duplicates:')
        for word, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f'{word}: {count} times')

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_duplicates.py <dictionary_file>")
        sys.exit(1)
    
    dict_path = sys.argv[1]
    
    if not Path(dict_path).exists():
        print(f"Error: Dictionary file not found: {dict_path}")
        sys.exit(1)
    
    analyze_dictionary(dict_path)

if __name__ == "__main__":
    main() 