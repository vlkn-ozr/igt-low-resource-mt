#!/usr/bin/env python3
"""
Script to convert dictionary file to lowercase, handling Turkish characters.
"""

import re
import sys
from pathlib import Path

def lowercase_dictionary(input_file, output_file):
    """Convert dictionary file to lowercase."""
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    def transform_word(match):
        return match.group(0).lower()
    
    clean_text = re.sub(r"\b[A-Za-z0-9]+\b", transform_word, text)
    clean_text = re.sub(r"\b[\wŞÇÖĞÜİı]+\b", transform_word, clean_text, flags=re.UNICODE)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(clean_text)

def main():
    if len(sys.argv) < 3:
        print("Usage: python lowercase_dict.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    lowercase_dictionary(input_file, output_file)
    print(f"Lowercased dictionary saved to {output_file}")

if __name__ == "__main__":
    main()
