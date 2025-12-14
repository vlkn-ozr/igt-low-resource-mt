#!/usr/bin/env python3
"""
Script to convert text to lowercase, handling Turkish characters.
"""

import re
import sys
from pathlib import Path

def lowercase_text(input_file, output_file):
    """Convert text to lowercase, preserving structure."""
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    if not text:
        print("Warning: Input file is empty")
        return
    
    text = text[0].lower() + text[1:]
    
    def lowercase_match(match):
        return match.group(1) + match.group(2).lower()
    
    clean_text = re.sub(r"(\s|\.\s)([A-Z])", lowercase_match, text)
    clean_text = re.sub(r"(\s)([A-Za-zŞÇÖĞÜİı]+)", lambda m: m.group(1) + m.group(2).lower(), clean_text)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(clean_text)

def main():
    if len(sys.argv) < 3:
        print("Usage: python lowercase.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    lowercase_text(input_file, output_file)
    print(f"Lowercased text saved to {output_file}")

if __name__ == "__main__":
    main()