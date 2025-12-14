#!/usr/bin/env python3
"""
Script to remove angle bracket tags (<...>) from text files.
"""

import re
import sys
from pathlib import Path

def remove_angle_tags(input_filepath, output_filepath):
    # Compile a regex pattern that matches anything in between < and >
    pattern = re.compile(r"<[^>]*>")
    
    with open(input_filepath, "r", encoding="utf-8") as infile, \
         open(output_filepath, "w", encoding="utf-8") as outfile:
        for line in infile:
            # Remove all <...> tags from the line
            clean_line = pattern.sub("", line)
            outfile.write(clean_line)

def main():
    if len(sys.argv) < 3:
        print("Usage: python remove_tags.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} not found")
        sys.exit(1)
    
    remove_angle_tags(input_path, output_path)
    print(f"Tags removed. Output written to {output_path}")

if __name__ == "__main__":
    main()