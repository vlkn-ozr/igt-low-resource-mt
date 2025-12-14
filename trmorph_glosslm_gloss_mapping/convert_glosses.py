#!/usr/bin/env python3
"""
Script to convert morphological analyses to GlossLM format using a mapping file.
"""

import sys
from pathlib import Path

def load_mapping(mapping_file='gloss_mapping.txt'):
    """Load the mapping from the mapping file."""
    mapping = {}
    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        in_core_mappings = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if "Core Mappings" in line:
                    in_core_mappings = True
                elif "Special Cases" in line:
                    in_core_mappings = False
                continue

            if in_core_mappings and '->' in line:
                morph_tag, gloss_tag = line.split('->')
                mapping[morph_tag.strip()] = gloss_tag.strip()
    
    return mapping

def convert_analysis(analysis, mapping):
    """Convert a single morphological analysis to GlossLM format."""
    if '???' in analysis:
        # Keep original word form for unanalyzed words
        return analysis.split()[0]
    
    # If there is no explicit '<', return the analysis as is.
    if '<' not in analysis:
        return analysis

    # Try to split on whitespace; if not, split using the first '<'
    parts = analysis.split(None, 1)
    if len(parts) == 2:
        word = parts[0]
        tag_string = parts[1]
    else:
        idx = analysis.find('<')
        word = analysis[:idx]
        tag_string = analysis[idx:]
    
    tag_string = tag_string.strip()
    # Remove the leading '<' if present
    if tag_string.startswith('<'):
        tag_string = tag_string[1:]
    
    # Split the remaining tag string into parts (each originally enclosed in < >)
    tag_parts = tag_string.split('<')
    
    # Process the first tag, which may include POS and its subtypes (e.g., "Prn:pers:1s>")
    pos_tag = tag_parts[0].rstrip('>')
    pos_parts = pos_tag.split(':')
    base_pos = pos_parts[0]
    converted_tags = []
    
    if base_pos in mapping:
        converted_tags.append(mapping[base_pos])
    else:
        converted_tags.append(base_pos.upper())
    
    # Process POS subtypes if any
    if len(pos_parts) > 1:
        for subtype in pos_parts[1:]:
            if subtype in mapping:
                converted_tags.append(mapping[subtype])
            else:
                converted_tags.append(subtype.upper())
    
    # Process the remaining tags from the analysis
    for part in tag_parts[1:]:
        tag = part.split('>')[0]
        if tag in mapping:
            converted_tags.append(mapping[tag])
        else:
            converted_tags.append(tag.upper())
    
    # Join the word and the converted gloss labels with dots.
    return f"{word}-{'-'.join(converted_tags)}"

def convert_file(input_file, output_file, mapping):
    """Convert all analyses in the input file to GlossLM format."""
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            analyses = line.strip().split()
            converted = [convert_analysis(analysis, mapping) for analysis in analyses]
            fout.write(' '.join(converted) + '\n')

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_glosses.py <input_file> <output_file> [mapping_file]")
        print("  mapping_file defaults to 'gloss_mapping.txt'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mapping_file = sys.argv[3] if len(sys.argv) > 3 else 'gloss_mapping.txt'
    
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    mapping = load_mapping(mapping_file)
    convert_file(input_file, output_file, mapping)
    
    print(f"Conversion complete. Output written to {output_file}")

if __name__ == "__main__":
    main() 