#!/usr/bin/env python3
import string
import os
import re

def clean_turkish_glosses(input_file, output_file):
    """
    Clean the Turkish glosses file by:
    1. Removing punctuation marks (except hyphens in morphological tags)
    2. Converting words to lowercase if they had quotes
    3. Preserving the hyphen in morphological tags (e.g., 'word-NOM')
    """
    # Define Turkish-specific punctuation to remove
    punctuation_to_remove = string.punctuation.replace('-', '')  # Keep hyphens for morphological tags
    translation_table = str.maketrans('', '', punctuation_to_remove)
    
    # For handling quotes
    quote_pattern = re.compile(r'[\'"]')
    
    print(f"Cleaning Turkish glosses file: {input_file}")
    print(f"This may take a while for large files...")
    
    total_lines = 0
    processed_lines = 0
    words_with_quotes = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            total_lines += 1
            tokens = line.strip().split()
            cleaned_tokens = []
            
            for token in tokens:
                # Check if token has quotes
                has_quotes = "'" in token or '"' in token
                
                # If token contains a hyphen, it's likely a morphological form (word-TAG)
                if '-' in token:
                    parts = token.split('-', 1)  # Split into word and tag
                    # Clean the word part but leave the tag part unchanged
                    word_part = parts[0]
                    word_has_quotes = "'" in word_part or '"' in word_part
                    
                    # Remove quotes from word part
                    word_part = quote_pattern.sub('', word_part)
                    # Remove other punctuation
                    word_part = word_part.translate(translation_table)
                    
                    # Convert to lowercase if had quotes
                    if word_has_quotes:
                        word_part = word_part.lower()
                        words_with_quotes += 1
                    
                    tag_part = parts[1]
                    cleaned_token = f"{word_part}-{tag_part}" if word_part else ""
                else:
                    # For tokens without hyphen
                    # Remove quotes
                    cleaned_token = quote_pattern.sub('', token)
                    # Remove other punctuation
                    cleaned_token = cleaned_token.translate(translation_table)
                    
                    # Convert to lowercase if had quotes
                    if has_quotes:
                        cleaned_token = cleaned_token.lower()
                        words_with_quotes += 1
                
                if cleaned_token:  # Only add non-empty tokens
                    cleaned_tokens.append(cleaned_token)
            
            # Write the cleaned line
            fout.write(' '.join(cleaned_tokens) + '\n')
            processed_lines += 1
            
            # Print progress every 100,000 lines
            if processed_lines % 100000 == 0:
                print(f"Processed {processed_lines:,} lines...")
    
    print(f"Finished processing {total_lines:,} lines.")
    print(f"Words with quotes that were converted to lowercase: {words_with_quotes}")
    print(f"Cleaned glosses file saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean punctuation from Turkish glosses file")
    parser.add_argument("--input-file", default="glosslm_disambiguated_glosses_60k_final.txt",
                       help="Path to the Turkish glosses file")
    parser.add_argument("--output-file", default="glosslm_disambiguated_glosses_60k_cleaned.txt",
                       help="Path to save the cleaned glosses file")
    
    args = parser.parse_args()
    
    # Clean the glosses file
    clean_turkish_glosses(args.input_file, args.output_file) 