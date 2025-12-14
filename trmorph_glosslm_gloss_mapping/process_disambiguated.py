#!/usr/bin/env python3
"""
Script to process disambiguated morphological analysis results and extract glosses.
"""

import sys
from pathlib import Path

def get_shortest_analysis(analyses):
    """Return the shortest analysis from a list of analyses with equal probabilities."""
    return min(analyses, key=lambda x: len(x))

def get_morphological_analysis(full_analysis):
    """Extract the morphological analysis in the format word<tags>."""
    # Split into word and analysis parts
    parts = full_analysis.split(None, 1)
    if len(parts) > 1:
        word_part = parts[0]
        analysis_part = parts[1]
        # If analysis contains ???, return the original word
        if "???" in analysis_part:
            return word_part
        # The analysis part already contains the lemma and tags
        return analysis_part
    return parts[0]

def process_file(input_file, output_file, sentence_file):
    # First, read all sentences to know sentence boundaries
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [sent.strip() for sent in f.readlines()]
    
    # Process disambiguated results
    current_word = None
    current_analyses = []
    max_prob = float('-inf')
    word_analyses = {}  # Store all word analyses
    unanalyzed_words = set()  # Store unique unanalyzed words
    
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_word and current_analyses:
                    best_analysis = get_shortest_analysis(current_analyses) if len(current_analyses) > 1 else current_analyses[0]
                    word_analyses[current_word] = best_analysis
                    if "???" in best_analysis:
                        unanalyzed_words.add(current_word)
                current_word = None
                current_analyses = []
                max_prob = float('-inf')
                continue
            
            # Parse line with probability and analysis
            try:
                prob_str, analysis = line.split(': ', 1)
                prob = float(prob_str)
                
                # Extract word and analysis
                word = analysis.split()[0]
                
                if word != current_word:
                    # Store previous word's best analysis if exists
                    if current_word and current_analyses:
                        best_analysis = get_shortest_analysis(current_analyses) if len(current_analyses) > 1 else current_analyses[0]
                        word_analyses[current_word] = best_analysis
                        if "???" in best_analysis:
                            unanalyzed_words.add(current_word)
                    
                    # Reset for new word
                    current_word = word
                    current_analyses = []
                    max_prob = prob
                
                if prob > max_prob:
                    current_analyses = [analysis]
                    max_prob = prob
                elif prob == max_prob:
                    current_analyses.append(analysis)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed line: {line}")
                continue
    
    # Don't forget to add the last word
    if current_word and current_analyses:
        best_analysis = get_shortest_analysis(current_analyses) if len(current_analyses) > 1 else current_analyses[0]
        word_analyses[current_word] = best_analysis
        if "???" in best_analysis:
            unanalyzed_words.add(current_word)
    
    # Write sentence analyses
    with open(output_file, 'w', encoding='utf-8') as fout:
        for sentence in sentences:
            words = sentence.split()
            sentence_analyses = []
            for word in words:
                # Skip punctuation marks
                if word in ".,!?;:-—()\"'":
                    continue
                if word in word_analyses:
                    analysis = get_morphological_analysis(word_analyses[word])
                    sentence_analyses.append(analysis)
            if sentence_analyses:  # Only write if there are analyses
                fout.write(' '.join(sentence_analyses) + '\n')
    
    # Write unanalyzed words to a separate file
    with open('unanalyzed_words.txt', 'w', encoding='utf-8') as f:
        for word in sorted(unanalyzed_words):
            f.write(word + '\n')

def main():
    if len(sys.argv) < 4:
        print("Usage: python process_disambiguated.py <input_file> <output_file> <sentence_file>")
        print("  input_file: Disambiguated analysis results file")
        print("  output_file: Output file for processed glosses")
        print("  sentence_file: Original sentence file (one sentence per line)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    sentence_file = sys.argv[3]
    
    for file_path in [input_file, sentence_file]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    process_file(input_file, output_file, sentence_file)
    print(f"Processing complete. Output written to {output_file}")

if __name__ == "__main__":
    main() 