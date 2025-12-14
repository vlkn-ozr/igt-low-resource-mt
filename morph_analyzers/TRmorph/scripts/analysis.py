#!/usr/bin/env python3
"""
Script to analyze Turkish words using TRmorph morphological analyzer.
"""

import os
import sys
import json
from subprocess import Popen, PIPE, TimeoutExpired
from pathlib import Path
from typing import List, Dict
from datetime import datetime

def analyze_word(word: str, trmorph_path: Path) -> List[str]:
    """Analyze a single word and return its analyses."""
    # Escape special characters in the word
    escaped_word = word.replace("'", "'\\''")
    
    cmd = f"echo '{escaped_word}' | /usr/local/bin/flookup -b -x {trmorph_path}"
    
    try:
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate(timeout=5)
        
        if stderr:
            print(f"Warning - Error analyzing '{word}': {stderr.decode('utf-8')}")
            return []
        
        analyses = [
            line for line in stdout.decode("utf-8").strip().split("\n")
            if line.strip()
        ]
        return analyses
    
    except (TimeoutExpired, Exception) as e:
        print(f"Warning - Failed to analyze '{word}': {str(e)}")
        return []

def process_file(input_file: Path, output_file: Path, trmorph_path: Path) -> Dict:
    """Process all sentences and return analysis results."""
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(-1)
    
    # Read all sentences from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    # Process each sentence and store results
    results = {
        'metadata': {
            'input_file': str(input_file),
            'timestamp': str(datetime.now()),
            'total_sentences': len(sentences)
        },
        'sentences': []
    }
    
    for i, sentence in enumerate(sentences, 1):
        print(f"Processing sentence {i}/{len(sentences)}")
        
        # Split sentence into words and analyze each
        words = sentence.split()
        sentence_analyses = {
            word: analyze_word(word, trmorph_path)
            for word in words
        }
        
        results['sentences'].append({
            'sentence_id': i,
            'text': sentence,
            'analyses': sentence_analyses
        })
    
    # Save results as JSON
    with open(output_file.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def main():
    # Get the directory containing the script
    script_dir = Path(__file__).parent.parent
    trmorph_path = script_dir / "trmorph.fst"

    if not trmorph_path.exists():
        print(f"Error: Could not find trmorph.fst at {trmorph_path}")
        sys.exit(-1)

    if len(sys.argv) < 2:
        print("Usage: python example.py <input_file> [output_file]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file.with_suffix('.analysis')

    results = process_file(input_file, output_file, trmorph_path)
    print(f"\nAnalysis complete. Results written to {output_file.with_suffix('.json')}")

if __name__ == "__main__":
    main()
