#!/usr/bin/env python3
"""
Script to analyze Turkish words using TRMOR morphological analyzer.
"""

import os
import sys
import json
from subprocess import Popen, PIPE, TimeoutExpired
from pathlib import Path
from typing import List, Dict
from datetime import datetime

def analyze_word(word: str, trmor_path: Path) -> List[str]:
    """Analyze a single word and return its analyses."""
    try:
        p = Popen(['fst-mor', str(trmor_path)], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
        
        stdout, stderr = p.communicate(input=f"{word}\n", timeout=5)
        
        if "reading" in stderr:
            stderr = ""
            
        if stderr and "reading transducer" not in stderr:
            print(f"Warning - Error analyzing '{word}': {stderr}")
            return []
        
        analyses = [
            line for line in stdout.split('\n') 
            if line.strip() and not line.startswith('analyse>')
        ]
        return analyses
    
    except (TimeoutExpired, Exception) as e:
        print(f"Warning - Failed to analyze '{word}': {str(e)}")
        return []

def process_file(input_file: Path, output_file: Path, trmor_path: Path) -> Dict:
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
            word: analyze_word(word, trmor_path)
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
    trmor_path = script_dir / "morph.a"  # Should be in the same directory as the Makefile

    if not trmor_path.exists():
        print(f"Error: Could not find morph.a at {trmor_path}")
        print("Please make sure you have compiled TRMOR by running 'make' in the TRMOR directory")
        sys.exit(-1)

    if len(sys.argv) < 2:
        print("Usage: python analysis.py <input_file> [output_file]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file.with_suffix('.analysis')

    results = process_file(input_file, output_file, trmor_path)
    print(f"\nAnalysis complete. Results written to {output_file.with_suffix('.json')}")

if __name__ == "__main__":
    main()
