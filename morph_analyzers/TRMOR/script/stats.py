#!/usr/bin/env python3
"""
Script to calculate statistics from TRMOR analysis JSON files.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class AnalysisStats:
    def __init__(self, analysis_file: Path):
        self.analysis_file = analysis_file
        with open(analysis_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def calculate_coverage_stats(self, words, analyses):
        unanalyzed_words = []
        total_words = len(words)
        
        for word, analysis in zip(words, analyses):
            # Check for empty, None, or any form of "no result" message
            if (not analysis or 
                analysis == "no result" or 
                (isinstance(analysis, list) and (
                    "no result" in str(analysis) or 
                    any("no result for" in str(item) for item in analysis)
                ))):
                unanalyzed_words.append(word)
        
        analyzed_words = total_words - len(unanalyzed_words)
        coverage_percentage = (analyzed_words / total_words * 100.0) if total_words > 0 else 0.0
        
        return {
            "total_words": total_words,
            "analyzed_words": analyzed_words,
            "coverage_percentage": coverage_percentage,
            "unanalyzed_words": unanalyzed_words
        }
    
    def calculate_coverage(self) -> Dict:
        """Calculate what percentage of words were successfully analyzed."""
        total_words = 0
        analyzed_words = 0
        
        for sentence in self.data['sentences']:
            for word, analyses in sentence['analyses'].items():
                total_words += 1
                # Consider a word analyzed only if it has valid analyses
                if analyses and isinstance(analyses, list):
                    # Check each analysis in the list
                    has_valid_analysis = True
                    for analysis in analyses:
                        if (not analysis or 
                            analysis == "" or
                            "no result" in str(analysis) or 
                            "no result for" in str(analysis)):
                            has_valid_analysis = False
                            break
                    if has_valid_analysis:
                        analyzed_words += 1
        
        coverage = (analyzed_words / total_words * 100) if total_words > 0 else 0
        return {
            'total_words': total_words,
            'analyzed_words': analyzed_words,
            'coverage_percentage': round(coverage, 2)
        }
    
    def calculate_analyses_stats(self) -> Dict:
        """Calculate statistics about analyses per word."""
        analyses_counts = []
        
        for sentence in self.data['sentences']:
            for analyses in sentence['analyses'].values():
                if analyses:  # only count words that were analyzed
                    analyses_counts.append(len(analyses))
        
        if analyses_counts:
            return {
                'average_analyses_per_word': round(sum(analyses_counts) / len(analyses_counts), 2),
                'max_analyses': max(analyses_counts),
                'min_analyses': min(analyses_counts),
                'total_analyses': sum(analyses_counts)
            }
        return {
            'average_analyses_per_word': 0,
            'max_analyses': 0,
            'min_analyses': 0,
            'total_analyses': 0
        }
    
    def get_unanalyzed_words(self) -> List[str]:
        """Get list of words that couldn't be analyzed."""
        unanalyzed = set()
        for sentence in self.data['sentences']:
            for word, analyses in sentence['analyses'].items():
                if not analyses or not isinstance(analyses, list):
                    unanalyzed.add(word)
                else:
                    # Check each analysis in the list
                    has_valid_analysis = True
                    for analysis in analyses:
                        if (not analysis or 
                            analysis == "" or
                            "no result" in str(analysis) or 
                            "no result for" in str(analysis)):
                            has_valid_analysis = False
                            break
                    if not has_valid_analysis:
                        unanalyzed.add(word)
        return sorted(list(unanalyzed))
    
    def generate_report(self) -> Dict:
        """Generate a complete statistical report."""
        coverage_stats = self.calculate_coverage()
        analyses_stats = self.calculate_analyses_stats()
        unanalyzed = self.get_unanalyzed_words()
        
        report = {
            'timestamp': str(datetime.now()),
            'input_file': self.data['metadata']['input_file'],
            'total_sentences': self.data['metadata']['total_sentences'],
            'coverage_stats': coverage_stats,
            'analyses_stats': analyses_stats,
            'unanalyzed_words': unanalyzed
        }
        
        return report
    
    def save_report(self, output_file: Path = None):
        """Generate and save statistical report."""
        if output_file is None:
            output_file = self.analysis_file.with_suffix('.stats.json')
        
        report = self.generate_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Also print summary to console
        print("\nAnalysis Statistics:")
        print(f"Coverage: {report['coverage_stats']['coverage_percentage']}%")
        print(f"Average analyses per word: {report['analyses_stats']['average_analyses_per_word']}")
        print(f"Total words: {report['coverage_stats']['total_words']}")
        print(f"Unanalyzed words: {len(report['unanalyzed_words'])}")
        print(f"\nFull report saved to: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python stats.py <analysis_file.json> [output_file.json]")
        sys.exit(1)
    
    analysis_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    stats = AnalysisStats(analysis_file)
    stats.save_report(output_file)

if __name__ == "__main__":
    main()
