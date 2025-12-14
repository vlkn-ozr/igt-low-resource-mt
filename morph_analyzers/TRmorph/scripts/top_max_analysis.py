#!/usr/bin/python3

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class TopAnalysisInspector:
    def __init__(self, analysis_file: Path):
        """Initialize with the analysis JSON file."""
        self.analysis_file = analysis_file
        with open(analysis_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get_word_analysis_counts(self) -> Dict[str, List[str]]:
        """
        Get a dictionary of words and their analyses, counting unique occurrences.
        Returns a dict with word as key and list of analyses as value.
        """
        word_analyses = defaultdict(set)
        
        # Collect all analyses for each word
        for sentence in self.data["sentences"]:
            for word, analyses in sentence["analyses"].items():
                if analyses and isinstance(analyses, list):
                    word_analyses[word].update(analyses)
        
        # Convert sets back to lists
        return {word: list(analyses) for word, analyses in word_analyses.items()}

    def get_top_k_analyses(self, k: int = 10) -> List[Tuple[str, List[str], int]]:
        """
        Get the top k words with the most analyses.
        Returns list of tuples (word, analyses, count).
        """
        word_analyses = self.get_word_analysis_counts()
        
        # Sort words by number of analyses in descending order
        sorted_words = sorted(
            [(word, analyses, len(analyses)) 
             for word, analyses in word_analyses.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        return sorted_words[:k]

    def save_top_analyses(self, k: int = 10, output_file: Path = None):
        """Save the top k analyses to a file."""
        if output_file is None:
            output_file = self.analysis_file.parent / f"top_{k}_analyses.txt"

        top_analyses = self.get_top_k_analyses(k)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Top {k} Words with Most Analyses\n")
            f.write("=" * 50 + "\n\n")
            
            for word, analyses, count in top_analyses:
                f.write(f"Word: {word}\n")
                f.write(f"Number of analyses: {count}\n")
                f.write("-" * 30 + "\n")
                for i, analysis in enumerate(analyses, 1):
                    f.write(f"{i}. {analysis}\n")
                f.write("\n" + "=" * 50 + "\n\n")

        print(f"\nAnalysis Summary:")
        print("-" * 30)
        for word, _, count in top_analyses:
            print(f"{word}: {count} analyses")
        print(f"\nDetailed analysis saved to: {output_file}")

    def print_top_k_summary(self, k: int = 10, output_file: Path = None):
        """
        Print and optionally save just the word names and their analysis counts.
        
        Args:
            k: Number of top words to show
            output_file: Optional file path to save the summary
        """
        top_analyses = self.get_top_k_analyses(k)
        
        # Create summary text
        summary_lines = [f"Top {k} Words with Most Analyses"]
        summary_lines.append("=" * 30)
        
        # Add each word and its analysis count
        for i, (word, _, count) in enumerate(top_analyses, 1):
            summary_lines.append(f"{i}. {word}: {count} analyses")
        
        # Print to console
        print("\n".join(summary_lines))
        
        # Save to file if specified
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))
            print(f"\nSummary saved to: {output_file}")


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python top_max_analysis.py <analysis_file.json> [k] [output_file] [--summary]")
        sys.exit(1)

    analysis_file = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    summary_only = "--summary" in sys.argv

    inspector = TopAnalysisInspector(analysis_file)
    
    if summary_only:
        summary_file = output_file.with_name(f"top_{k}_summary.txt") if output_file else None
        inspector.print_top_k_summary(k, summary_file)
    else:
        inspector.save_top_analyses(k, output_file)


if __name__ == "__main__":
    main()
