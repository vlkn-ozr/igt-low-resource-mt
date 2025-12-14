#!/usr/bin/env python3
"""
Script to remove duplicate unanalyzed words from statistics and recalculate coverage.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Set


class DuplicateAnalysisStats:
    def __init__(self, stats_file: Path):
        """Initialize with a stats file containing analysis statistics."""
        self.stats_file = stats_file
        with open(stats_file, "r", encoding="utf-8") as f:
            self.stats_data = json.load(f)

    def remove_duplicates_and_recalculate(self) -> Dict:
        """Remove duplicate unanalyzed words and recalculate coverage statistics."""
        # Get original stats
        coverage_stats = self.stats_data["coverage_stats"]
        original_total = coverage_stats["total_words"]
        original_analyzed = coverage_stats["analyzed_words"]  # Use original analyzed count
        unanalyzed_words = self.stats_data["unanalyzed_words"]

        # Count duplicates
        word_counts = Counter(unanalyzed_words)
        duplicate_count = sum(count - 1 for count in word_counts.values())
        
        # Create set of unique unanalyzed words
        unique_unanalyzed = list(set(unanalyzed_words))
        
        # Recalculate statistics
        adjusted_total = original_total - duplicate_count
        analyzed_words = original_analyzed  # Keep original analyzed words count
        coverage_percentage = (analyzed_words / adjusted_total * 100) if adjusted_total > 0 else 0

        # Create updated report
        updated_report = {
            "timestamp": str(datetime.now()),
            "input_file": self.stats_data["input_file"],
            "total_sentences": self.stats_data["total_sentences"],
            "coverage_stats": {
                "total_words": adjusted_total,
                "analyzed_words": analyzed_words,
                "coverage_percentage": round(coverage_percentage, 2)
            },
            "analyses_stats": self.stats_data["analyses_stats"],
            "duplicate_stats": {
                "original_unanalyzed_count": len(unanalyzed_words),
                "unique_unanalyzed_count": len(unique_unanalyzed),
                "duplicates_removed": duplicate_count
            },
            "unanalyzed_words": unique_unanalyzed
        }


        return updated_report

    def save_report(self, output_file: Path = None):
        """Generate and save the updated statistical report."""
        if output_file is None:
            # Create new filename by adding '_unique' before the extension
            output_file = self.stats_file.with_name(
                self.stats_file.stem + '_unique' + self.stats_file.suffix
            )

        report = self.remove_duplicates_and_recalculate()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Print summary to console
        print("\nUpdated Analysis Statistics:")
        print(f"Adjusted Coverage: {report['coverage_stats']['coverage_percentage']}%")
        print(f"Adjusted Total Words: {report['coverage_stats']['total_words']}")
        print(f"Unique Unanalyzed Words: {len(report['unanalyzed_words'])}")
        print(f"Duplicates Removed: {report['duplicate_stats']['duplicates_removed']}")
        print(f"\nFull report saved to: {output_file}")


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stats_duplicate.py <stats_file.json> [output_file.json]")
        sys.exit(1)

    stats_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    stats = DuplicateAnalysisStats(stats_file)
    stats.save_report(output_file)


if __name__ == "__main__":
    main()
