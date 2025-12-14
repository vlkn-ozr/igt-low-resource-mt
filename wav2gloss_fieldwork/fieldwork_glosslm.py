#!/usr/bin/env python3
"""
Script to analyze overlap between wav2gloss/fieldwork and GlossLM datasets.
"""

from datasets import load_dataset

def analyze_overlap():
    """Analyze language overlap between fieldwork and GlossLM datasets."""
    try:
        field = load_dataset("wav2gloss/fieldwork", split="test")
        print("Field dataset columns:", field.column_names)
        print("First field example:", field[0])
    except Exception as e:
        print(f"Error loading fieldwork dataset: {e}")
        return

    try:
        glosslm = load_dataset("lecslab/glosslm-corpus-split", split="train")
        print("GlossLM dataset columns:", glosslm.column_names)
        print("First glosslm example:", glosslm[0])
    except Exception as e:
        print(f"Error loading glosslm dataset: {e}")
        return

    field_glottocodes = set(field["language"])
    print("Field glottocodes:", field_glottocodes)

    glosslm_glottocodes = set(glosslm["glottocode"])
    print("GlossLM glottocodes count:", len(glosslm_glottocodes))

    glottocode_to_name = {}
    try:
        with open("fieldwork_iso.txt", "r") as f:
            lines = f.readlines()
            for line in lines[2:]:
                parts = [part.strip() for part in line.strip().split("|")]
                if len(parts) >= 3:
                    glottocode = parts[1].strip()
                    language_name = parts[2].strip()
                    glottocode_to_name[glottocode] = language_name
    except FileNotFoundError:
        print("Warning: fieldwork_iso.txt not found. Language names will be 'Unknown'.")

    overlap = field_glottocodes & glosslm_glottocodes

    overlap_counts = {}
    for example in field:
        if example["language"] in overlap:
            overlap_counts[example["language"]] = overlap_counts.get(example["language"], 0) + 1

    print("\nFieldwork dataset sizes for languages that overlap with GlossLM:")
    total_examples = 0
    for glottocode, count in sorted(overlap_counts.items(), key=lambda x: (-x[1], x[0])):
        language_name = glottocode_to_name.get(glottocode, "Unknown")
        print(f"{language_name} ({glottocode}): {count} examples")
        total_examples += count

    print(f"\nTotal overlapping languages: {len(overlap)}")
    print(f"Total examples in overlapping languages: {total_examples}")

if __name__ == "__main__":
    analyze_overlap()
