#!/usr/bin/env python3
"""
Script to extract transcription, gloss, and translation columns from the wav2gloss dataset
and organize them by language with split-specific files.

Directory layout produced:
seen|unseen/<language_code>/{split}.{type}.txt
"""

import os
from collections import defaultdict
from datasets import load_from_disk, load_dataset

SPLITS = ["train", "validation", "test"]


def extract_by_language():
    """Extract and organize the dataset by language and split."""
    print("Extracting wav2gloss dataset by language and split ...")

    dataset_path = "./wav2gloss_fieldwork_text_only"
    dataset = {}

    try:
        if os.path.exists(dataset_path):
            print("Loading dataset from local disk ...")
            for split in SPLITS:
                split_path = f"{dataset_path}/{split}"
                if os.path.exists(split_path):
                    dataset[split] = load_from_disk(split_path)
                    print(f"  {split}: {len(dataset[split])} examples")
        else:
            print("Local dataset not found. Downloading filtered dataset from HF ...")
            full_dataset = load_dataset("wav2gloss/fieldwork")
            columns_to_keep = ["transcription", "gloss", "translation", "language"]
            for split_name, split_data in full_dataset.items():
                cols_to_drop = [c for c in split_data.column_names if c not in columns_to_keep]
                dataset[split_name] = split_data.remove_columns(cols_to_drop)
                print(f"  {split_name}: {len(dataset[split_name])} examples (downloaded)")
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        return

    lang_split_dict = defaultdict(lambda: defaultdict(list))
    for split_name, split_data in dataset.items():
        for ex in split_data:
            lang = ex.get("language", "unknown")
            lang_split_dict[lang][split_name].append(ex)

    print(f"\nFound {len(lang_split_dict)} languages. Writing files ...")

    for lang, split_examples in lang_split_dict.items():
        category = "seen" if len(split_examples.get("train", [])) > 0 else "unseen"
        lang_dir = os.path.join(category, lang)
        os.makedirs(lang_dir, exist_ok=True)
        print(f"  {category}/{lang}:")

        for split_name in SPLITS:
            examples = split_examples.get(split_name, [])
            if not examples:
                continue

            trans_lines = [ex.get("transcription", "").strip() for ex in examples]
            gloss_lines = [ex.get("gloss", "").strip() for ex in examples]
            transl_lines = [ex.get("translation", "").strip() for ex in examples]

            out_prefix = os.path.join(lang_dir, split_name)

            with open(f"{out_prefix}.transcriptions.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(trans_lines))
            with open(f"{out_prefix}.glosses.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(gloss_lines))
            with open(f"{out_prefix}.translations.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(transl_lines))

            print(f"    • {split_name}: {len(examples)} lines written")

    print("\nExtraction done. Directory layout:")
    print("  seen|unseen/<language>/{split}.{type}.txt")


if __name__ == "__main__":
    extract_by_language() 