#!/usr/bin/env python3
"""
Sample aligned sentences (transcription, gloss, translation) for each language and split.

Directory layout expected (created by extract_by_language.py):
seen|unseen/<lang>/{split}.{type}.txt

For every split found, this script creates:
    {split}.sample.{type}.txt

If there are fewer than sample_size examples in a split, it samples all.
"""

import os
import random
import argparse
from typing import List
from pathlib import Path
import shutil


def read_lines(path: str) -> List[str]:
    """Read lines from a file, stripping newline characters."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def write_lines(path: str, lines: List[str]):
    """Write lines to a file joined by newline."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))




def sample_language_dir(lang_dir: str, sample_size: int, rng: random.Random, splits_to_sample, out_dir: Path):
    print(f"[LANG] {os.path.basename(lang_dir)}")
    any_sampled = False
    for split in splits_to_sample:
        base_path = os.path.join(lang_dir, split)
        trans_path = f"{base_path}.transcriptions.txt"
        gloss_path = f"{base_path}.glosses.txt"
        transl_path = f"{base_path}.translations.txt"

        if not all(os.path.isfile(p) for p in (trans_path, gloss_path, transl_path)):
            continue

        # Read files
        t_lines = read_lines(trans_path)
        g_lines = read_lines(gloss_path)
        tr_lines = read_lines(transl_path)
        len_g = len(g_lines)
        len_tr = len(tr_lines)
        n = min(len(t_lines), len_g, len_tr)
        if not (len_g == len_tr == len(t_lines)):
            print(f"    [WARN] length mismatch for {split} in {os.path.basename(lang_dir)}: T={len(t_lines)}, G={len_g}, Tr={len_tr}. Using first {n} aligned rows.")

        k = min(sample_size, n)
        idx = rng.sample(range(n), k)
        idx.sort()

        write_lines(f"{base_path}.sample.transcriptions.txt", [t_lines[i] for i in idx])
        write_lines(f"{base_path}.sample.glosses.txt", [g_lines[i] for i in idx])
        write_lines(f"{base_path}.sample.translations.txt", [tr_lines[i] for i in idx])

        print(f"    {split}: sampled {k}/{n}")
        any_sampled = True

    # Determine category based on original path
    parent_name = Path(lang_dir).parent.name
    if parent_name in ["seen", "unseen"]:
        dest_lang_dir = out_dir / parent_name / Path(lang_dir).name
    else:
        dest_lang_dir = out_dir / Path(lang_dir).name

    dest_lang_dir.mkdir(parents=True, exist_ok=True)

    # Always copy full TRAIN files if they exist
    train_prefix = os.path.join(lang_dir, "train")
    for typ in ["transcriptions", "glosses", "translations"]:
        train_file = f"{train_prefix}.{typ}.txt"
        if os.path.isfile(train_file):
            dst_path = dest_lang_dir / Path(os.path.basename(train_file))
            # Skip if source and destination are the same file
            if os.path.abspath(train_file) != os.path.abspath(str(dst_path)):
                shutil.copy2(train_file, dst_path)

    # Copy sample files for selected splits only
    for split in splits_to_sample:
        for typ in ["transcriptions", "glosses", "translations"]:
            sample_file = f"{os.path.join(lang_dir, split)}.sample.{typ}.txt"
            if os.path.isfile(sample_file):
                dst_path = dest_lang_dir / Path(os.path.basename(sample_file))
                # Skip if source and destination are the same file
                if os.path.abspath(sample_file) != os.path.abspath(str(dst_path)):
                    shutil.copy2(sample_file, dst_path)

    if not any_sampled:
        print("    (no valid split files found, skipping)")
    else:
        print(f"    Copied files to {dest_lang_dir}")


def is_language_dir(path: Path) -> bool:
    """Return True if directory contains at least one split file pattern and is not inside samples dir."""
    if "samples" in path.parts:
        return False
    
    for split in ["train", "validation", "test"]:
        if (path / f"{split}.transcriptions.txt").exists():
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Sample parallel sentences for each language & split.")
    parser.add_argument("--root", type=str, default=".", help="Root directory with language folders")
    parser.add_argument("--sample_size", type=int, default=200, help="Samples per split (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", type=str, default="test", help="Comma-separated list of splits to sample (default: 'test')")
    parser.add_argument("--out_dir", type=str, default="samples", help="Directory to collect easy-access files")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Determine which splits to sample
    splits_to_sample = [s.strip() for s in args.splits.split(',') if s.strip()]

    def traverse(dir_path: Path):
        if is_language_dir(dir_path):
            sample_language_dir(str(dir_path), args.sample_size, rng, splits_to_sample, Path(args.out_dir))
        else:
            for child in dir_path.iterdir():
                if child.is_dir():
                    traverse(child)

    traverse(Path(args.root))


if __name__ == "__main__":
    main() 