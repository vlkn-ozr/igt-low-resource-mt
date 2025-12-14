# Turkish Morphological Gloss Simplification

This project simplifies the morphological gloss labels in Turkish text by applying multiple levels of simplification.

## Overview

The simplification process includes:
1. **Basic Simplification**: Merges similar gloss types and removes very low-frequency tags
2. **Max 2/3 Labels Simplification**: Limits each word to a maximum of 2 or 3 gloss labels based on priority
3. **POS Tag Removal**: Removes POS tags from gloss files

## Scripts

### Core Simplification Scripts

- `simplify_glosses.py` - Performs basic simplification (merges similar tags, removes low-frequency tags)
- `simplify_max_2.py` - Limits each word to max 2 labels (generic version)
- `simplify_max_3.py` - Limits each word to max 3 labels (generic version)

### Analysis Scripts

- `count_gloss_types.py` - Counts the frequency of gloss types in a file

### Utility Scripts

- `remove_first_pos_tag.py` - Removes POS tags from gloss files

## Usage

### Basic Simplification

```bash
python simplify_glosses.py --input gloss.txt --output simplified_gloss.txt
```

### Max Labels Simplification

```bash
# Max 3 labels
python simplify_max_3.py input_file.txt [output_file.txt]

# Max 2 labels
python simplify_max_2.py input_file.txt [output_file.txt]
```

If output file is not specified, it will be auto-generated based on the input filename.

### Analysis

```bash
# Count gloss types
python count_gloss_types.py --input gloss.txt --output gloss_counts.txt
```

### Remove POS Tags

```bash
# Remove first POS tag
python remove_first_pos_tag.py input_file.txt

# Remove all POS tags
python remove_first_pos_tag.py input_file.txt --all
```

## Input Format

Gloss files can have the following format:
```
lang_code word1-gloss1-gloss2 word2-gloss1 word3-gloss1-gloss2-gloss3
```

Or without language code:
```
word1-gloss1-gloss2 word2-gloss1 word3-gloss1-gloss2-gloss3
```

Example:
```
tur ev-NOM-PL var-V-3SG
```

## Priority Levels

When limiting to max 2/3 labels, glosses are prioritized as follows:

1. **Level 1** (highest): Part of Speech tags (NOM, V, ADJ, ADV, CNJ, PROP, DET, PRN, POSTP, NUM)
2. **Level 2**: Inflectional morphology (PL, 3SG, PST, FUT, PROG, NEG, etc.)
3. **Level 3**: Case markers (GEN, DAT, ACC, LOC, ABL, INS)
4. **Level 4**: Derivational morphology (VN, PART, CV, CPL, etc.)
5. **Level 5** (lowest): All other tags

## Simplification Rules

### Basic Simplification

- Merges similar tags:
  - `PART:PRES`, `PART:PAST`, `PART:FUT` → `PART`
  - `VN:INF`, `VN:PAST`, etc. → `VN`
  - `CV:EREK`, `CV:KEN`, etc. → `CV`
  - `CPL:PRES`, `CPL:PAST`, etc. → `CPL`
- Removes very low-frequency tags (< 10 occurrences)
- Keeps high-frequency labels and essential grammatical categories

### Max Labels Simplification

- If a word has ≤ N labels, all are kept
- If a word has > N labels, only the N highest-priority labels are kept
- Dropped labels are tracked in statistics files (automatically generated)
