# GlossLM Qwen Finetune

This directory contains scripts for processing the GlossLM dataset for finetuning Qwen models.

## Overview

The scripts in this directory help:
1. Download GlossLM dataset from Hugging Face
2. Create subsets of the dataset (focusing on unsegmented examples)
3. Extract and organize data by language
4. Create source-target training files
5. Filter and deduplicate training data
6. Analyze fieldwork languages in GlossLM

## Scripts

### Data Download and Processing

- `download_glosslm_via_api.py` - Downloads GlossLM dataset from Hugging Face and creates a subset
- `create_final_subset.py` - Creates a final subset focusing on unsegmented examples (Turkic languages: 100%, others: 42%)
- `create_source_target_files.py` - Creates aligned source-target training files with ISO language codes

### Analysis Scripts

- `analyze_fieldwork_in_glosslm.py` - Analyzes fieldwork languages in GlossLM train split using cached parquet
- `fieldwork_glosslm.py` - Analyzes fieldwork samples in target translation file

### Utility Scripts

- `filter_fieldwork_languages.py` - Filters training data to remove specific fieldwork languages
- `find_duplicates.py` - Finds and removes duplicates in training files

## Usage

### Download GlossLM Dataset

```bash
python download_glosslm_via_api.py
```

This downloads the GlossLM train split and saves it to `raw_data/glosslm_train.parquet`, then creates a subset CSV.

### Create Final Subset

```bash
python create_final_subset.py
```

Creates a subset with:
- All unsegmented examples from Turkic languages
- 42% sample of unsegmented examples from non-Turkic languages

### Create Source-Target Files

```bash
# From CSV subset
python create_source_target_files.py

# From deduplicated TSV
python create_source_target_files.py --from-tsv --input training_data_100k_glosslm/aligned_pairs.tsv --output training_data_100k_glosslm_dedup
```

Creates aligned `source.txt` and `target.txt` files with format:
- Source: `iso_code glosses` (e.g., "tur 1SG yesterday evening PRO eat")
- Target: English translation

### Find and Remove Duplicates

```bash
# Find duplicates in aligned pairs TSV
python find_duplicates.py training_data_100k_glosslm/aligned_pairs.tsv --mode aligned --output duplicates.txt

# Remove duplicates
python find_duplicates.py training_data_100k_glosslm/aligned_pairs.tsv --mode aligned --remove

# Find duplicates within a file
python find_duplicates.py source.txt target.txt --mode within

# Find duplicates across files
python find_duplicates.py file1.txt file2.txt --mode across
```

### Filter Fieldwork Languages

```bash
python filter_fieldwork_languages.py
```

Filters training data to keep only Arapaho, Mandarin Chinese, and non-fieldwork languages.

### Analyze Fieldwork in GlossLM

```bash
python analyze_fieldwork_in_glosslm_cached.py
```

Analyzes which fieldwork languages appear in the GlossLM train split.

## Data Format

### Source-Target Format

- **Source file**: Each line contains `iso_code glosses` where `iso_code` is a 3-letter ISO 639-3 code
- **Target file**: Each line contains the corresponding English translation
- Files must be perfectly aligned (same number of lines)

### Aligned Pairs TSV Format

```
line_num	source	target
1	iso_code glosses	translation
2	iso_code glosses	translation
...
```

## Requirements

See `requirements.txt` for required Python packages.

