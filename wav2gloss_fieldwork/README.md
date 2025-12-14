# Wav2Gloss Fieldwork Dataset Processing

This directory contains scripts for downloading, processing, and analyzing the wav2gloss/fieldwork dataset from Hugging Face.

## Overview

The wav2gloss/fieldwork dataset contains parallel data with transcriptions, glosses, and translations for multiple languages. These scripts help:

1. Download the dataset (text columns only, no audio)
2. Extract and organize data by language and split
3. Sample parallel sentences for analysis
4. Analyze overlap with GlossLM dataset
5. Calculate typological distances using WALS features

## Scripts

### Data Download and Extraction

- `download_dataset.py` - Downloads the wav2gloss/fieldwork dataset from Hugging Face (text columns only)
- `extract_by_language.py` - Extracts and organizes data by language and split into `seen/` and `unseen/` directories

### Analysis Scripts

- `fieldwork_glosslm.py` - Analyzes language overlap between fieldwork and GlossLM datasets
- `wals_typological_distance.py` - Calculates typological similarity between languages using WALS features
- `sample_parallel.py` - Samples aligned sentences for each language and split

## Usage

### Download Dataset

```bash
python download_dataset.py
```

This downloads the dataset and saves it to `./wav2gloss_fieldwork_text_only/` with only text columns (transcription, gloss, translation, language).

### Extract by Language

```bash
python extract_by_language.py
```

This organizes the dataset by language into:
- `seen/<language>/` - Languages with training data
- `unseen/<language>/` - Languages without training data

Each language directory contains files like:
- `train.transcriptions.txt`
- `train.glosses.txt`
- `train.translations.txt`
- `validation.transcriptions.txt`
- `test.transcriptions.txt`
- etc.

### Sample Parallel Sentences

```bash
python sample_parallel.py --root . --sample_size 200 --splits test --out_dir samples
```

Options:
- `--root`: Root directory with language folders (default: ".")
- `--sample_size`: Number of samples per split (default: 200)
- `--splits`: Comma-separated list of splits to sample (default: "test")
- `--seed`: Random seed (default: 42)
- `--out_dir`: Output directory for samples (default: "samples")

### Analyze Overlap with GlossLM

```bash
python fieldwork_glosslm.py
```

This script requires `fieldwork_iso.txt` to map glottocodes to language names. It analyzes which languages appear in both datasets.

### Calculate Typological Distances

```bash
python wals_typological_distance.py
```

This script:
1. Downloads WALS (World Atlas of Language Structures) data
2. Matches languages from the fieldwork dataset to WALS IDs
3. Calculates typological similarity to English
4. Saves results to `wals_typological_distances.csv`

Requires `fieldwork_iso.txt` for language name mapping.

## Data Format

The dataset contains parallel data with:
- **Transcription**: Phonetic/phonemic transcription of the utterance
- **Gloss**: Morphological glosses
- **Translation**: English translation
- **Language**: Glottocode of the language

## Directory Structure

After running `extract_by_language.py`, the structure will be:

```
seen/
  <language_code>/
    train.transcriptions.txt
    train.glosses.txt
    train.translations.txt
    validation.transcriptions.txt
    validation.glosses.txt
    validation.translations.txt
    test.transcriptions.txt
    test.glosses.txt
    test.translations.txt
unseen/
  <language_code>/
    test.transcriptions.txt
    test.glosses.txt
    test.translations.txt
```

## Requirements

See `requirements.txt` for required Python packages.

