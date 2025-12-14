# GlossLM Data Processing

This directory contains scripts for processing the GlossLM dataset.

## Scripts

- `download_glosslm_via_api.py`: Download GlossLM dataset from HuggingFace
- `analyze_glosslm.py`: Analyze GlossLM dataset statistics
- `create_final_subset.py`: Create a subset of the dataset with Turkic languages
- `create_source_target_files.py`: Create aligned source and target files for training
- `filter_turkish.py`: Filter Turkish entries from specific sources
- `find_duplicates.py`: Find and remove duplicate entries in files
- `add_lang_code.py`: Add language code prefix to text files

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset:
   ```bash
   python download_glosslm_via_api.py
   ```

2. Analyze the dataset:
   ```bash
   python analyze_glosslm.py
   ```

3. Create subset:
   ```bash
   python create_final_subset.py
   ```

4. Create training files:
   ```bash
   python create_source_target_files.py
   ```

