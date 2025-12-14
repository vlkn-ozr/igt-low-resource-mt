# GlossLM Out-of-Domain Language Data Extraction

This repository contains scripts to extract and process data for out-of-domain languages from the GlossLM corpus. The scripts are designed to work with the four languages mentioned in Table 2 of the GlossLM paper:

- **Gitksan (git)** - 74 training samples
- **Lezgi (lez)** - 705 training samples  
- **Natugu (ntu)** - 791 training samples
- **Nyangbo (nyb)** - 2,100 training samples

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.7+ with the following packages:
   ```bash
   pip install pandas huggingface_hub
   ```

2. **Conda Environment** (recommended): If using conda, activate your environment:
   ```bash
   conda activate morph  # or your preferred environment name
   ```

## Script Execution Order

### Step 1: Download the Dataset

First, download the GlossLM corpus from Hugging Face:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lecslab/glosslm-corpus-split', repo_type='dataset', local_dir='glosslm-corpus-split')"
```

This will create a `glosslm-corpus-split/` directory with the dataset files.

### Step 2: Extract All Language Data (Optional)

Run the initial extraction script to get all data for the four languages:

```bash
python extract_data.py
```

**Output**: 
- `extracted_data/` directory with subdirectories for each language
- Contains both segmented and unsegmented data
- Provides statistics and sample data for inspection

### Step 3: Extract Unsegmented Data Only

Run the script to extract only unsegmented samples (marked as 'no' in the segmentation field):

```bash
python extract_unsegmented.py
```

**Output**:
- `unsegmented_data_strict/` directory with filtered data
- Only includes samples where `is_segmented == 'no'`
- Matches the counts from Table 2 in the paper

### Step 4: Extract Text Fields by Split

Extract transcription, glosses, and translation into separate text files for each split:

```bash
python extract_text_fields.py
```

**Output**:
- `text_fields/` directory
- Separate `.txt` files for each field and split
- Format: `{language}/{split}.{field}.txt`

### Step 5: Extract Filtered Text Fields

Extract text fields while filtering out entries with empty glosses or translations:

```bash
python extract_text_fields_filtered.py
```

**Output**:
- `text_fields_filtered/` directory
- Only includes complete gloss-translation pairs
- **Note**: Nyangbo will have 0 samples as all translations are empty

### Step 6: Create Complete Combined Files

Generate combined text files with all splits merged:

```bash
python extract_complete_files.py
```

**Output**:
- `text_fields_complete/` directory with three subdirectories:
  - `complete/`: All data including empty values
  - `filtered/`: Only complete gloss-translation pairs
  - `nyb_transcription_glosses/`: Special handling for Nyangbo (transcription-glosses pairs only)

## Directory Structure After Running All Scripts

```
├── glosslm-corpus-split/           # Downloaded dataset
├── extracted_data/                 # All language data (Step 2)
├── unsegmented_data_strict/        # Unsegmented data only (Step 3)
├── text_fields/                    # Text fields by split (Step 4)
├── text_fields_filtered/           # Filtered text fields by split (Step 5)
├── text_fields_complete/           # Combined text files (Step 6)
│   ├── complete/
│   ├── filtered/
│   └── nyb_transcription_glosses/
└── scripts...
```

## Data Statistics

### Unsegmented Sample Counts (Step 3 output)

| Language | Train | Eval | Test | Total |
|----------|-------|------|------|-------|
| Gitksan (git) | 74 | 42 | 37 | 153 |
| Lezgi (lez) | 705 | 88 | 87 | 880 |
| Natugu (ntu) | 791 | 99 | 99 | 989 |
| Nyangbo (nyb) | 2100 | 263 | 263 | 2626 |

### Filtered Data (Steps 5-6 output)

| Language | Complete Samples | Filtered Samples | Retention Rate |
|----------|------------------|------------------|----------------|
| Gitksan (git) | 153 | 151 | 98.7% |
| Lezgi (lez) | 880 | 866 | 98.4% |
| Natugu (ntu) | 989 | 989 | 100% |
| Nyangbo (nyb) | 2626 | 0 | 0% (no translations) |

## Important Notes

1. **Nyangbo Translations**: All Nyangbo samples have empty translations. Use the special transcription-glosses files in `text_fields_complete/nyb_transcription_glosses/` for this language.

2. **Segmentation**: The scripts distinguish between:
   - `'yes'`: Morphologically segmented (with hyphens and boundaries)
   - `'no'`: Unsegmented (morphemes combined without separators)
   - `''`: Empty string (treated separately)

3. **File Formats**: All output text files have one entry per line, making them suitable for NLP processing pipelines.

4. **Data Integrity**: The filtered versions ensure that transcription, glosses, and translation files have the same number of lines and correspond to each other.

## Usage Examples

### For Machine Translation
Use the filtered complete files:
```
text_fields_complete/filtered/{language}.transcription.txt
text_fields_complete/filtered/{language}.translation.txt
```

### For Morphological Analysis
Use the complete files with glosses:
```
text_fields_complete/complete/{language}.transcription.txt
text_fields_complete/complete/{language}.glosses.txt
```

### For Nyangbo (Transcription-Glosses Only)
```
text_fields_complete/nyb_transcription_glosses/nyb.transcription.txt
text_fields_complete/nyb_transcription_glosses/nyb.glosses.txt
```

## Troubleshooting

1. **Missing Files**: Ensure you've run the scripts in order and that the dataset was downloaded successfully.

2. **Empty Output**: Check that the `glosslm-corpus-split/` directory contains the expected `.parquet` files.

3. **Permission Errors**: Ensure you have write permissions in the current directory.

4. **Memory Issues**: The scripts process data incrementally, but very large datasets might require more RAM.

## References

- GlossLM Paper: [https://aclanthology.org/2024.emnlp-main.683.pdf](https://aclanthology.org/2024.emnlp-main.683.pdf)
- Hugging Face Dataset: [https://huggingface.co/lecslab/glosslm](https://huggingface.co/lecslab/glosslm) 