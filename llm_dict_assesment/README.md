# Turkish-English Dictionary Alignment Assessment Tool

This tool uses Qwen2.5-7B-Instruct to assess whether Turkish-English dictionary word pairs are correctly aligned. Designed for lemmatized dictionaries where both Turkish and English words are in their base/root forms. When incorrect alignments are found, the model suggests the correct English lemma.

## Features

- Automated assessment using state-of-the-art LLM
- Intelligent correction suggestions for incorrect pairs
- Turkish morphology aware evaluation
- Proper noun and abbreviation handling
- Progress saving and resume functionality
- Structured output with multiple dictionary formats
- Batch processing with configurable batch sizes

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- At least 16GB RAM for GPU inference, 32GB+ for CPU
- ~15GB disk space for the model

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python dict_assessment_script.py --dict-file dict_llm_5k_lemma_lowercase.txt
```

### Test Run

```bash
python dict_assessment_script.py --test-run
```

### Advanced Options

```bash
python dict_assessment_script.py \
    --dict-file your_dictionary.txt \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 10 \
    --max-pairs 1000 \
    --resume-from-line 100
```

### Command Line Arguments

- `--dict-file`: Path to dictionary file (default: dict_llm_5k_lemma_lowercase.txt)
- `--model`: Model name or local path (default: Qwen/Qwen2.5-7B-Instruct)
- `--batch-size`: How often to save progress (default: 10)
- `--max-pairs`: Maximum pairs to process (useful for testing)
- `--test-run`: Quick test with 20 pairs
- `--resume-from-line`: Resume from specific line number (1-indexed)
- `--show-progress`: Show current progress status and exit

## Input Format

The dictionary file should contain one Turkish-English pair per line:

```
turkish_word -> english_translation
```

Example:
```
ev -> house
köpek -> dog
123 -> 123
ankara -> ankara
```

## Output Files

All output files are saved in the `results/` directory with timestamps:

1. **`assessment_results_YYYYMMDD_HHMMSS.json`**: Complete results with prompts and model responses
2. **`dict_correct_pairs_YYYYMMDD_HHMMSS.txt`**: Dictionary with only correct pairs
3. **`dict_incorrect_pairs_YYYYMMDD_HHMMSS.txt`**: Dictionary with only incorrect pairs
4. **`dict_corrected_pairs_YYYYMMDD_HHMMSS.txt`**: Dictionary with model-suggested corrections
5. **`dict_clean_combined_YYYYMMDD_HHMMSS.txt`**: Clean dictionary (correct + corrected pairs)
6. **`assessment_summary_YYYYMMDD_HHMMSS.json`**: Summary statistics
7. **`dict_assessment.log`**: Detailed execution log

## Assessment Criteria

The model evaluates pairs based on:

1. **Lemma Validity**: Are both words in their correct base/root forms?
2. **Translation Accuracy**: Is the English lemma a valid translation of the Turkish lemma?
3. **Semantic Equivalence**: Do both lemmas convey the same core meaning?
4. **Proper Noun Handling**: Are proper nouns identical when appropriate?
5. **Type Consistency**: Do corrections maintain the same type as the source?

## Correction Rules

### Type-Matching Requirements

- **Numbers**: If source is a number, correction must be a number
  - ✅ `1000 -> 1000` (not `1000 -> one_thousand`)
- **Text**: If source is text, correction must be text
  - ✅ `ev -> house` (Turkish word to English word)
- **Symbols**: If source is a symbol, correction must be the same symbol
  - ✅ `& -> &` (identical symbols)
- **Abbreviations**: Should maintain abbreviation format
  - ✅ `akp -> akp` (political party abbreviation)
- **Proper Nouns**: Should remain identical
  - ✅ `ankara -> ankara` (city name)

## Response Format

The tool uses structured XML format for responses:

### Correct Alignment
```
<VERDICT>CORRECT</VERDICT>
```

### Incorrect Alignment with Correction
```
<VERDICT>INCORRECT</VERDICT>
<CORRECTION>correct_english_lemma</CORRECTION>
```

## Lemmatization Rules

This tool is designed for **lemmatized dictionaries**:

### Turkish Lemmas
- **Verbs**: Base form without -mek/-mak infinitive endings (e.g., "git" not "gitmek")
- **Nouns**: Singular form without plural suffixes (e.g., "ev" not "evler")
- **Adjectives**: Base form without comparative suffixes (e.g., "büyük" not "büyükçe")

### English Lemmas
- **Verbs**: Infinitive form without "to" (e.g., "go" not "goes" or "going")
- **Nouns**: Singular form (e.g., "house" not "houses")
- **Adjectives**: Base form (e.g., "big" not "bigger")

## Performance

- **GPU**: ~2-4 seconds per pair
- **CPU**: ~8-20 seconds per pair
- **Memory**: 8-16GB GPU VRAM, 16-32GB system RAM

## Troubleshooting

### Model Loading Issues
```bash
export CUDA_VISIBLE_DEVICES=0
python dict_assessment_script.py --test-run
```

### Memory Issues
- Reduce batch size: `--batch-size 5`
- Use CPU: Ensure no CUDA devices available
- Close other applications

### Progress Recovery
Progress is automatically saved. To resume:
```bash
python dict_assessment_script.py --show-progress
python dict_assessment_script.py --resume-from-line <line_number>
```

## Example Run

```bash
$ python dict_assessment_script.py --test-run

2024-12-01 14:30:22,123 - INFO - Using device: cuda
2024-12-01 14:30:22,124 - INFO - Loading model: Qwen/Qwen2.5-7B-Instruct
2024-12-01 14:30:45,678 - INFO - Model loaded successfully
2024-12-01 14:30:45,689 - INFO - Loaded 4177 dictionary pairs
2024-12-01 14:30:45,689 - INFO - Processing first 20 pairs only
2024-12-01 14:30:45,689 - INFO - Starting assessment of 20 pairs
...
2024-12-01 14:31:45,456 - INFO - Results saved to 'results' directory
```

## Use Cases

1. **Dictionary Cleaning**: Remove incorrect alignments and get corrected versions
2. **Quality Assessment**: Measure accuracy of existing dictionaries
3. **Data Improvement**: Enhance dictionary quality with AI-suggested corrections
4. **Translation Preparation**: Prepare clean dictionaries for MT systems
