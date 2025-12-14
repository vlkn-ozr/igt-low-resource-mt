# SETIMES-v2 Processing Scripts

This directory contains scripts for processing the SETIMES parallel corpus (Turkish-English news articles) and creating aligned dictionaries.

## Scripts

### create_aligned_dictionary.py

Creates a simple Turkish-English dictionary from aligned parallel corpus.

**Usage:**
```bash
python create_aligned_dictionary.py <tr_file> <en_file> <alignment_file> <output_file>
```

**Example:**
```bash
python create_aligned_dictionary.py tr_corpus.txt en_corpus.txt alignments.txt dictionary.txt
```

**What it does:**
- Processes aligned sentence pairs
- Counts word alignments
- Selects most frequent English translation for each Turkish word
- Outputs simple dictionary format: `turkish_word -> english_word`

### create_aligned_dictionary_improved.py

Creates an improved Turkish-English dictionary with filtering options.

**Usage:**
```bash
python create_aligned_dictionary_improved.py \
    --tr-file <tr_file> \
    --en-file <en_file> \
    --alignment-file <alignment_file> \
    --output-file <output_file> \
    [--min-freq N] \
    [--max-translations N] \
    [--include-counts] \
    [--keep-function-words]
```

**Example:**
```bash
python create_aligned_dictionary_improved.py \
    --tr-file tr_corpus.txt \
    --en-file en_corpus.txt \
    --alignment-file alignments.txt \
    --output-file dictionary.txt \
    --min-freq 3 \
    --max-translations 3
```

**Parameters:**
- `--tr-file`: Turkish text file (one sentence per line)
- `--en-file`: English text file (one sentence per line)
- `--alignment-file`: Word alignment file (format: "0-1 2-3 ...")
- `--output-file`: Output dictionary file
- `--min-freq`: Minimum frequency for alignments (default: 1)
- `--max-translations`: Maximum translations per Turkish word (default: 1)
- `--include-counts`: Include alignment counts in output
- `--keep-function-words`: Don't filter out English function words

**Output format:**
```
turkish_word -> english_word1, english_word2, english_word3
```

### filter_translations.py

Filters English translations by removing function words.

**Usage:**
```bash
python filter_translations.py <input_file> <output_file> [--keep-function-words] [--min-length N]
```

**Example:**
```bash
python filter_translations.py translations.txt filtered_translations.txt --min-length 2
```

**Parameters:**
- `input_file`: Input translations file
- `output_file`: Output filtered translations file
- `--keep-function-words`: Keep function words (default: remove them)
- `--min-length`: Minimum word length to keep (default: 2)

**Output:**
- Filtered translations file
- Report file (`<output_file>.report.txt`) with statistics

## Complete Workflow

### Step 1: Filter English Translations (Optional)

Filter out function words from English translations:

```bash
python filter_translations.py en_corpus.txt filtered_en.txt
```

### Step 2: Run Word Alignment

Use fast_align, awesome-align, or another word alignment tool on your parallel corpus to generate alignments.

**Input format for alignment tools:**
```
turkish_sentence ||| english_sentence
```

**Output format:**
```
0-1 2-3 4-5
```

Each pair `i-j` means Turkish word at position `i` aligns with English word at position `j`.

### Step 3: Create Dictionary

**Simple dictionary (one translation per word):**
```bash
python create_aligned_dictionary.py tr_corpus.txt en_corpus.txt alignments.txt dictionary.txt
```

**Improved dictionary (multiple translations, filtering options):**
```bash
python create_aligned_dictionary_improved.py \
    --tr-file tr_corpus.txt \
    --en-file filtered_en.txt \
    --alignment-file alignments.txt \
    --output-file dictionary.txt \
    --min-freq 3 \
    --max-translations 3
```

## Input/Output Formats

### Alignment Format

```
0-1 2-3 4-5
```

Each pair `i-j` means Turkish word at position `i` aligns with English word at position `j`.

### Dictionary Format

**Simple format:**
```
turkish_word -> english_word
```

**Improved format (with multiple translations):**
```
turkish_word -> english_word1, english_word2, english_word3
```

**With counts:**
```
turkish_word -> english_word1 (15), english_word2 (8), english_word3 (3)
```

## Requirements

- Python 3
- Standard library only (no external dependencies)

## Notes

- All scripts use UTF-8 encoding
- Turkish character handling is supported
- Function words can be filtered when creating dictionaries
- Alignment files should have one alignment line per sentence pair
- The SETIMES corpus is available from OPUS: http://opus.nlpl.eu/SETIMES-v2.php
- Both dictionary creation scripts require aligned parallel corpus files with the same number of lines

## TurkishSpellChecker-Py

This directory contains a separate Turkish spell checker package. See `TurkishSpellChecker-Py/README.md` for details.
