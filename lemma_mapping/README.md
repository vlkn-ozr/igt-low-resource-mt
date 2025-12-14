# Source-Target Processing Scripts

This directory contains scripts for processing Turkish-English parallel data, creating dictionaries, and translating glosses.

## Scripts

### check_duplicates.py

Analyzes dictionary files for duplicate Turkish word entries.

**Usage:**
```bash
python check_duplicates.py <dictionary_file>
```

**Example:**
```bash
python check_duplicates.py tr_en_dictionary.txt
```

**Output:**
- Total lines with entries
- Unique Turkish words
- Number of duplicate words
- Top 20 most duplicated words

### clean_turkish_glosses.py

Cleans Turkish glosses by removing punctuation and handling quotes.

**Usage:**
```bash
python clean_turkish_glosses.py --input-file <input_file> --output-file <output_file>
```

**Example:**
```bash
python clean_turkish_glosses.py --input-file glosses.txt --output-file cleaned_glosses.txt
```

**What it does:**
- Removes punctuation marks (except hyphens in morphological tags)
- Converts words to lowercase if they had quotes
- Preserves hyphen-separated morphological tags (e.g., 'word-NOM')
- Shows progress for large files

### create_aligned_dictionary_improved.py

Creates a Turkish-English dictionary from parallel corpus and word alignments.

**Usage:**
```bash
python create_aligned_dictionary_improved.py \
    --tr-file <turkish_file> \
    --en-file <english_file> \
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
- `--max-translations`: Maximum translations per Turkish word (default: 3)
- `--include-counts`: Include alignment counts in output
- `--keep-function-words`: Don't filter out English function words

**Output format:**
```
turkish_word -> english_word1, english_word2, english_word3
```

### lowercase.py

Converts text to lowercase, handling Turkish characters.

**Usage:**
```bash
python lowercase.py <input_file> <output_file>
```

**Example:**
```bash
python lowercase.py glosses.txt glosses_lowercase.txt
```

**What it does:**
- Converts text to lowercase
- Handles Turkish characters (Ş, Ç, Ö, Ğ, Ü, İ, ı)
- Preserves sentence structure

### lowercase_dict.py

Converts dictionary file to lowercase.

**Usage:**
```bash
python lowercase_dict.py <input_file> <output_file>
```

**Example:**
```bash
python lowercase_dict.py dictionary.txt dictionary_lowercase.txt
```

**What it does:**
- Converts all words in dictionary to lowercase
- Handles Turkish characters correctly

### translate_gloss.py

Translates Turkish glosses to English using a dictionary.

**Usage:**
```bash
python translate_gloss.py <dict_file> <gloss_file> <output_file>
```

**Example:**
```bash
python translate_gloss.py tr_en_dict.txt turkish_glosses.txt english_glosses.txt
```

**What it does:**
- Loads Turkish-English dictionary
- Translates Turkish words in glosses to English
- Preserves morphological tags (e.g., "word-NOM" → "translation-NOM")
- Handles case-insensitive matching
- Preserves original case patterns when possible

**Input format:**
```
akşam-NOM eve-DAT git-V.PST.1SG
```

**Output format:**
```
evening-NOM home-DAT go-V.PST.1SG
```

### translate_gloss_stats.py

Translates Turkish glosses to English and generates detailed statistics.

**Usage:**
```bash
python translate_gloss_stats.py \
    --dict-path <dict_file> \
    --gloss-path <gloss_file> \
    --output-path <output_file> \
    [--stats-path <stats_file>] \
    [--replaced-words-path <replaced_file>] \
    [--not-replaced-words-path <not_replaced_file>]
```

**Example:**
```bash
python translate_gloss_stats.py \
    --dict-path dictionary.txt \
    --gloss-path turkish_glosses.txt \
    --output-path english_glosses.txt \
    --stats-path stats/translation_stats.json
```

**Parameters:**
- `--dict-path`: Turkish-English dictionary file
- `--gloss-path`: Turkish gloss file to translate
- `--output-path`: Output English gloss file
- `--stats-path`: JSON file for statistics (default: translation_stats/translation_stats.json)
- `--replaced-words-path`: File to save replaced words (default: translation_stats/replaced_words.txt)
- `--not-replaced-words-path`: File to save non-replaced words (default: translation_stats/not_replaced_words.txt)
- `--skipped-entries-path`: File to save skipped dictionary entries
- `--duplicates-path`: File to save duplicate dictionary entries

**Output:**
- Translated gloss file
- Statistics JSON with:
  - Total words processed
  - Replacement rate
  - Dictionary usage rate
  - Unique replaced/not-replaced words
- Lists of replaced and non-replaced words
- Dictionary loading statistics

## Complete Workflow

### Step 1: Create Dictionary from Alignments

```bash
python create_aligned_dictionary_improved.py \
    --tr-file tr_corpus.txt \
    --en-file en_corpus.txt \
    --alignment-file alignments.txt \
    --output-file dictionary.txt \
    --min-freq 3 \
    --max-translations 3
```

### Step 2: Check for Duplicates

```bash
python check_duplicates.py dictionary.txt
```

### Step 3: Clean Dictionary (Optional)

```bash
python lowercase_dict.py dictionary.txt dictionary_lowercase.txt
```

### Step 4: Clean Turkish Glosses

```bash
python clean_turkish_glosses.py \
    --input-file turkish_glosses.txt \
    --output-file cleaned_glosses.txt
```

### Step 5: Translate Glosses

```bash
python translate_gloss_stats.py \
    --dict-path dictionary.txt \
    --gloss-path cleaned_glosses.txt \
    --output-path english_glosses.txt \
    --stats-path stats/translation_stats.json
```

## Input/Output Formats

### Dictionary Format

```
turkish_word -> english_word1, english_word2, english_word3
```

### Alignment Format

```
0-1 2-3 4-5
```

Each pair `i-j` means Turkish word at position `i` aligns with English word at position `j`.

### Gloss Format

```
word1-TAG1 word2-TAG2 word3-TAG3
```

## Requirements

- Python 3
- Standard library only (no external dependencies)

## Notes

- All scripts use UTF-8 encoding
- Turkish character handling is supported (Ş, Ç, Ö, Ğ, Ü, İ, ı)
- Dictionary lookups are case-insensitive
- Morphological tags are preserved during translation
- Function words can be filtered when creating dictionaries

