# TRmorph Notebook Usage Guide

This guide documents the workflow for using TRmorph Python scripts as demonstrated in the notebook (`trmorph.ipynb`).

## Overview

The notebook demonstrates a complete pipeline for:
1. Installing and compiling TRmorph
2. Analyzing Turkish text files
3. Calculating analysis statistics
4. Disambiguating tokens

## Prerequisites

### Install foma

**On Ubuntu/Debian:**
```bash
sudo apt-get install foma libfoma0
```

**On macOS:**
```bash
brew install foma
```

### Download Model File

Download the disambiguation model:
```bash
wget http://www.let.rug.nl/~coltekin/trmorph/1M.m2 -O scripts/1M.m2
```

Or manually download from: http://www.let.rug.nl/~coltekin/trmorph/1M.m2

Place it in the `scripts/` directory.

## Step-by-Step Workflow

### Step 1: Compile TRmorph

```bash
make
```

This creates `trmorph.fst`, the compiled morphological analyzer.

### Step 2: Prepare Input File

Create a text file with one sentence per line:

```bash
cat > preprocessed_sampled_transcription_600.txt << EOF
okudum kitabı
bugün hava çok güzel
...
EOF
```

**Input format:**
- One sentence per line
- UTF-8 encoding
- Words separated by spaces

### Step 3: Analyze Text File

Run the analysis script to analyze all sentences:

```bash
python scripts/analysis.py preprocessed_sampled_transcription_600.txt output_sampled_transcription_600
```

**What this does:**
- Reads sentences from the input file
- Analyzes each word in each sentence using TRmorph
- Saves results to `output_sampled_transcription_600.json`

**Output:**
- Creates `output_sampled_transcription_600.json` with morphological analyses

**Progress:**
The script prints progress for each sentence:
```
Processing sentence 1/600
Processing sentence 2/600
...
```

### Step 4: Calculate Statistics

Get coverage and analysis statistics:

```bash
python scripts/stats.py output_sampled_transcription_600.json output_sampled_transcription_600_stats.json
```

**Output:**
```
Analysis Statistics:
Coverage: 94.8%
Average analyses per word: 15.27
Total words: 12052
Unanalyzed words: 627

Full report saved to: output_sampled_transcription_600_stats.json
```

**Statistics include:**
- Coverage percentage (words successfully analyzed)
- Average analyses per word
- Total words analyzed
- List of unanalyzed words

### Step 5: Disambiguate Tokens (Optional)

Disambiguate tokens using the trained model:

**First, prepare tokenized input (one token per line):**
```bash
# If you have a tokenized file:
cat preprocessed_sampled_transcription_600_tokens.txt | python scripts/disambiguate.py -m scripts/1M.m2 > dis_results_sampled_transcription_600.txt
```

**Or tokenize from sentences:**
```bash
# Convert sentences to tokens (one per line)
cat preprocessed_sampled_transcription_600.txt | tr ' ' '\n' > tokens.txt
cat tokens.txt | python scripts/disambiguate.py -m scripts/1M.m2 > dis_results.txt
```

**Input format for disambiguation:**
- One token per line
- UTF-8 encoding

**Output:**
- Disambiguated analyses with scores
- Best analysis per token (if using `-1` flag)

## Complete Example

Here's the complete workflow from the notebook:

```bash
# 1. Install foma (if not already installed)
sudo apt-get install foma libfoma0

# 2. Compile TRmorph
make

# 3. Download model file (if not already present)
wget http://www.let.rug.nl/~coltekin/trmorph/1M.m2 -O scripts/1M.m2

# 4. Analyze text file
python scripts/analysis.py preprocessed_sampled_transcription_600.txt output_sampled_transcription_600

# 5. Calculate statistics
python scripts/stats.py output_sampled_transcription_600.json output_sampled_transcription_600_stats.json

# 6. Disambiguate tokens (if you have tokenized input)
cat preprocessed_sampled_transcription_600_tokens.txt | python scripts/disambiguate.py -m scripts/1M.m2 > dis_results_sampled_transcription_600.txt
```

## Input/Output Formats

### Input File Format (for analysis.py)

```
okudum kitabı
bugün hava çok güzel
dün okula gittim
```

- One sentence per line
- Words separated by spaces
- UTF-8 encoding

### Analysis JSON Output

```json
{
  "metadata": {
    "input_file": "preprocessed_sampled_transcription_600.txt",
    "timestamp": "2024-01-01 12:00:00",
    "total_sentences": 600
  },
  "sentences": [
    {
      "sentence_id": 1,
      "text": "okudum kitabı",
      "analyses": {
        "okudum": ["oku<v><past><1s>"],
        "kitabı": ["kitap<n><sg><acc>"]
      }
    }
  ]
}
```

### Statistics JSON Output

```json
{
  "timestamp": "2024-01-01 12:00:00",
  "input_file": "preprocessed_sampled_transcription_600.txt",
  "total_sentences": 600,
  "coverage_stats": {
    "total_words": 12052,
    "analyzed_words": 11425,
    "coverage_percentage": 94.8
  },
  "analyses_stats": {
    "average_analyses_per_word": 15.27,
    "max_analyses": 50,
    "min_analyses": 1,
    "total_analyses": 174456
  },
  "unanalyzed_words": ["word1", "word2", ...]
}
```

### Disambiguation Output

```
okudum -2.45: oku<v><past><1s>
kitabı -1.23: kitap<n><sg><acc>
```

Format: `word score: analysis`

## Script Options

### analysis.py

```bash
python scripts/analysis.py <input_file> [output_file]
```

- `input_file`: Text file with one sentence per line
- `output_file`: Base name for output (default: input filename with `.analysis` suffix)
- Output: `{output_file}.json`

### stats.py

```bash
python scripts/stats.py <analysis_file.json> [output_stats.json]
```

- `analysis_file.json`: JSON file from `analysis.py`
- `output_stats.json`: Output statistics file (default: `{analysis_file}.stats.json`)

### disambiguate.py

```bash
cat tokens.txt | python scripts/disambiguate.py [options] > output.txt
```

**Options:**
- `-1, --best-parse`: Only print the best analysis
- `-s, --no-score`: Don't print scores
- `-w, --no-word`: Don't print the word
- `-m, --model-file`: Path to model file (default: `1M.m2`)
- `-f, --flookup-cmd`: Custom flookup command
- `-N, --no-newline`: Suppress newline between analyses

## Troubleshooting

**Error: `flookup: command not found`**
- Install foma: `sudo apt-get install foma libfoma0` (Ubuntu/Debian) or `brew install foma` (macOS)

**Error: `Could not find trmorph.fst`**
- Run `make` to compile the analyzer

**Error: `Cannot open the model file`**
- Download `1M.m2` from http://www.let.rug.nl/~coltekin/trmorph/1M.m2
- Place it in the `scripts/` directory

**Error: `Error analyzing 'word'`**
- Check that `trmorph.fst` exists and is compiled
- Verify foma is installed correctly
- Ensure input is UTF-8 encoded

**Low coverage percentage**
- Some words may be unknown to the analyzer
- Check the `unanalyzed_words` list in statistics
- Verify input contains valid Turkish text

## Notes

- The analysis script processes sentences sequentially and may take time for large files
- The disambiguation model (`1M.m2`) is required for disambiguation but not for basic analysis
- All scripts use standard Python libraries (no additional packages required)
- Input files should be UTF-8 encoded
- The analyzer works best with standard Turkish orthography

