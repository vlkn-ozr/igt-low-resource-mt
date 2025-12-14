# TRMOR Scripts

This directory contains Python utilities for working with TRMOR morphological analyzer.

## Scripts

### analysis.py

Analyzes Turkish words from a text file and saves results as JSON.

**Usage:**
```bash
python analysis.py <input_file> [output_file]
```

**Example:**
```bash
python analysis.py input.txt output
```

This will:
- Read sentences from the input file (one sentence per line)
- Analyze each word in each sentence using TRMOR
- Save results to `output.json`

**Input format:** One sentence per line
```
okudum kitabı
bugün hava güzel
```

**Output:** JSON file with analyses for each word

### stats.py

Calculates statistics from analysis JSON files.

**Usage:**
```bash
python stats.py <analysis_file.json> [output_stats.json]
```

**Example:**
```bash
python stats.py output.json output.stats.json
```

**Output includes:**
- Coverage percentage (percentage of words successfully analyzed)
- Average analyses per word
- Total words analyzed
- List of unanalyzed words

### stats_duplicate.py

Removes duplicate unanalyzed words from statistics and recalculates coverage.

**Usage:**
```bash
python stats_duplicate.py <stats_file.json> [output_file.json]
```

**Example:**
```bash
python stats_duplicate.py output.stats.json output.stats_unique.json
```

## Complete Workflow

```bash
# 1. Compile TRMOR (in parent directory)
cd ..
make

# 2. Analyze text file
cd script
python analysis.py input.txt output

# 3. Get statistics
python stats.py output.json output.stats.json

# 4. Remove duplicate statistics (optional)
python stats_duplicate.py output.stats.json output.stats_unique.json
```

## Output Format

### Analysis JSON

```json
{
  "metadata": {
    "input_file": "input.txt",
    "timestamp": "2024-01-01 12:00:00",
    "total_sentences": 2
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

## Requirements

- TRMOR compiled (run `make` in parent directory)
- `fst-mor` command available (part of TRMOR)
- Python 3 (standard library only)

## Notes

- The analyzer requires `morph.a` to be compiled in the parent directory
- Input files should be UTF-8 encoded
- The analysis script processes sentences sequentially

