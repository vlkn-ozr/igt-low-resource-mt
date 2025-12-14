# Gloss Data Processing Scripts

This directory contains scripts for processing and converting morphological glosses to GlossLM format.

## Scripts

### create_gloss_mapping.py

Creates a mapping file from morphological tags to GlossLM format tags.

**Usage:**
```bash
python create_gloss_mapping.py [output_file]
```

**Example:**
```bash
python create_gloss_mapping.py gloss_mapping.txt
```

**Output:**
Creates a mapping file with morphological tag to GlossLM tag conversions, including:
- Part of Speech tags (N, V, Adj, etc.)
- Case markers (nom, acc, dat, etc.)
- Verb features (past, pres, fut, etc.)
- Person/Number markers (p1s, p2s, p3s, etc.)

### convert_glosses.py

Converts morphological analyses to GlossLM format using a mapping file.

**Usage:**
```bash
python convert_glosses.py <input_file> <output_file> [mapping_file]
```

**Example:**
```bash
python convert_glosses.py disambiguated_glosses.txt glosslm_glosses.txt gloss_mapping.txt
```

**Input format:**
One line per sentence, with space-separated morphological analyses:
```
okudum<v><past><1s> kitabı<n><sg><acc>
```

**Output format:**
One line per sentence, with space-separated GlossLM format analyses:
```
okudum-V.PST.1SG kitabı-NOM.SG.ACC
```

### process_disambiguated.py

Processes disambiguated morphological analysis results and extracts glosses.

**Usage:**
```bash
python process_disambiguated.py <input_file> <output_file> <sentence_file>
```

**Example:**
```bash
python process_disambiguated.py dis_results.txt disambiguated_glosses.txt preprocessed_sentences.txt
```

**Parameters:**
- `input_file`: Disambiguated analysis results file (with probabilities)
- `output_file`: Output file for processed glosses
- `sentence_file`: Original sentence file (one sentence per line)

**What it does:**
- Extracts best analyses from disambiguated results
- Maps analyses back to sentences
- Writes glosses in morphological format
- Creates `unanalyzed_words.txt` with words that couldn't be analyzed

### remove_tags.py

Removes angle bracket tags (`<...>`) from text files.

**Usage:**
```bash
python remove_tags.py <input_file> <output_file>
```

**Example:**
```bash
python remove_tags.py disambiguated_glosses.txt cleaned_glosses.txt
```

**What it does:**
- Removes all content between `<` and `>` brackets
- Useful for cleaning morphological analyses

## Complete Workflow

### Step 1: Create Mapping File

```bash
python create_gloss_mapping.py gloss_mapping.txt
```

### Step 2: Process Disambiguated Results

```bash
python process_disambiguated.py dis_results.txt disambiguated_glosses.txt sentences.txt
```

### Step 3: Convert to GlossLM Format

```bash
python convert_glosses.py disambiguated_glosses.txt glosslm_glosses.txt gloss_mapping.txt
```


### Step 5: Remove Tags (Optional)

```bash
python remove_tags.py cleaned_glosslm_glosses.txt final_glosses.txt
```

## Input/Output Formats

### Disambiguated Results Format

```
-2.45: okudum oku<v><past><1s>
-1.23: kitabı kitap<n><sg><acc>
```

Format: `probability: word analysis`

### Morphological Analysis Format

```
okudum<v><past><1s> kitabı<n><sg><acc>
```

### GlossLM Format

```
okudum-V.PST.1SG kitabı-NOM.SG.ACC
```

Format: `word-TAG1.TAG2.TAG3`

## Requirements

- Python 3
- Standard library only (no external dependencies)

## Notes

- All scripts use UTF-8 encoding
- Input files should have one sentence per line
- The mapping file can be customized by editing `create_gloss_mapping.py`
- Unanalyzed words are tracked in `unanalyzed_words.txt` (created by `process_disambiguated.py`)

