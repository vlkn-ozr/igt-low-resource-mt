# Awesome Align - Word Alignment Tool

This directory contains scripts for word alignment between parallel sentences using Awesome Align.

## Installation

```bash
pip install -r requirements.txt
```

Or install awesome-align from source:
```bash
git clone https://github.com/neulab/awesome-align.git
cd awesome-align
pip install -r requirements.txt
python setup.py install
```

## Files

- `awesome_align.ipynb`: Main notebook for running alignment experiments
- `create_dictionary.py`: Create aligned dictionary from word pairs
- `train_alignments.sh`: Train Awesome Align model
- `extract_alignments.sh`: Extract alignments using trained model

## Usage

### Training

```bash
bash train_alignments.sh [train_file] [eval_file] [output_dir] [model_name]
```

### Alignment

```bash
bash extract_alignments.sh [data_file] [model_path] [output_file]
```

### Create Dictionary

```bash
python create_dictionary.py --input aligned_words.txt --output dictionary.txt
```

## Input Format

Parallel sentences should be separated by `|||`:
```
source sentence ||| target sentence
```

## Output Format

- Alignments: `i-j` format (Pharaoh format)
- Aligned words: `source|||target` format
- Dictionary: `source -> target` format 