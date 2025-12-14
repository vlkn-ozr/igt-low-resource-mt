# Neural Machine Translation with BPE Tokenization

This repository contains code for training and evaluating neural machine translation models using Byte Pair Encoding (BPE) tokenization with OpenNMT-py.

## Requirements

- Python 3.6+
- OpenNMT-py
- SentencePiece
- PyTorch
- sacrebleu
- Other dependencies (see requirements.txt if available)

Install dependencies:
```bash
pip install opennmt-py sentencepiece torch sacrebleu pyyaml psutil GPUtil matplotlib numpy pandas spacy nltk
```

## Directory Structure

```
onmt_multi/
├── scripts/              # Python scripts for preprocessing, training, translation, and evaluation
├── configs/              # OpenNMT configuration files
├── README.md            # This file
└── data/                # Data directory (to be created)
    ├── raw/             # Raw data files
    └── processed/       # Processed data files
```

## Quick Start

### 1. Prepare Data

Place your raw data files in the data directory:
- Source file (e.g., `gloss.txt` or `transcription.txt`)
- Target file (e.g., `translation.txt`)

### 2. Preprocess Data with BPE

Preprocess the data using SentencePiece BPE tokenization:

```bash
python scripts/preprocess_bpe.py --vocab_size 8000 --model_type bpe
```

Options:
- `--vocab_size`: Vocabulary size for SentencePiece model (default: 8000)
- `--model_type`: SentencePiece model type - 'bpe' or 'unigram' (default: 'bpe')
- `--save_raw`: Save raw untokenized versions of test and validation data (default: True)

This script will:
- Split data into train/validation/test sets
- Train SentencePiece models for source and target languages
- Apply BPE tokenization to all splits
- Save processed files and vocabulary files

### 3. Create Configuration File

Copy the example configuration file and update paths:

```bash
cp configs/config_example.yaml configs/my_config.yaml
# Edit configs/my_config.yaml to match your data paths
```

Or create a new configuration file:

```bash
python scripts/create_config_bpe.py --vocab_size 8000 --model_type bpe
```

### 4. Train Model

Train the NMT model:

```bash
python scripts/train_onmt.py \
    --config configs/config_example.yaml \
    --gpu 0 \
    --experiment_name my_experiment \
    --monitor_resources
```

Options:
- `--config`: Path to configuration file (required)
- `--gpu`: GPU device ID (default: 0)
- `--experiment_name`: Name of the experiment for logging (default: 'onmt_training')
- `--log_dir`: Directory to save logs
- `--monitor_resources`: Monitor system resources during training
- `--monitor_interval`: Interval in seconds for resource monitoring (default: 60)
- `--train_from`: Path to checkpoint to continue training from
- `--continue_from_last`: Continue from the last checkpoint in models directory

### 5. Translate

Translate text using a trained model:

```bash
python scripts/translate_onmt_bpe.py \
    --model path/to/model.pt \
    --input path/to/input.txt \
    --output path/to/output.txt \
    --config configs/config_example.yaml \
    --gpu 0
```

Options:
- `--model`: Path to trained model (required)
- `--input`: Path to input file (required)
- `--output`: Path to output file (required)
- `--config`: Path to configuration file with BPE settings
- `--gpu`: GPU device ID (default: 0)
- `--language`: Language code for detokenization rules (default: 'en')
- `--normalize_unicode`: Apply Unicode normalization
- `--consistency_check`: Perform consistency check on detokenization

### 6. Evaluate

Evaluate translations:

```bash
python scripts/evaluate_onmt.py \
    --reference path/to/reference.txt \
    --hypothesis path/to/hypothesis.txt \
    --input path/to/input.txt \
    --experiment_name evaluation
```

Options:
- `--reference`: Path to reference file (required)
- `--hypothesis`: Path to hypothesis file (required)
- `--input`: Path to input file (required)
- `--experiment_name`: Name of the experiment for logging
- `--reference_type`: Type of reference - 'raw' or 'tokenized' (default: 'raw')

For linguistic metrics evaluation:

```bash
python scripts/evaluate_linguistic_metrics.py \
    --reference path/to/reference.txt \
    --hypothesis path/to/hypothesis.txt \
    --language en
```

## Complete Pipeline Example

```bash
# 1. Preprocess data
python scripts/preprocess_bpe.py --vocab_size 8000

# 2. Prepare config file
cp configs/config_example.yaml configs/my_config.yaml
# Edit configs/my_config.yaml to match your data paths

# 3. Train model
python scripts/train_onmt.py \
    --config configs/my_config.yaml \
    --gpu 0 \
    --experiment_name my_experiment

# 4. Translate test set
python scripts/translate_onmt_bpe.py \
    --model models/rnn_gloss_nmt_bpe_step_10000.pt \
    --input data/processed/test.transcription \
    --output translations.txt \
    --config configs/my_config.yaml

# 4. Evaluate
python scripts/evaluate_onmt.py \
    --reference data/processed/test.translation.raw \
    --hypothesis translations.txt \
    --input data/processed/test.transcription.raw
```

## Configuration Files

Configuration files are YAML files that specify:
- Data paths (training, validation, test)
- Model architecture (RNN, Transformer)
- Training hyperparameters (learning rate, batch size, etc.)
- BPE model paths
- Output paths

Edit configuration files in the `configs/` directory or create new ones using `create_config_bpe.py`.

## Data Format

Input data should be plain text files with one sentence per line:
- Source file: One source sentence per line
- Target file: One target sentence per line (aligned with source)

Example:
```
# source.txt
hello world
how are you

# target.txt
hola mundo
cómo estás
```

## Output Files

After preprocessing:
- `train.transcription`, `train.translation`: Tokenized training data
- `valid.transcription`, `valid.translation`: Tokenized validation data
- `test.transcription`, `test.translation`: Tokenized test data
- `spm.transcription.model`, `spm.translation.model`: SentencePiece models
- `vocab.transcription`, `vocab.translation`: Vocabulary files

After training:
- Model checkpoints in the models directory
- Training logs in the logs directory
- Metrics plots (if enabled)

## Troubleshooting

1. **File not found errors**: Ensure data files are in the correct directory and paths in configuration files are correct.

2. **GPU out of memory**: Reduce batch size in configuration file or use gradient accumulation.

3. **BPE model not found**: Ensure preprocessing step completed successfully and BPE model paths in config are correct.

4. **Import errors**: Install all required dependencies listed in Requirements section.

## Scripts Overview

- `preprocess_bpe.py`: Preprocess data with BPE tokenization
- `train_onmt.py`: Train NMT model using OpenNMT-py
- `translate_onmt_bpe.py`: Translate text with BPE support
- `evaluate_onmt.py`: Evaluate translations using BLEU scores
- `evaluate_linguistic_metrics.py`: Evaluate using linguistic metrics
- `create_config_bpe.py`: Generate OpenNMT configuration files
- `detokenization_utils.py`: Utilities for detokenizing BPE output
- `nmt_logger.py`: Logging utilities for experiments

## Citation

If you use this code in your research, please cite the relevant paper.

