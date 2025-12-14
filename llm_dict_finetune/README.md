# Turkish-English Word Alignment Finetuning System

This system finetunes Qwen2.5-7B-Instruct to output word alignments between Turkish and English parallel sentences using parallel corpus and dictionary data.

## Overview

The system provides:
- **Data Processing Pipeline**: Processes parallel sentences and dictionary to create training examples
- **Training Infrastructure**: Finetunes Qwen2.5-7B-Instruct using PEFT/LoRA
- **Inference Pipeline**: Generates word alignments for Turkish-English sentence pairs
- **Dictionary Building**: Constructs Turkish-English dictionaries from alignment outputs

## Installation

```bash
pip install -r requirements.txt
```

## Data Structure

The system expects the following data files:
- `parallel_data_40k/merged_transcriptions_40k.txt`: Turkish sentences (inflected forms)
- `parallel_data_40k/merged_translations_40k.txt`: English sentences
- `dict_40k.txt`: Turkish→English word alignments
- `lemmatized_dict.txt`: English inflected→lemma mappings
- `glosslm_lemma.txt`: Turkish words in lemma form

## Quick Start

### 1. Process Data

```bash
python src/data_processor.py --max_examples 2000
```

This creates training examples in `outputs/training_data.jsonl`.

### 2. Train Model

```bash
python src/trainer.py --config configs/config_example.yaml
```

### 3. Run Inference

```bash
python src/inference.py --model_path ./outputs/qwen-alignment-model --turkish "Kitap okudum." --english "I read a book."
```

### 4. Build Dictionary

```bash
python batch_alignment_inference.py \
    --model_path ./outputs/qwen-alignment-model \
    --max_pairs 1000 \
    --batch_size 10
```

## Pipeline Script

Run the complete pipeline:

```bash
python run_pipeline.py --step all --max_examples 2000 --config configs/config_example.yaml
```

Options:
- `--step`: Choose `data`, `train`, `inference`, or `all`
- `--max_examples`: Number of training examples (default: 2000)
- `--config`: Path to configuration file

## Training Data Format

The system creates prompts in this format:

```
### Instruction:
Generate word-level alignments between these two sentences, matching each word in the source language to its corresponding word(s) in the target language. Use the lemma (dictionary form) of each word. Format the output as 'source_lemma - target_lemma' pairs, one per line.

### Source sentence (Turkish):
Kıbrıs Rum Kesimi'nin Larnaka kentinde yapılan sokak çalışmaları sırasında 2500 yıllık olduğuna inanılan antik bir mezar bulundu

### Target sentence (English):
An ancient tomb, believed to be 2,500 years old, was discovered during street works in Larnaca, Cyprus.

### Word alignments:
<alignments>
antik - ancient
mezar - tomb
2500 - 2,500
yıllık - year
inan - believe
bul - discover
sokak - street
çalışma - work
Larnaka - Larnaca
Kıbrıs - Cyprus
</alignments>
```

## Configuration

Edit `configs/config_example.yaml` to customize:
- Training parameters (epochs, batch size, learning rate)
- LoRA settings (rank, alpha, dropout)
- Data filtering criteria
- Model hyperparameters

## Inference Scripts

### Simple Inference

```bash
python src/inference.py \
    --model_path ./outputs/qwen-alignment-model \
    --turkish "Turkish sentence" \
    --english "English sentence"
```

### Batch Processing

```bash
python batch_alignment_inference.py \
    --model_path ./outputs/qwen-alignment-model \
    --turkish_file parallel_data_40k/test_transcription.txt \
    --english_file parallel_data_40k/test_translation.txt \
    --max_pairs 1000 \
    --batch_size 10 \
    --output_dir batch_alignment_outputs
```

### Demo Mode

Test with sample data using the base Qwen model:

```bash
python demo_alignment_inference.py --num_pairs 10
```

## Model Architecture

- **Base model**: Qwen2.5-7B-Instruct
- **Finetuning method**: LoRA (Low-Rank Adaptation)
- **Training objective**: Instruction following for alignment generation
- **Output format**: Structured alignments with `<alignments>` tags

## Features

- Automatic prompt generation from parallel data and dictionary
- Lemmatization support for Turkish inflected forms
- Quality filtering of training examples
- Optimal dataset sizing (2,000 examples recommended)
- PEFT/LoRA integration for efficient training
- Structured output format with alignment tags for easy extraction
- Inference pipeline with format validation
- Configurable training parameters

## Output Formats

The inference scripts generate multiple output formats:

1. **JSON**: Complete dictionary with frequency counts
2. **Simple Text**: Turkish → English mappings
3. **CSV**: Detailed format with frequencies and confidence scores
4. **Statistics**: Comprehensive quality metrics

## Memory Requirements

- **GPU Memory**: ~16-24GB VRAM (with 4-bit quantization)
- **System RAM**: ~32GB recommended
- **Storage**: ~50GB for model weights and training data

## Recommended Training Settings

| Dataset Size | Epochs | Expected Quality | Training Time |
|-------------|--------|------------------|---------------|
| 1,000       | 3      | Good for testing | 1-2 hours     |
| 2,000       | 2      | **Recommended**  | 2-4 hours     |
| 5,000       | 2      | High quality     | 3-6 hours     |

## Troubleshooting

### Out of Memory
- Reduce batch size in config
- Increase gradient accumulation steps
- Enable more aggressive quantization

### Model Loading Error
- Check model path exists
- Ensure PEFT model is in `model_path/peft_model/`

### Empty Alignments
- Check model quality
- Verify prompt format
- Review input sentence quality
