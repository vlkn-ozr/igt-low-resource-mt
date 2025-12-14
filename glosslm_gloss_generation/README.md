# GlossLM Model Evaluation

This directory contains scripts for evaluating the GlossLM model for zero-shot gloss generation.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: For CUDA support, you may need to install PyTorch with CUDA separately:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Evaluate GlossLM Model

Run evaluation on test data:

```bash
python eval_glosslm.py
```

The script expects a `test_set` directory with language-specific subdirectories. Each subdirectory should contain:
- `test.glosses.txt`: Reference glosses
- `test.transcription.txt`: Input transcriptions
- `test.translation.txt`: Translations

The script will:
- Load the GlossLM model from HuggingFace (`lecslab/glosslm`)
- Generate glosses for each language in the test set
- Save results to `translation_results_<lang_code>.json` files

### 2. Extract Predictions

Extract generated predictions from JSON results:

```bash
python extract_predictions.py
```

This script:
- Finds all `translation_results_*.json` files
- Extracts generated glosses
- Saves them to `test_set/<lang_code>/predicted.glosses.txt`

## Directory Structure

```
glosslm_model/
├── eval_glosslm.py          # Main evaluation script
├── extract_predictions.py   # Extract predictions from results
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── test_set/               # Test data directory (to be created)
    ├── usp/                # Language-specific folders
    ├── git/
    └── ...
```

## Supported Languages

The script supports the following languages (by glottocode):
- `usp`: Uspanteco
- `git`: Gitxsan
- `lez`: Lezgian
- `ddo`: Tsez
- `ntu`: Natügu

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "zero_shot": [
    {
      "input": "transcription text",
      "expected": "reference gloss",
      "generated": "model prediction"
    },
    ...
  ]
}
```

## Notes

- The model uses ByT5 tokenizer and T5 architecture
- Evaluation runs on GPU if available, otherwise CPU
- Results are saved incrementally during evaluation
- The script processes all language folders in the test_set directory

