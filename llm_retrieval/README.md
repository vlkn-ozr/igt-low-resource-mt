# LLM-Based Translation of Turkish Interlinear Glossed Text

This project evaluates different prompting approaches for translating Turkish interlinear glossed text to English using large language models.

## Supported Models

- Meta's Llama 3.1 8B Instruct
- Qwen 2.5 7B Instruct
- Qwen 3 8B

## Prompt Approaches

### 1. Zero-Shot Prompt
Provides general instructions without examples. See `zero_shot_prompt.txt`.

### 2. Few-Shot Prompt
Provides examples of glossed text and their translations. See `few_shot_prompt.txt`.

### 3. Advanced Few-Shot Prompt
Includes examples with linguistic feature analysis. See `advanced_few_shot_prompt.txt`.

### 4. ChrF++ Few-Shot Retrieval
Uses chrF++ scores to find the most similar examples for few-shot prompting. Dynamically selects the most relevant examples from the training set for each input.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Accept model licenses on Hugging Face:
   - Visit [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   - Visit [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
   - Visit [Qwen 3 8B](https://huggingface.co/Qwen/Qwen3-8B)
   - Click "Accept" on each license agreement

3. Set up Hugging Face authentication:
```bash
export HF_TOKEN=your_huggingface_token
huggingface-cli login --token your_huggingface_token
```

## Usage

### Basic Usage

```bash
python run_tests.py --model llama
python run_tests.py --model qwen
python run_tests.py --model qwen3
python run_tests.py --model all
```

### Template Selection

```bash
python run_tests.py --templates zero_shot
python run_tests.py --templates few_shot
python run_tests.py --templates advanced_few_shot
python run_tests.py --templates zero_shot few_shot advanced_few_shot
```

### ChrF++ Retrieval

```bash
python run_tests.py --model all --use-chrf
python run_tests.py --model all --only-chrf --templates advanced_few_shot
```

### Custom Data Files

```bash
python run_tests.py \
    --gloss-file data/test_gloss.txt \
    --translation-file data/test_trans.txt \
    --train-gloss-file data/train_gloss.txt \
    --train-translation-file data/train_trans.txt
```

### Additional Options

```bash
python run_tests.py --force-cpu
python run_tests.py --limit 10
python run_tests.py --batch-size 2
python run_tests.py --prompt-dir my_prompts --results-dir my_results
```

## Data Format

All files should have one example per line, with corresponding lines between gloss and translation files:

**Gloss file:**
```
EU-NOM-PROP-ABBR and-CNJ-COO Serbia-NOM-PROP a-DET-INDEF agreement-NOM on-V-PST-3S
Turkey-NOM-PROP EU-DAT-PROP-ABBR membership-NOM for-POSTP apply-V-PST-3S
```

**Translation file:**
```
The EU and Serbia signed an agreement.
Turkey applied for EU membership.
```

## Output Files

Results are saved in the `results/` directory:
- `translation_results_[model].json`: Final results with all translations
- `translation_results_[model]_partial.json`: Intermediate results
- `prompts_log_[model].json`: Log of all generated prompts

Prompts are saved in the `prompts/` directory with pattern: `[model]_[template]_[input_preview]_[timestamp].txt`

## Evaluation Scripts

### Extract and Restructure Results

```bash
python combined_extract_restructure.py translation_results_llama.json
```

### Calculate Metrics

```bash
python combined_metrics_calculator.py restructured_translation_results_llama.json
```

This calculates BLEU, chrF++, and XCOMET scores.

### Calculate XCOMET Only

```bash
python calculate_xcomet.py translation_results_llama.json
```

## Single Example Testing

Test with a single example:

```bash
python chrf_prompt_builder.py "EU-NOM-PROP-ABBR and-CNJ-COO Serbia-NOM-PROP a-DET-INDEF agreement-NOM on-V-PST-3S" --template advanced
```

## ChrF++ Retrieval

The `--use-chrf` option enables chrF++ retrieval for few-shot examples:

1. Computes chrF++ similarity between input and all training examples
2. Selects top 3 most similar examples
3. Uses these in the few-shot prompt

The `--only-chrf` option runs only zero-shot and chrF++ tests, skipping standard few-shot approaches.

## Requirements

- Python 3.8+
- GPU recommended (16GB+ VRAM) for larger models
- Sufficient disk space for model downloads (~15GB per model)
