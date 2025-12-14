# Qwen Fine-tuning for Linguistic Gloss Translation

This repository contains scripts for fine-tuning Qwen models (Qwen2.5-7B-Instruct and Qwen3-8B) to translate linguistic glosses to natural English using LoRA (Low-Rank Adaptation).

## Dataset Format

The dataset consists of two files:
- `data/gloss.txt`: Each line contains a linguistic gloss annotation
- `data/translation.txt`: Each line contains the corresponding English translation

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Prepare the Dataset

Process the raw data into the format required for fine-tuning:

```bash
python prepare_dataset.py
```

This creates a processed dataset file at `processed/gloss_translation_dataset.jsonl`.

## Fine-tuning

### Basic Usage

```bash
# For Qwen2.5-7B-Instruct
python finetune_qwen25.py

# For Qwen3-8B
python finetune_qwen3.py
```

### Customization

```bash
python finetune_qwen25.py \
  --output_dir ./output \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lora_r 8 \
  --lora_alpha 16
```

### Qwen3 Thinking Mode

Thinking mode is only supported for Qwen3-8B and is disabled by default (not recommended for translation tasks):

```bash
python finetune_qwen3.py --enable_thinking
```

## Using the Fine-tuned Model

### Understanding Adapter Models

This project uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters:
- Only adapter parameters are trained and saved (much smaller than the full model)
- For inference, you need both the base model and the adapter weights
- The output directory contains adapter weights, not a full model

### Code Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load the adapter weights
model = PeftModel.from_pretrained(
    base_model,
    "./output",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Example gloss to translate
gloss = "spending-V-VN:INF-NOM-PL 1,6-NUM-ARA-PERC increase-V-PST-3S"

# Format messages and apply chat template
messages = [{"role": "user", "content": f"Translate the following linguistic gloss to English: {gloss}"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

# Generate the translation
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        do_sample=True
    )

# Extract the assistant's response
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = generated_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
print(f"Gloss: {gloss}")
print(f"Translation: {response}")
```

## Testing

### Run Tests

```bash
python run_tests.py --model finetuned --custom-model-path ./output
```

### Testing Options

- `--model`: Choose the model (finetuned_qwen, finetuned_qwen3, or finetuned)
- `--template`: Prompt template (zero_shot, few_shot, advanced_few_shot)
- `--use-chrf`: Enable chrF++ retrieval for few-shot examples
- `--only-chrf`: Run only zero-shot and chrF++ tests
- `--limit`: Limit number of test examples
- `--custom-model-path`: Path to adapter model (required for finetuned models)

## Uploading to Hugging Face

Upload a fine-tuned adapter model to Hugging Face Hub:

```bash
python upload_to_huggingface.py \
  --model-path ./output \
  --repo-id username/model-name
```

Options:
- `--private`: Make the repository private
- `--include-checkpoints`: Include training checkpoints
- `--token`: Hugging Face token (or set HF_TOKEN environment variable)

Before running, authenticate:
```bash
huggingface-cli login
# or set HF_TOKEN environment variable
```

## Models

### Qwen2.5-7B-Instruct
- 7.61 billion parameters
- Context length: 131,072 tokens
- Strong multilingual capabilities

### Qwen3-8B
- 8.2 billion parameters
- Support for thinking and non-thinking modes
- Context length: 32,768 tokens (expandable to 131,072 with YaRN)
- Advanced reasoning capabilities

For translation tasks, non-thinking mode is recommended for more efficient processing.

## Notes

- LoRA is used for efficient fine-tuning, requiring less GPU memory
- Thinking mode is only supported for Qwen3-8B and disabled by default
- Recommended generation parameters: Temperature=0.7, TopP=0.8, TopK=20
- When using adapter models, always ensure you have both adapter weights and the base model
