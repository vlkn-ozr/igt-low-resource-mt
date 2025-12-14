#!/usr/bin/env python3
"""
Trainer for Turkish-English word alignment using Qwen2.5-7B-Instruct with PEFT/LoRA.
"""

import os
import json
import yaml
import argparse
from typing import Dict, List

# Set environment variables to disable distributed training if not already set
if 'WORLD_SIZE' not in os.environ:
    os.environ['WORLD_SIZE'] = '1'
if 'RANK' not in os.environ:
    os.environ['RANK'] = '0'
if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = '0'
if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = 'localhost'
if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '12355'

# Disable tensor parallelism
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb

@dataclass
class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling that handles padding properly.
    """
    tokenizer: Any
    max_length: int = 2048
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # Pad sequences
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Pad labels manually
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        
        for label_seq in labels:
            if len(label_seq) < max_len:
                # Pad with -100 (ignore index for loss calculation)
                padded_label = label_seq + [-100] * (max_len - len(label_seq))
            else:
                padded_label = label_seq[:max_len]
            padded_labels.append(padded_label)
        
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch

class AlignmentTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with quantization and PEFT."""
        print("Loading model and tokenizer...")
        
        # Quantization config for memory efficiency
        if self.config.get('load_in_4bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.get('bnb_4bit_compute_dtype', 'bfloat16')),
                bnb_4bit_use_double_quant=self.config.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=self.config.get('bnb_4bit_quant_type', 'nf4')
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=self.config.get('trust_remote_code', True),
            padding_side="right"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model without device_map to avoid tensor parallelism
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            quantization_config=bnb_config,
            trust_remote_code=self.config.get('trust_remote_code', True),
            torch_dtype=torch.bfloat16 if self.config.get('bf16', True) else torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Move to GPU manually if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        
        # Prepare model for k-bit training if using quantization
        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        if self.config.get('use_lora', True):
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.get('lora_r', 64),
                lora_alpha=self.config.get('lora_alpha', 16),
                lora_dropout=self.config.get('lora_dropout', 0.1),
                target_modules=self.config.get('lora_target_modules', [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ])
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        print("Model and tokenizer loaded successfully!")
        
    def load_and_prepare_dataset(self):
        """Load and prepare the training dataset."""
        print("Loading dataset...")
        
        # Load JSONL data
        data_path = self.config['data_path']
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found at {data_path}. Run data_processor.py first.")
        
        # Read JSONL file
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        print(f"Loaded {len(examples)} training examples")
        
        # Split into train/eval
        train_split = self.config.get('train_split', 0.9)
        split_idx = int(len(examples) * train_split)
        
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:] if split_idx < len(examples) else examples[-100:]  # At least 100 eval examples
        
        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        eval_dataset = Dataset.from_list(eval_examples)
        
        # Tokenize
        def tokenize_function(example):
            # Tokenize the full prompt (instruction + input + output)
            tokenized = self.tokenizer(
                example['text'],
                truncation=True,
                padding=False,  # We'll pad in the data collator
                max_length=self.config.get('max_seq_length', 2048),
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=False,  # Process one at a time to avoid length mismatch
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=False,  # Process one at a time to avoid length mismatch
            remove_columns=eval_dataset.column_names
        )
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        print(f"Prepared {len(train_dataset)} training and {len(eval_dataset)} evaluation examples")
        
    def setup_training_arguments(self):
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config.get('num_train_epochs', 3),
            per_device_train_batch_size=self.config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=self.config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            learning_rate=float(self.config.get('learning_rate', 2e-4)),
            weight_decay=float(self.config.get('weight_decay', 0.01)),
            warmup_ratio=float(self.config.get('warmup_ratio', 0.1)),
            lr_scheduler_type=self.config.get('lr_scheduler_type', 'cosine'),
            logging_steps=self.config.get('logging_steps', 10),
            save_steps=self.config.get('save_steps', 500),
            save_total_limit=self.config.get('save_total_limit', 3),
            eval_strategy=self.config.get('eval_strategy', 'steps'),
            eval_steps=self.config.get('eval_steps', 500),
            load_best_model_at_end=self.config.get('load_best_model_at_end', True),
            metric_for_best_model=self.config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=self.config.get('greater_is_better', False),
            optim=self.config.get('optim', 'adamw_torch'),
            adam_beta1=float(self.config.get('adam_beta1', 0.9)),
            adam_beta2=float(self.config.get('adam_beta2', 0.999)),
            adam_epsilon=float(self.config.get('adam_epsilon', 1e-8)),
            max_grad_norm=float(self.config.get('max_grad_norm', 1.0)),
            fp16=self.config.get('fp16', False),
            bf16=self.config.get('bf16', True),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=self.config.get('report_to', 'wandb'),
            run_name=self.config.get('run_name', 'qwen-turkish-english-alignment'),
            seed=self.config.get('seed', 42),
            # Disable distributed training to avoid WORLD_SIZE error
            local_rank=-1,
            ddp_find_unused_parameters=False,
        )
        
    def train(self):
        """Run the training process."""
        print("Starting training...")
        
        # Setup components
        self.setup_model_and_tokenizer()
        self.load_and_prepare_dataset()
        
        # Data collator
        data_collator = DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            max_length=self.config.get('max_seq_length', 2048)
        )
        
        # Training arguments
        training_args = self.setup_training_arguments()
        
        # Initialize wandb if specified
        if self.config.get('report_to') == 'wandb':
            wandb.init(
                project="turkish-english-alignment",
                name=self.config.get('run_name', 'qwen-alignment'),
                config=self.config
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        print(f"Training completed! Model saved to {training_args.output_dir}")
        
        # Save PEFT model separately
        if self.config.get('use_lora', True):
            peft_model_path = os.path.join(training_args.output_dir, "peft_model")
            self.model.save_pretrained(peft_model_path)
            print(f"PEFT model saved to {peft_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-7B for Turkish-English word alignment")
    parser.add_argument("--config", type=str, default="configs/qwen_alignment.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Check if training data exists
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data_path']
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        print("Please run 'python src/data_processor.py' first to generate training data.")
        return
    
    # Initialize and run trainer
    trainer = AlignmentTrainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main() 