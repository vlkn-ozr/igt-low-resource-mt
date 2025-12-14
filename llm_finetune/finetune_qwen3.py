import os
import argparse
from typing import List, Dict, Any, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 on gloss-to-translation data")
    parser.add_argument("--dataset_path", type=str, default="processed/gloss_translation_dataset.jsonl",
                        help="Path to the processed dataset file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="Qwen model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--use_lora", action="store_true", default=True, 
                        help="Whether to use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, 
                        help="Rank for LoRA adaptation")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="Alpha for LoRA adaptation")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="Dropout for LoRA adaptation")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Whether to enable thinking mode (disabled by default)")
    return parser.parse_args()

def load_and_prepare_data(dataset_path):
    dataset = load_dataset("json", data_files=dataset_path)
    
    def preprocess_function(examples):
        tokenized_inputs = []
        tokenized_targets = []
        
        for messages in examples["messages"]:
            user_message = next(msg["content"] for msg in messages if msg["role"] == "user")
            assistant_message = next(msg["content"] for msg in messages if msg["role"] == "assistant")
            
            tokenized_inputs.append(user_message)
            tokenized_targets.append(assistant_message)
        
        return {"input": tokenized_inputs, "target": tokenized_targets}
    
    processed_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )
    
    return processed_dataset["train"]

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
    
    train_dataset = load_and_prepare_data(args.dataset_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.use_lora:
        print("Using LoRA for fine-tuning")
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            use_cache=False,
            attn_implementation="eager",
        )
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj"],
        )
        
        model = get_peft_model(model, lora_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            use_cache=False,
            attn_implementation="eager",
        )
    
    def tokenize_function(examples):
        formatted_texts = []
        for input_text, target_text in zip(examples["input"], examples["target"]):
            messages = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": target_text}
            ]
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=args.enable_thinking
            )
            formatted_texts.append(formatted_text)
        
        tokenized = tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    tokenized_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        disable_tqdm=False,
        local_rank=-1,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        ddp_broadcast_buffers=False,
        use_cpu=False,
        use_ipex=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    print("Starting training...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if not args.enable_thinking:
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "do_sample": True,
            "max_new_tokens": 100
        }
        with open(os.path.join(args.output_dir, "generation_config.json"), "w") as f:
            import json
            json.dump(generation_config, f, indent=2)
        print(f"Saved generation config with non-thinking mode parameters")
    
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 