import os
import argparse
from typing import List, Dict, Any, Tuple
import json

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
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class CheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print(f"\n💾 Checkpoint saved at step {state.global_step}")
        print(f"   Checkpoint directory: {args.output_dir}")
        print(f"   Current epoch: {state.epoch:.2f}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n📊 Epoch {state.epoch:.2f} completed")
        print(f"   Global step: {state.global_step}")
        print(f"   Training loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 on gloss-to-translation data")
    parser.add_argument("--dataset_path", type=str, default="processed/gloss_translation_dataset.jsonl",
                        help="Path to the processed dataset file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Qwen model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--use_lora", action="store_true", default=True, 
                        help="Whether to use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=4,
                        help="Rank for LoRA adaptation")
    parser.add_argument("--lora_alpha", type=int, default=8,
                        help="Alpha for LoRA adaptation")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout for LoRA adaptation")
    parser.add_argument("--max_seq_length", type=int, default=64,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints (defaults to output_dir)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--save_total_limit", type=int, default=7,
                        help="Maximum number of checkpoints to keep (default: 5)")
    return parser.parse_args()

def load_and_prepare_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset from: {os.path.abspath(dataset_path)}")
    
    dataset = load_dataset("json", data_files=dataset_path)
    
    print(f"Dataset sample: {dataset['train'][0]}")
    
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                user_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "user"), None)
                assistant_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "assistant"), None)
                if user_msg and assistant_msg:
                    examples.append({"input": user_msg, "target": assistant_msg})
            except Exception as e:
                print(f"Error parsing line: {e}")
    
    if examples:
        from datasets import Dataset
        processed_dataset = Dataset.from_list(examples)
        print(f"Created dataset from direct file parsing with {len(processed_dataset)} examples")
    else:
        def preprocess_function(examples):
            inputs = []
            targets = []
            
            for messages in examples["messages"]:
                try:
                    user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
                    assistant_msg = next((msg["content"] for msg in messages if msg["role"] == "assistant"), None)
                    
                    if user_msg and assistant_msg:
                        inputs.append(user_msg)
                        targets.append(assistant_msg)
                except Exception as e:
                    print(f"Error processing message: {e}")
            
            return {"input": inputs, "target": targets}
        
        processed_dataset = dataset.map(
            preprocess_function, 
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
    print(f"Original dataset size: {len(dataset['train'])}")
    print(f"Processed dataset size: {len(processed_dataset)}")
    
    if len(processed_dataset) == 0:
        raise ValueError("No valid examples found in the dataset.")
    
    return processed_dataset["train"] if "train" in processed_dataset else processed_dataset

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created/verified: {os.path.abspath(args.output_dir)}")
    
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
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            quantization_config=quantization_config,
            use_cache=True,
            attn_implementation="eager"
        )
        
        model = prepare_model_for_kbit_training(model)
        
        module_names = set()
        for name, module in model.named_modules():
            if any(keyword in name for keyword in ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down", "lm_head", "w1", "w2", "w3"]):
                module_names.add(name.split(".")[-1])
        
        print(f"\nAvailable modules for LoRA targeting: {sorted(list(module_names))}")
        
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            init_lora_weights="gaussian",
            modules_to_save=[]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            use_cache=True,
            attn_implementation="eager"
        )
    
    def tokenize_function(examples):
        user_prefix = "<|im_start|>user\n"
        assistant_prefix = "<|im_start|>assistant\n"
        im_end = "<|im_end|>"
        
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        for input_text, target_text in zip(examples["input"], examples["target"]):
            try:
                conversation = f"{user_prefix}{input_text}{im_end}\n{assistant_prefix}{target_text}{im_end}"
                
                encoded = tokenizer.encode(conversation)
                
                encoded = [token for token in encoded if 0 <= token < tokenizer.vocab_size]
                
                assistant_token_ids = tokenizer.encode(f"{assistant_prefix}")
                assistant_token_ids = [token for token in assistant_token_ids if 0 <= token < tokenizer.vocab_size]
                
                assistant_start_idx = len(encoded)
                for i in range(len(encoded) - len(assistant_token_ids)):
                    if encoded[i:i+len(assistant_token_ids)] == assistant_token_ids:
                        assistant_start_idx = i
                        break
                else:
                    user_tokens = tokenizer.encode(f"{user_prefix}{input_text}{im_end}\n")
                    user_tokens = [token for token in user_tokens if 0 <= token < tokenizer.vocab_size]
                    assistant_start_idx = len(user_tokens)
                
                labels = [-100] * assistant_start_idx + encoded[assistant_start_idx:]
                
                if len(encoded) > args.max_seq_length:
                    encoded = encoded[:args.max_seq_length]
                    labels = labels[:args.max_seq_length]
                
                attention_mask = [1] * len(encoded)
                padding_length = args.max_seq_length - len(encoded)
                
                if padding_length > 0:
                    encoded = encoded + [tokenizer.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                    labels = labels + [-100] * padding_length
                
                all_input_ids.append(encoded)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"Error processing example: {e}")
                print(f"Input text: {input_text[:100]}...")
                print(f"Target text: {target_text[:100]}...")
                continue

        if not all_input_ids:
            raise ValueError("No valid examples could be processed. Check your dataset and tokenizer.")

        input_example = all_input_ids[0]
        label_example = all_labels[0]
        
        label_start = 0
        while label_start < len(label_example) and label_example[label_start] == -100:
            label_start += 1
        
        print("\nDEBUG TOKENIZATION:")
        print(f"Input length: {len(input_example)}")
        print(f"Label length: {len(label_example)}")
        print(f"First 10 input tokens: {input_example[:10]}")
        print(f"First non-masked label index: {label_start}")
        print(f"Label tokens from that index: {label_example[label_start:label_start+10]}")
        
        try:
            label_tokens_to_decode = label_example[label_start:label_start+10] if label_start < len(label_example) else []
            valid_tokens = [token for token in label_tokens_to_decode if 0 <= token < tokenizer.vocab_size]
            decoded_text = tokenizer.decode(valid_tokens) if valid_tokens else ""
            print(f"Decode these tokens: {decoded_text}")
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        print(f"Total non-masked tokens: {sum(1 for x in label_example if x != -100)}")
        
        return {
            "input_ids": torch.tensor(all_input_ids),
            "attention_mask": torch.tensor(all_attention_mask),
            "labels": torch.tensor(all_labels)
        }
    
    tokenized_dataset = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    total_tokens = sum(len(example["labels"]) for example in tokenized_dataset)
    non_masked_tokens = sum(sum(1 for token in example["labels"] if token != -100) for example in tokenized_dataset)
    masked_ratio = 1 - (non_masked_tokens / total_tokens) if total_tokens > 0 else 0
    
    print(f"\nTOKENIZATION STATS:")
    print(f"Total tokens: {total_tokens}")
    print(f"Non-masked tokens (for loss): {non_masked_tokens}")
    print(f"Masked ratio: {masked_ratio:.2%}")
    
    if non_masked_tokens == 0:
        raise ValueError("No tokens are available for training (all labels are masked). Check tokenization logic.")
    
    # Set checkpoint directory
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else args.output_dir
    
    # Define training arguments with minimal configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,  # Disable weight decay
        warmup_ratio=0.2,  # Longer warmup
        logging_steps=10,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,  # Keep last N checkpoints
        save_steps=args.save_steps,  # Save every N steps
        fp16=False,  # Disable fp16
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        disable_tqdm=False,
        
        # Explicitly disable all distributed training
        local_rank=-1,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        ddp_broadcast_buffers=False,
        
        # Fallback to GPU training when distributed fails
        use_cpu=False,
        use_ipex=False,
        
        # Add gradient clipping (lower for more stability)
        max_grad_norm=0.5,
        
        # Debug mode
        debug=["underflow_overflow"],
        
        # Optimizer settings for stability
        optim="adamw_torch",  # Use standard AdamW
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8
    )
    
    # Print training arguments for debugging
    print(f"\nTRAINING ARGUMENTS:")
    print(f"Output directory: {training_args.output_dir}")
    print(f"Save strategy: {training_args.save_strategy}")
    print(f"Save total limit: {training_args.save_total_limit}")
    print(f"Save steps: {training_args.save_steps}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    
    # Calculate approximate steps per epoch for reference
    total_steps_per_epoch = len(tokenized_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    print(f"Approximate steps per epoch: {total_steps_per_epoch}")
    print(f"Checkpoints will be saved every {args.save_steps} steps (roughly every {args.save_steps/total_steps_per_epoch:.1f} epochs)")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[CheckpointCallback()],
    )
    
    # Check if we should resume from checkpoint
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        if not os.path.exists(args.resume_from_checkpoint):
            raise ValueError(f"Checkpoint directory not found: {args.resume_from_checkpoint}")
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Check for saved checkpoints
    print(f"\n🔍 Checking for saved checkpoints in: {args.output_dir}")
    if os.path.exists(args.output_dir):
        checkpoint_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            print(f"Found {len(checkpoint_dirs)} checkpoint directories:")
            for checkpoint_dir in sorted(checkpoint_dirs):
                checkpoint_path = os.path.join(args.output_dir, checkpoint_dir)
                print(f"  - {checkpoint_dir}")
                # Check if it contains model files
                if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")):
                    print(f"    ✓ Contains model files")
                else:
                    print(f"    ⚠ No model files found")
        else:
            print("⚠ No checkpoint directories found!")
    
    # Save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save generation config with recommended parameters
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
        "max_new_tokens": 100
    }
    with open(os.path.join(args.output_dir, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)
    print(f"Saved generation config with recommended parameters")
    
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 