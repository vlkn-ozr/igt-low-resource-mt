import os
import json
import sys
import argparse
import gc
import datetime
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from chrf_retriever import ChrfRetriever

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_output_directories(prompt_dir="prompts", results_dir="results"):
    """Create directories for outputs."""
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return prompt_dir, results_dir

def gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Free memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB reserved, {torch.cuda.memory_allocated(i) / 1e9:.2f} GB allocated")
    else:
        print("CUDA available: No")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        print("Clearing GPU memory cache...")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory after clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

def prepare_offload_folder():
    """Create offload folder if it doesn't exist."""
    offload_dir = "offload_folder"
    if not os.path.exists(offload_dir):
        print(f"Creating offload folder: {offload_dir}")
        os.makedirs(offload_dir)
    return offload_dir

gpu_info()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
clear_gpu_memory()

prepare_offload_folder()

MODEL_OPTIONS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
}

def load_model(model_name, force_cpu=False):
    """Load model and tokenizer with CPU fallback for OOM cases."""
    print(f"Loading tokenizer from {model_name}...")
    try:
        if "Qwen3" in model_name:
            print("Loading Qwen3 model with specific configuration...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            if torch.cuda.is_available() and not force_cpu:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except Exception as e:
                    print(f"GPU loading failed for Qwen3: {e}")
                    print("Falling back to CPU...")
                    force_cpu = True
            
            if force_cpu:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            if torch.cuda.is_available() and not force_cpu:
                try:
                    print("GPU is available, attempting to load model with optimizations...")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    loading_approaches = [
                        {"name": "4-bit quantization", "function": load_model_4bit},
                        {"name": "8-bit quantization", "function": load_model_8bit},
                        {"name": "16-bit with offload", "function": load_model_16bit_offload},
                        {"name": "CPU mapping with device_map='auto'", "function": load_model_auto_map},
                    ]
                    
                    last_error = None
                    for approach in loading_approaches:
                        try:
                            print(f"Trying {approach['name']} approach...")
                            model = approach["function"](model_name)
                            if model is not None:
                                print(f"Successfully loaded model with {approach['name']}")
                                break
                        except Exception as e:
                            print(f"Failed with {approach['name']}: {e}")
                            last_error = e
                            torch.cuda.empty_cache()
                            gc.collect()
                            time.sleep(1)
                    else:
                        print("All GPU approaches failed, falling back to CPU...")
                        force_cpu = True
                        if last_error:
                            print(f"Last error was: {last_error}")
                except Exception as e:
                    print(f"GPU loading failed with error: {e}")
                    print("Falling back to CPU...")
                    force_cpu = True
            
            if force_cpu:
                print("Loading model on CPU (this will be slower but more reliable)...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="cpu",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    print(f"CPU loading failed with standard approach: {e}")
                    print("Trying safetensors CPU loading as last resort...")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="cpu",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            use_safetensors=True,
                            torch_dtype=torch.float32
                        )
                    except Exception as e:
                        print(f"All loading approaches failed: {e}")
                        return None, None
        
        model_device = next(model.parameters()).device
        print(f"Model loaded on: {model_device}")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def load_model_4bit(model_name):
    """Try to load model with 4-bit quantization."""
    try:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    except ImportError:
        print("4-bit quantization not available (bitsandbytes not installed)")
        return None
    except Exception as e:
        print(f"4-bit loading failed: {e}")
        return None

def load_model_8bit(model_name):
    """Try to load model with 8-bit quantization."""
    try:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    except ImportError:
        print("8-bit quantization not available (bitsandbytes not installed)")
        return None
    except Exception as e:
        print(f"8-bit loading failed: {e}")
        return None

def load_model_16bit_offload(model_name):
    """Try to load model with 16-bit and CPU offloading."""
    try:
        offload_folder = prepare_offload_folder()
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="balanced",  # Try balanced instead of auto
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=offload_folder,
            offload_state_dict=True
        )
    except Exception as e:
        print(f"16-bit with offload failed: {e}")
        return None

def load_model_auto_map(model_name):
    """Try to load model with automatic device mapping."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Auto device mapping failed: {e}")
        return None

def read_prompt_template(filename):
    """Read prompt template from file."""
    with open(filename, 'r') as file:
        return file.read()

def extract_template_parts(template):
    """Extract the parts of the template before and after the examples section."""
    lines = template.split('\n')
    
    before_examples = []
    after_examples = []
    in_examples_section = False
    example_section_end = False
    
    example_start_markers = [
        "Here are some examples",
        "Here are examples",
        "Example 1:",
        "Examples:"
    ]
    
    example_end_markers = [
        "Now translate",
        "translate the following",
        "Translate the following"
    ]
    
    for i, line in enumerate(lines):
        if not in_examples_section and any(marker in line for marker in example_start_markers):
            in_examples_section = True
            before_examples.append(line)
            continue
        
        if in_examples_section and not example_section_end and any(marker in line for marker in example_end_markers):
            example_section_end = True
            after_examples.append(line)
            continue
            
        if "<translation>" in line or "</translation>" in line:
            if not in_examples_section:
                before_examples.append(line)
            elif example_section_end:
                after_examples.append(line)
            continue
        
        if not in_examples_section:
            before_examples.append(line)
        elif example_section_end:
            after_examples.append(line)
    
    if not in_examples_section or not example_section_end:
        print("Using fallback template parsing method")
        before_examples = []
        after_examples = []
        input_placeholder_found = False
        
        for line in lines:
            if "{input_text}" in line:
                input_placeholder_found = True
                after_examples.append(line)
            elif not input_placeholder_found:
                before_examples.append(line)
            else:
                after_examples.append(line)
    
    return "\n".join(before_examples), "\n".join(after_examples)

def build_examples_section(examples):
    """Build the examples section of the prompt."""
    examples_text = "\n\n"
    for i, example in enumerate(examples):
        examples_text += f"Example {i+1}:\n"
        examples_text += f"Input: {example['input']}\n"
        examples_text += f"<translation>{example['expected']}</translation>\n\n"
    
    return examples_text

def prepare_custom_prompt(template, input_text, examples):
    """Prepare a custom prompt by replacing the built-in examples with custom ones."""
    try:
        before_examples, after_examples = extract_template_parts(template)
        
        examples_text = build_examples_section(examples)
        
        prompt = before_examples + examples_text + after_examples
        
        prompt = prompt.replace("{input_text}", input_text)
        
        return prompt
    except Exception as e:
        print(f"Warning: Exception in custom prompt preparation: {e}")
        print("Using simple template replacement")
        
        return template.replace("{input_text}", input_text)

def generate_prompt_filename(input_text, prompt_type, model_name, prompt_dir="prompts"):
    """Generate a filename for saving the prompt."""
    clean_input = re.sub(r'[^\w\s-]', '', input_text[:30])
    words = clean_input.split()
    filename_base = "_".join(words[:3])
    
    if not filename_base:
        filename_base = "prompt"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(prompt_dir, f"{model_name}_{prompt_type}_{filename_base}_{timestamp}.txt")

def load_test_examples(gloss_file='eval_gloss.txt', translation_file='eval_translation.txt'):
    """Load test examples from gloss and translation files."""
    examples = []
    
    # Read glosses
    with open(gloss_file, 'r') as f:
        glosses = [line.strip() for line in f.readlines() if line.strip()]
    
    # Read translations
    with open(translation_file, 'r') as f:
        translations = [line.strip() for line in f.readlines() if line.strip()]
    
    # Pair glosses with translations
    for gloss, translation in zip(glosses, translations):
        examples.append({
            "input": gloss,
            "expected": translation
        })
    
    return examples

def load_training_examples(train_gloss_file='train_gloss.txt', train_translation_file='train_translation.txt'):
    """Load training examples from gloss and translation files for few-shot selection."""
    examples = []
    
    # Read glosses
    with open(train_gloss_file, 'r') as f:
        glosses = [line.strip() for line in f.readlines() if line.strip()]
    
    # Read translations
    with open(train_translation_file, 'r') as f:
        translations = [line.strip() for line in f.readlines() if line.strip()]
    
    # Pair glosses with translations
    for gloss, translation in zip(glosses, translations):
        examples.append({
            "input": gloss,
            "expected": translation
        })
    
    # Save examples to a file for the retriever
    with open('examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2)
    
    return examples

def get_few_shot_examples(current_example, training_examples, n=3, use_chrf=False):
    """Get few-shot examples for prompting from training data."""
    if not use_chrf:
        import random
        if len(training_examples) <= n:
            return training_examples, []
        selected = random.sample(training_examples, n)
        return selected, []
    else:
        print("Using chrF++ retrieval for few-shot examples from training data")
        retriever = ChrfRetriever('examples.json', n_examples=n+5)
        similar_examples = retriever.retrieve(current_example['input'])
        
        similar_examples = [ex for ex in similar_examples if ex['input'].lower() != current_example['input'].lower()]
        
        similarity_scores = []
        for example in similar_examples:
            score = retriever._compute_chrf_score(current_example['input'], example['input'])
            similarity_scores.append({
                'input': example['input'],
                'score': score
            })
        
        if len(similar_examples) < n:
            print("Need more examples after filtering duplicates, finding most similar from all training examples...")
            
            scored_candidates = []
            for ex in training_examples:
                if ex in similar_examples or ex['input'].lower() == current_example['input'].lower():
                    continue
                    
                score = retriever._compute_chrf_score(current_example['input'], ex['input'])
                scored_candidates.append((score, ex))
            
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            remaining_needed = n - len(similar_examples)
            additional_examples = [ex for _, ex in scored_candidates[:remaining_needed]]
            
            if additional_examples:
                print(f"Adding {len(additional_examples)} additional similar examples from training data")
                similar_examples.extend(additional_examples)
                
                for example in additional_examples:
                    score = retriever._compute_chrf_score(current_example['input'], example['input'])
                    similarity_scores.append({
                        'input': example['input'],
                        'score': score
                    })
        
        similarity_scores.sort(key=lambda x: next((i for i, e in enumerate(similar_examples) if e['input'] == x['input']), 999), reverse=False)
        
        return similar_examples[:n], similarity_scores[:n]

def generate_translation(model, tokenizer, prompt, model_type):
    """Generate translation using the model."""
    model_device = next(model.parameters()).device
    
    if "Qwen3" in model_type:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model_device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
    
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        prompt_text = tokenizer.decode(prompt_tokens.input_ids[0], skip_special_tokens=True)
        generated_text = translation[len(prompt_text):]
        
        return generated_text

def test_prompts(model_option="llama", force_cpu=False, limit=None, batch_size=1, use_chrf=False, templates=["few_shot"], only_chrf=False, gloss_file='eval_gloss.txt', translation_file='eval_translation.txt', train_gloss_file='train_gloss.txt', train_translation_file='train_translation.txt', prompt_dir="prompts", results_dir="results"):
    """Test different prompt approaches."""
    model_name = MODEL_OPTIONS.get(model_option)
    if not model_name:
        print(f"Error: Unknown model option '{model_option}'")
        return False
    
    # Create output directories
    prompt_dir, results_dir = create_output_directories(prompt_dir, results_dir)
    
    print(f"Testing using model: {model_name}")
    print(f"Using test gloss file: {gloss_file}")
    print(f"Using test translation file: {translation_file}")
    print(f"Using training gloss file: {train_gloss_file}")
    print(f"Using training translation file: {train_translation_file}")
    print(f"Prompts will be saved to: {prompt_dir}")
    print(f"Results will be saved to: {results_dir}")
    model, tokenizer = load_model(model_name, force_cpu)
    
    if model is None or tokenizer is None:
        return False
    
    # Verify model is on GPU if available
    if torch.cuda.is_available():
        model_device = next(model.parameters()).device
        if not model_device.type == 'cuda':
            print(f"Warning: Model is on {model_device}, attempting to move to GPU...")
            # Move model to GPU
            model = model.to(device)
            model_device = next(model.parameters()).device
            print(f"Model is now on: {model_device}")
    
    # Load different prompt templates
    zero_shot_template = read_prompt_template('zero_shot_prompt.txt')
    standard_few_shot_template = read_prompt_template('few_shot_prompt.txt')
    advanced_few_shot_template = read_prompt_template('advanced_few_shot_prompt.txt')
    
    # Define which templates to run based on user selection
    run_zero_shot = "zero_shot" in templates
    run_few_shot = "few_shot" in templates and not only_chrf
    run_advanced_few_shot = "advanced_few_shot" in templates and not only_chrf
    
    # Load test examples and training examples separately
    test_examples = load_test_examples(gloss_file, translation_file)
    
    training_examples = []
    if train_gloss_file and train_translation_file:
        try:
            training_examples = load_training_examples(train_gloss_file, train_translation_file)
        except FileNotFoundError:
            print("Training files not found – running in zero-shot mode only")
            training_examples = []
    else:
        print("No training files provided – running in zero-shot mode only")
        training_examples = []
    
    print(f"Loaded {len(test_examples)} test examples")
    print(f"Loaded {len(training_examples)} training examples for few-shot selection")
    
    # Adjust template flags if no training examples
    if len(training_examples) == 0:
        if run_few_shot or run_advanced_few_shot or use_chrf:
            print("No training examples available – disabling few-shot and chrF++ prompts")
        run_few_shot = False
        run_advanced_few_shot = False
        use_chrf = False
    
    if limit:
        test_examples = test_examples[:limit]
        print(f"Limited to {limit} test examples")
    
    # Update results structure based on selected templates
    results = {}
    if run_zero_shot:
        results["zero_shot"] = []
    if run_few_shot:
        results["few_shot"] = []
    if run_advanced_few_shot:
        results["advanced_few_shot"] = []
    if use_chrf:
        results["chrf_few_shot"] = [] if "few_shot" in templates else None
        results["chrf_advanced_few_shot"] = [] if "advanced_few_shot" in templates else None
    
    # For saving prompt details
    prompts_log = []
    
    try:
        for i, example in enumerate(test_examples):
            print(f"\nTesting Example {i+1}/{len(test_examples)}:")
            print(f"Input: {example['input']}")
            print(f"Expected: {example['expected']}")
            
            current_prompts = {}
            
            # Test zero-shot prompt if selected
            if run_zero_shot:
                zero_shot_prompt = zero_shot_template.replace("{input_text}", example['input'])
                
                # Save zero-shot prompt
                zero_shot_filename = generate_prompt_filename(example['input'], "zero_shot", model_option, prompt_dir)
                with open(zero_shot_filename, 'w', encoding='utf-8') as f:
                    f.write(zero_shot_prompt)
                
                # Add to prompts log
                current_prompts["zero_shot"] = {
                    "prompt": zero_shot_prompt,
                    "file": zero_shot_filename
                }
                
                # Generate translation
                zero_shot_result = generate_translation(model, tokenizer, zero_shot_prompt, model_option)
                print(f"Zero-shot: {zero_shot_result}")
                
                # Save results
                results["zero_shot"].append({
                    "input": example['input'],
                    "expected": example['expected'],
                    "generated": zero_shot_result
                })
                
                # Clear memory after each operation if needed
                if torch.cuda.is_available() and batch_size == 1:
                    clear_gpu_memory()
            
            # Only run standard few-shot and advanced few-shot if not in only_chrf mode
            standard_few_shot_examples = None
            
            if run_few_shot or run_advanced_few_shot:
                # Get examples for both few-shot templates (if needed) from training data
                standard_few_shot_examples, _ = get_few_shot_examples(example, training_examples, n=3, use_chrf=False)
            
            if run_few_shot:
                # Test few-shot prompt with random examples
                few_shot_prompt = prepare_custom_prompt(standard_few_shot_template, example['input'], standard_few_shot_examples)
                
                # Save standard few-shot prompt
                few_shot_filename = generate_prompt_filename(example['input'], "few_shot", model_option, prompt_dir)
                with open(few_shot_filename, 'w', encoding='utf-8') as f:
                    f.write(few_shot_prompt)
                
                # Add to prompts log
                current_prompts["few_shot"] = {
                    "prompt": few_shot_prompt,
                    "file": few_shot_filename,
                    "examples_used": [ex['input'] for ex in standard_few_shot_examples]
                }
                
                # Generate translation
                few_shot_result = generate_translation(model, tokenizer, few_shot_prompt, model_option)
                print(f"Few-shot: {few_shot_result}")
                
                # Save results
                results["few_shot"].append({
                    "input": example['input'],
                    "expected": example['expected'],
                    "generated": few_shot_result,
                    "examples_used": [ex['input'] for ex in standard_few_shot_examples]
                })
                
                # Clear memory after each operation if needed
                if torch.cuda.is_available() and batch_size == 1:
                    clear_gpu_memory()
            
            if run_advanced_few_shot:
                # Test advanced few-shot prompt with the same examples as standard few-shot
                advanced_few_shot_prompt = prepare_custom_prompt(advanced_few_shot_template, example['input'], standard_few_shot_examples)
                
                # Save advanced few-shot prompt
                advanced_few_shot_filename = generate_prompt_filename(example['input'], "advanced_few_shot", model_option, prompt_dir)
                with open(advanced_few_shot_filename, 'w', encoding='utf-8') as f:
                    f.write(advanced_few_shot_prompt)
                
                # Add to prompts log
                current_prompts["advanced_few_shot"] = {
                    "prompt": advanced_few_shot_prompt,
                    "file": advanced_few_shot_filename
                }
                
                # Generate translation
                advanced_few_shot_result = generate_translation(model, tokenizer, advanced_few_shot_prompt, model_option)
                print(f"Advanced few-shot: {advanced_few_shot_result}")
                
                # Save results
                results["advanced_few_shot"].append({
                    "input": example['input'],
                    "expected": example['expected'],
                    "generated": advanced_few_shot_result
                })
            
            if use_chrf:
                if torch.cuda.is_available():
                    clear_gpu_memory()
                
                chrf_few_shot_examples, similarity_scores = get_few_shot_examples(example, training_examples, n=3, use_chrf=True)
                
                if "few_shot" in templates:
                    chrf_few_shot_prompt = prepare_custom_prompt(standard_few_shot_template, example['input'], chrf_few_shot_examples)
                    
                    chrf_few_shot_filename = generate_prompt_filename(example['input'], "chrf_few_shot", model_option, prompt_dir)
                    with open(chrf_few_shot_filename, 'w', encoding='utf-8') as f:
                        f.write(chrf_few_shot_prompt)
                    
                    current_prompts["chrf_few_shot"] = {
                        "prompt": chrf_few_shot_prompt,
                        "file": chrf_few_shot_filename,
                        "examples_used": [ex['input'] for ex in chrf_few_shot_examples],
                        "similarity_scores": similarity_scores,
                        "template_used": "few_shot"
                    }
                    
                    chrf_few_shot_result = generate_translation(model, tokenizer, chrf_few_shot_prompt, model_option)
                    print(f"chrF++ few-shot: {chrf_few_shot_result}")
                    
                    results["chrf_few_shot"].append({
                        "input": example['input'],
                        "expected": example['expected'],
                        "generated": chrf_few_shot_result,
                        "examples_used": [ex['input'] for ex in chrf_few_shot_examples],
                        "similarity_scores": similarity_scores,
                        "template_used": "few_shot"
                    })
                    
                    if torch.cuda.is_available() and batch_size == 1:
                        clear_gpu_memory()
                
                if "advanced_few_shot" in templates:
                    chrf_advanced_prompt = prepare_custom_prompt(advanced_few_shot_template, example['input'], chrf_few_shot_examples)
                    
                    chrf_advanced_filename = generate_prompt_filename(example['input'], "chrf_advanced_few_shot", model_option, prompt_dir)
                    with open(chrf_advanced_filename, 'w', encoding='utf-8') as f:
                        f.write(chrf_advanced_prompt)
                    
                    current_prompts["chrf_advanced_few_shot"] = {
                        "prompt": chrf_advanced_prompt,
                        "file": chrf_advanced_filename,
                        "examples_used": [ex['input'] for ex in chrf_few_shot_examples],
                        "similarity_scores": similarity_scores,
                        "template_used": "advanced_few_shot"
                    }
                    
                    chrf_advanced_result = generate_translation(model, tokenizer, chrf_advanced_prompt, model_option)
                    print(f"chrF++ advanced few-shot: {chrf_advanced_result}")
                    
                    results["chrf_advanced_few_shot"].append({
                        "input": example['input'],
                        "expected": example['expected'],
                        "generated": chrf_advanced_result,
                        "examples_used": [ex['input'] for ex in chrf_few_shot_examples],
                        "similarity_scores": similarity_scores,
                        "template_used": "advanced_few_shot"
                    })
            
            prompts_log.append({
                "input": example['input'],
                "expected": example['expected'],
                "prompts": current_prompts
            })
            
            prompts_log_file = f"prompts_log_{model_option}.json"
            with open(os.path.join(results_dir, prompts_log_file), 'w') as file:
                json.dump(prompts_log, file, indent=2)
            
            if (i + 1) % max(1, batch_size) == 0 or (i + 1) == len(test_examples):
                intermediate_file = f"translation_results_{model_option}_partial.json"
                with open(os.path.join(results_dir, intermediate_file), 'w') as file:
                    json.dump(results, file, indent=2)
                print(f"Intermediate results saved to {os.path.join(results_dir, intermediate_file)}")
                
                if torch.cuda.is_available():
                    clear_gpu_memory()
    
    except Exception as e:
        print(f"Error during translation: {e}")
        # Save partial results
        results_file = f"translation_results_{model_option}_error.json"
        with open(os.path.join(results_dir, results_file), 'w') as file:
            json.dump(results, file, indent=2)
        print(f"Partial results saved to {os.path.join(results_dir, results_file)}")
        
        # Clean up resources
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return False
    
    # Save final results to file
    results_file = f"translation_results_{model_option}.json"
    with open(os.path.join(results_dir, results_file), 'w') as file:
        json.dump(results, file, indent=2)
    
    print(f"\nResults saved to {os.path.join(results_dir, results_file)}")
    print(f"Prompts log saved to {os.path.join(results_dir, prompts_log_file)}")
    
    # Clean up resources
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test translation prompts')
    parser.add_argument('--model', choices=list(MODEL_OPTIONS.keys()), default='llama')
    parser.add_argument('--all', action='store_true', help='Test all models')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage regardless of GPU availability')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of test examples to process')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of examples to process before saving intermediate results')
    parser.add_argument('--use-chrf', action='store_true', help='Use chrF++ retrieval for few-shot examples')
    parser.add_argument('--templates', nargs='+', choices=['zero_shot', 'few_shot', 'advanced_few_shot'], default=['few_shot'], 
                       help='Select the templates for few-shot prompting')
    parser.add_argument('--only-chrf', action='store_true', 
                       help='Only run zero-shot and chrF++ tests (both standard and advanced templates)')
    parser.add_argument('--gloss-file', default='eval_gloss.txt', 
                       help='Path to the gloss file (default: eval_gloss.txt)')
    parser.add_argument('--translation-file', default='eval_translation.txt', 
                       help='Path to the translation file (default: eval_translation.txt)')
    parser.add_argument('--train-gloss-file', default='train_gloss.txt', 
                       help='Path to the training gloss file (default: train_gloss.txt)')
    parser.add_argument('--train-translation-file', default='train_translation.txt', 
                       help='Path to the training translation file (default: train_translation.txt)')
    parser.add_argument('--prompt-dir', default='prompts', 
                       help='Directory to save generated prompts (default: prompts)')
    parser.add_argument('--results-dir', default='results', 
                       help='Directory to save results (default: results)')
    args = parser.parse_args()
    
    print(f"Running with options: model={args.model}, force_cpu={args.force_cpu}, limit={args.limit}, " 
         f"batch_size={args.batch_size}, use_chrf={args.use_chrf}, templates={args.templates}, "
         f"only_chrf={args.only_chrf}, gloss_file={args.gloss_file}, translation_file={args.translation_file}, "
         f"train_gloss_file={args.train_gloss_file}, train_translation_file={args.train_translation_file}, "
         f"prompt_dir={args.prompt_dir}, results_dir={args.results_dir}")
    
    if args.only_chrf and not args.use_chrf:
        print("Warning: --only-chrf requires --use-chrf. Enabling --use-chrf automatically.")
        args.use_chrf = True
    
    if args.all:
        success = True
        for model_key in MODEL_OPTIONS:
            print(f"\n=== Testing {MODEL_OPTIONS[model_key]} ===")
            if not test_prompts(model_key, force_cpu=args.force_cpu, limit=args.limit, 
                              batch_size=args.batch_size, use_chrf=args.use_chrf, 
                              templates=args.templates, only_chrf=args.only_chrf,
                              gloss_file=args.gloss_file, translation_file=args.translation_file,
                              train_gloss_file=args.train_gloss_file, train_translation_file=args.train_translation_file,
                              prompt_dir=args.prompt_dir, results_dir=args.results_dir):
                print(f"Tests for {model_key} failed!")
                success = False
        
        if not success:
            sys.exit(1)
    else:
        if not test_prompts(args.model, force_cpu=args.force_cpu, limit=args.limit, 
                          batch_size=args.batch_size, use_chrf=args.use_chrf, 
                          templates=args.templates, only_chrf=args.only_chrf,
                          gloss_file=args.gloss_file, translation_file=args.translation_file,
                          train_gloss_file=args.train_gloss_file, train_translation_file=args.train_translation_file,
                          prompt_dir=args.prompt_dir, results_dir=args.results_dir):
            sys.exit(1)

if __name__ == "__main__":
    main() 