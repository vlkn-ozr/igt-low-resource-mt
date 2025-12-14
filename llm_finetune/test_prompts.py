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

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    print("PEFT not installed. Cannot load adapter models. Install with: pip install peft")
    PEFT_AVAILABLE = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT_DIR = "prompts"
RESULTS_DIR = "results"
os.makedirs(PROMPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    "finetuned_qwen": "./finetuned_qwen25_7b_model",
    "finetuned_qwen3": "./finetuned_qwen3_8b_model",
}

def load_model(model_name, force_cpu=False, custom_model_path=None):
    """Load model and tokenizer with CPU fallback for OOM cases."""
    if custom_model_path:
        print(f"Using custom model path: {custom_model_path}")
        model_path = custom_model_path
        
        if os.path.exists(os.path.join(model_path, "adapter_config.json")) and PEFT_AVAILABLE:
            print(f"Found adapter config, loading as PEFT model from base model {model_name}")
            
            try:
                print(f"Loading base tokenizer from {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=False
                )
                
                print(f"Loading base model from {model_name}...")
                if torch.cuda.is_available() and not force_cpu:
                    try:
                        base_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                    except Exception as e:
                        print(f"GPU loading failed for base model: {e}")
                        print("Falling back to CPU...")
                        force_cpu = True
                
                if force_cpu:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="cpu",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                
                print(f"Loading adapter from {model_path}...")
                model = PeftModel.from_pretrained(
                    base_model,
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() and not force_cpu else torch.float32,
                    device_map="auto" if torch.cuda.is_available() and not force_cpu else "cpu"
                )
                
                model_device = next(model.parameters()).device
                print(f"PEFT Model loaded on: {model_device}")
                
                return model, tokenizer
                
            except Exception as e:
                print(f"Error loading PEFT model: {e}")
                return None, None
    else:
        model_path = model_name

    print(f"Loading tokenizer from {model_path}...")
    try:
        if "Qwen3" in model_path or model_name == "finetuned":
            print("Loading Qwen3 or finetuned model with specific configuration...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            if torch.cuda.is_available() and not force_cpu:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except Exception as e:
                    print(f"GPU loading failed for Qwen3 or finetuned model: {e}")
                    print("Falling back to CPU...")
                    force_cpu = True
            
            if force_cpu:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
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
    """
    Extract the parts of the template before and after the examples section.
    
    This function splits a prompt template into a part before the examples and a part after the examples,
    which allows for dynamic insertion of different examples.
    
    Args:
        template: The prompt template
        
    Returns:
        Tuple of (before_examples, after_examples)
    """
    lines = template.split('\n')
    
    # Initialize variables
    before_examples = []
    after_examples = []
    in_examples_section = False
    example_section_end = False
    
    # Common marker phrases that might indicate example sections
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
        # Check if this line contains any of the start markers
        if not in_examples_section and any(marker in line for marker in example_start_markers):
            in_examples_section = True
            before_examples.append(line)
            continue
        
        # Check if this is the first line after examples section
        if in_examples_section and not example_section_end and any(marker in line for marker in example_end_markers):
            example_section_end = True
            after_examples.append(line)
            continue
            
        # If it's an example block marker but not our detected start/end, include it in the respective section
        if "<translation>" in line or "</translation>" in line:
            if not in_examples_section:
                before_examples.append(line)
            elif example_section_end:
                after_examples.append(line)
            # Skip if inside the examples section
            continue
        
        # Add lines to the appropriate section
        if not in_examples_section:
            before_examples.append(line)
        elif example_section_end:
            after_examples.append(line)
    
    # If we couldn't detect example sections with markers, use a fallback approach
    # Find the line with "{input_text}" and divide the template there
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

def generate_prompt_filename(input_text, prompt_type, model_name):
    """Generate a filename for saving the prompt."""
    clean_input = re.sub(r'[^\w\s-]', '', input_text[:30])
    words = clean_input.split()
    filename_base = "_".join(words[:3])
    
    if not filename_base:
        filename_base = "prompt"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(PROMPT_DIR, f"{model_name}_{prompt_type}_{filename_base}_{timestamp}.txt")

def load_test_examples():
    """Load test examples from eval_gloss.txt and eval_translation.txt files."""
    examples = []
    
    with open('eval_gloss.txt', 'r') as gloss_file:
        glosses = [line.strip() for line in gloss_file.readlines() if line.strip()]
    
    with open('eval_translation.txt', 'r') as translation_file:
        translations = [line.strip() for line in translation_file.readlines() if line.strip()]
    
    for gloss, translation in zip(glosses, translations):
        examples.append({
            "input": gloss,
            "expected": translation
        })
    
    with open('examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2)
    
    return examples

def get_few_shot_examples(current_example, examples, n=3, use_chrf=False):
    """Get few-shot examples for prompting."""
    if not use_chrf:
        import random
        candidates = [ex for ex in examples if ex != current_example]
        if len(candidates) <= n:
            return candidates, []
        selected = random.sample(candidates, n)
        return selected, []
    else:
        print("Using chrF++ retrieval for few-shot examples")
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
            print("Need more examples after filtering duplicates, finding most similar from all examples...")
            
            scored_candidates = []
            for ex in examples:
                if ex == current_example or ex in similar_examples or ex['input'].lower() == current_example['input'].lower():
                    continue
                    
                score = retriever._compute_chrf_score(current_example['input'], ex['input'])
                scored_candidates.append((score, ex))
            
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            remaining_needed = n - len(similar_examples)
            additional_examples = [ex for _, ex in scored_candidates[:remaining_needed]]
            
            if additional_examples:
                print(f"Adding {len(additional_examples)} additional similar examples")
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
    
    if "Qwen3" in model_type or model_type == "finetuned":
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
        
        # Extract the generated part
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
    
    # Special handling for Llama 4
    elif "Llama-4" in model_type:
        # Prepare the model input using chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model_device)
        
        # Generate with Llama 4 parameters from official documentation
        # https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768,  # Official max context length
                temperature=0.7,       # Default from official examples
                top_p=0.9,            # Default from official examples
                do_sample=True,       # Required for temperature and top_p
                repetition_penalty=1.1 # Recommended for better output quality
            )
        
        # Extract the generated part
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
    
    else:
        # Original generation logic for other models
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,    # Shorter context for other models
                temperature=0.7,       # Standard temperature
                top_p=0.9,            # Standard top_p
                do_sample=True        # Required for temperature and top_p
            )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (not including the prompt)
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        prompt_text = tokenizer.decode(prompt_tokens.input_ids[0], skip_special_tokens=True)
        generated_text = translation[len(prompt_text):]
        
        return generated_text

def test_prompts(model_option="llama", force_cpu=False, limit=None, batch_size=1, use_chrf=False, template="few_shot", only_chrf=False, custom_model_path=None):
    """Test different prompt approaches."""
    model_name = MODEL_OPTIONS.get(model_option)
    if not model_name:
        print(f"Error: Unknown model option '{model_option}'")
        return False
    
    print(f"Testing using model: {model_name}")
    model, tokenizer = load_model(model_name, force_cpu, custom_model_path)
    
    if model is None or tokenizer is None:
        return False
    
    if torch.cuda.is_available():
        model_device = next(model.parameters()).device
        if not model_device.type == 'cuda':
            print(f"Warning: Model is on {model_device}, attempting to move to GPU...")
            model = model.to(device)
            model_device = next(model.parameters()).device
            print(f"Model is now on: {model_device}")
    
    zero_shot_template = read_prompt_template('zero_shot_prompt.txt')
    standard_few_shot_template = read_prompt_template('few_shot_prompt.txt')
    advanced_few_shot_template = read_prompt_template('advanced_few_shot_prompt.txt')
    
    print(f"Using template: {template}")
    
    test_examples = load_test_examples()
    if limit:
        test_examples = test_examples[:limit]
        print(f"Limited to {limit} examples")
    
    run_zero_shot = template == "zero_shot"
    run_few_shot = template == "few_shot" and not only_chrf
    run_advanced_few_shot = template == "advanced_few_shot" and not only_chrf
    run_chrf_few_shot = use_chrf and (template == "few_shot" or only_chrf)
    run_chrf_advanced_few_shot = use_chrf and (template == "advanced_few_shot" or only_chrf)
    
    results = {}
    
    if run_zero_shot:
        results["zero_shot"] = []
    if run_few_shot:
        results["few_shot"] = []
    if run_advanced_few_shot:
        results["advanced_few_shot"] = []
    if run_chrf_few_shot:
        results["chrf_few_shot"] = []
    if run_chrf_advanced_few_shot:
        results["chrf_advanced_few_shot"] = []
    
    prompts_log = []
    
    try:
        for i, example in enumerate(test_examples):
            print(f"\nTesting Example {i+1}/{len(test_examples)}:")
            print(f"Input: {example['input']}")
            print(f"Expected: {example['expected']}")
            
            current_prompts = {}
            
            if run_zero_shot:
                zero_shot_prompt = zero_shot_template.replace("{input_text}", example['input'])
                
                zero_shot_filename = generate_prompt_filename(example['input'], "zero_shot", model_option)
                with open(zero_shot_filename, 'w', encoding='utf-8') as f:
                    f.write(zero_shot_prompt)
                
                current_prompts["zero_shot"] = {
                    "prompt": zero_shot_prompt,
                    "file": zero_shot_filename
                }
                
                zero_shot_result = generate_translation(model, tokenizer, zero_shot_prompt, model_option)
                print(f"Zero-shot: {zero_shot_result}")
                
                if torch.cuda.is_available() and batch_size == 1:
                    clear_gpu_memory()
                
                results["zero_shot"].append({
                    "input": example['input'],
                    "expected": example['expected'],
                    "generated": zero_shot_result
                })
            
            if run_few_shot:
                standard_few_shot_examples, _ = get_few_shot_examples(example, test_examples, n=3, use_chrf=False)
                few_shot_prompt = prepare_custom_prompt(standard_few_shot_template, example['input'], standard_few_shot_examples)
                
                few_shot_filename = generate_prompt_filename(example['input'], "few_shot", model_option)
                with open(few_shot_filename, 'w', encoding='utf-8') as f:
                    f.write(few_shot_prompt)
                
                current_prompts["few_shot"] = {
                    "prompt": few_shot_prompt,
                    "file": few_shot_filename,
                    "examples_used": [ex['input'] for ex in standard_few_shot_examples]
                }
                
                few_shot_result = generate_translation(model, tokenizer, few_shot_prompt, model_option)
                print(f"Few-shot: {few_shot_result}")
                
                results["few_shot"].append({
                    "input": example['input'],
                    "expected": example['expected'],
                    "generated": few_shot_result,
                    "examples_used": [ex['input'] for ex in standard_few_shot_examples]
                })
                
                if torch.cuda.is_available() and batch_size == 1:
                    clear_gpu_memory()
            
            if run_advanced_few_shot:
                if run_few_shot:
                    advanced_few_shot_examples = standard_few_shot_examples
                else:
                    advanced_few_shot_examples, _ = get_few_shot_examples(example, test_examples, n=3, use_chrf=False)
                
                advanced_few_shot_prompt = prepare_custom_prompt(advanced_few_shot_template, example['input'], advanced_few_shot_examples)
                
                advanced_few_shot_filename = generate_prompt_filename(example['input'], "advanced_few_shot", model_option)
                with open(advanced_few_shot_filename, 'w', encoding='utf-8') as f:
                    f.write(advanced_few_shot_prompt)
                
                current_prompts["advanced_few_shot"] = {
                    "prompt": advanced_few_shot_prompt,
                    "file": advanced_few_shot_filename,
                    "examples_used": [ex['input'] for ex in advanced_few_shot_examples]
                }
                
                advanced_few_shot_result = generate_translation(model, tokenizer, advanced_few_shot_prompt, model_option)
                print(f"Advanced few-shot: {advanced_few_shot_result}")
                
                results["advanced_few_shot"].append({
                    "input": example['input'],
                    "expected": example['expected'],
                    "generated": advanced_few_shot_result,
                    "examples_used": [ex['input'] for ex in advanced_few_shot_examples]
                })
                
                if torch.cuda.is_available() and batch_size == 1:
                    clear_gpu_memory()
            
            if run_chrf_few_shot or run_chrf_advanced_few_shot:
                if torch.cuda.is_available():
                    clear_gpu_memory()
                
                chrf_few_shot_examples, similarity_scores = get_few_shot_examples(example, test_examples, n=3, use_chrf=True)
                
                if run_chrf_few_shot:
                    chrf_few_shot_prompt = prepare_custom_prompt(standard_few_shot_template, example['input'], chrf_few_shot_examples)
                    
                    chrf_few_shot_filename = generate_prompt_filename(example['input'], "chrf_few_shot", model_option)
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
                    
                    # Save chrF++ standard result
                    results["chrf_few_shot"].append({
                        "input": example['input'],
                        "expected": example['expected'],
                        "generated": chrf_few_shot_result,
                        "examples_used": [ex['input'] for ex in chrf_few_shot_examples],
                        "similarity_scores": similarity_scores,
                        "template_used": "few_shot"
                    })
                    
                    # Clear memory if needed
                    if torch.cuda.is_available() and batch_size == 1:
                        clear_gpu_memory()
                
                # Generate chrF++ with advanced template
                if run_chrf_advanced_few_shot:
                    chrf_advanced_prompt = prepare_custom_prompt(advanced_few_shot_template, example['input'], chrf_few_shot_examples)
                    
                    # Save chrF++ advanced few-shot prompt
                    chrf_advanced_filename = generate_prompt_filename(example['input'], "chrf_advanced_few_shot", model_option)
                    with open(chrf_advanced_filename, 'w', encoding='utf-8') as f:
                        f.write(chrf_advanced_prompt)
                    
                    # Add to prompts log
                    current_prompts["chrf_advanced_few_shot"] = {
                        "prompt": chrf_advanced_prompt,
                        "file": chrf_advanced_filename,
                        "examples_used": [ex['input'] for ex in chrf_few_shot_examples],
                        "similarity_scores": similarity_scores,
                        "template_used": "advanced_few_shot"
                    }
                    
                    # Generate translation
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
                    
                    if torch.cuda.is_available() and batch_size == 1:
                        clear_gpu_memory()
            
            prompts_log.append({
                "input": example['input'],
                "expected": example['expected'],
                "prompts": current_prompts
            })
            
            prompts_log_file = f"prompts_log_{model_option}.json"
            with open(os.path.join(RESULTS_DIR, prompts_log_file), 'w') as file:
                json.dump(prompts_log, file, indent=2)
            
            if (i + 1) % max(1, batch_size) == 0 or (i + 1) == len(test_examples):
                intermediate_file = f"translation_results_{model_option}_partial.json"
                with open(os.path.join(RESULTS_DIR, intermediate_file), 'w') as file:
                    json.dump(results, file, indent=2)
                print(f"Intermediate results saved to {os.path.join(RESULTS_DIR, intermediate_file)}")
                
                if torch.cuda.is_available():
                    clear_gpu_memory()
    
    except Exception as e:
        print(f"Error during translation: {e}")
        results_file = f"translation_results_{model_option}_error.json"
        with open(os.path.join(RESULTS_DIR, results_file), 'w') as file:
            json.dump(results, file, indent=2)
        print(f"Partial results saved to {os.path.join(RESULTS_DIR, results_file)}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return False
    
    results_file = f"translation_results_{model_option}.json"
    with open(os.path.join(RESULTS_DIR, results_file), 'w') as file:
        json.dump(results, file, indent=2)
    
    print(f"\nResults saved to {os.path.join(RESULTS_DIR, results_file)}")
    print(f"Prompts log saved to {os.path.join(RESULTS_DIR, prompts_log_file)}")
    
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
    parser.add_argument('--template', choices=['few_shot', 'advanced_few_shot', 'zero_shot'], default='few_shot', 
                       help='Select which prompt template to run (only the specified template will be tested unless --only-chrf is used)')
    parser.add_argument('--only-chrf', action='store_true', 
                       help='Only run zero-shot and chrF++ tests (both standard and advanced templates)')
    parser.add_argument('--custom-model-path', type=str, help='Custom path to a finetuned model or adapter')
    args = parser.parse_args()
    
    print(f"Running with options: model={args.model}, force_cpu={args.force_cpu}, limit={args.limit}, " 
         f"batch_size={args.batch_size}, use_chrf={args.use_chrf}, template={args.template}, "
         f"only_chrf={args.only_chrf}, custom_model_path={args.custom_model_path}")
    
    if args.only_chrf and not args.use_chrf:
        print("Warning: --only-chrf requires --use-chrf. Enabling --use-chrf automatically.")
        args.use_chrf = True
    
    if args.all:
        success = True
        for model_key in MODEL_OPTIONS:
            print(f"\n=== Testing {MODEL_OPTIONS[model_key]} ===")
            if not test_prompts(model_key, force_cpu=args.force_cpu, limit=args.limit, 
                              batch_size=args.batch_size, use_chrf=args.use_chrf, 
                              template=args.template, only_chrf=args.only_chrf,
                              custom_model_path=args.custom_model_path):
                print(f"Tests for {model_key} failed!")
                success = False
        
        if not success:
            sys.exit(1)
    else:
        if not test_prompts(args.model, force_cpu=args.force_cpu, limit=args.limit, 
                          batch_size=args.batch_size, use_chrf=args.use_chrf, 
                          template=args.template, only_chrf=args.only_chrf,
                          custom_model_path=args.custom_model_path):
            sys.exit(1)

if __name__ == "__main__":
    main() 