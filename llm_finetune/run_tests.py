#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

# Available models
MODEL_OPTIONS = {
    #"llama": "meta-llama/Llama-3.1-8B-Instruct",
    #"qwen": "Qwen/Qwen2.5-7B-Instruct",
    #"qwen3": "Qwen/Qwen3-8B",
    "finetuned_qwen": "./finetuned_qwen25_7b_model",  # Add path to finetuned model
    "finetuned_qwen3": "./finetuned_qwen3_8b_model",  # Add path to finetuned model
    #"llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
}

# Available prompt templates
PROMPT_TEMPLATES = ["zero_shot", "few_shot", "advanced_few_shot"]

def check_dependencies():
    """Check if all required packages are installed."""
    try:
        import torch
        import transformers
        import nltk
        
        # Download NLTK data if needed
        try:
            from nltk.tokenize import word_tokenize
            word_tokenize('Test')
        except LookupError:
            print("Downloading NLTK data...")
            import nltk
            nltk.download('punkt')
        
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        print("Please install all dependencies: pip install -r requirements.txt")
        return False

def check_hf_token():
    """Check if Hugging Face token is set."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("WARNING: HF_TOKEN environment variable not set.")
        print("To set: export HF_TOKEN=your_huggingface_token")
        return False
    return True

def run_translation_tests(model_option="llama", force_cpu=False, limit=None, batch_size=1, use_chrf=False, template="few_shot", only_chrf=False, custom_model_path=None):
    """Run the translation tests for the specified model.
    
    Args:
        model_option: Which model to use
        force_cpu: Whether to force CPU usage
        limit: Limit number of test examples
        batch_size: Batch size for processing examples
        use_chrf: Whether to use chrF++ retrieval
        template: Which prompt template to use (zero_shot, few_shot, advanced_few_shot)
        only_chrf: Only run chrF++ tests
        custom_model_path: Path to a custom model
    """
    print(f"Running Translation Tests for {MODEL_OPTIONS[model_option]}")
    cmd = [sys.executable, 'test_prompts.py', '--model', model_option]
    
    if force_cpu:
        cmd.append('--force-cpu')
    
    if limit is not None:
        cmd.extend(['--limit', str(limit)])
    
    if batch_size != 1:
        cmd.extend(['--batch-size', str(batch_size)])
    
    if use_chrf:
        cmd.append('--use-chrf')
    
    if template and template in PROMPT_TEMPLATES:
        cmd.extend(['--template', template])
    
    if only_chrf:
        cmd.append('--only-chrf')
    
    if custom_model_path:
        cmd.extend(['--custom-model-path', custom_model_path])
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Test translation prompts')
    parser.add_argument('--model', choices=list(MODEL_OPTIONS.keys()) + ['all'], default='llama')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage for model loading')
    parser.add_argument('--limit', type=int, help='Limit number of test examples')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing examples')
    parser.add_argument('--use-chrf', action='store_true', help='Use chrF++ retrieval for few-shot examples')
    parser.add_argument('--template', choices=PROMPT_TEMPLATES, default='few_shot',
                        help='Prompt template to use for translation')
    parser.add_argument('--only-chrf', action='store_true', 
                       help='Only run zero-shot and chrF++ tests (both standard and advanced templates)')
    parser.add_argument('--custom-model-path', type=str, help='Custom path to a finetuned model or adapter')
    args = parser.parse_args()
    
    # Print arguments
    print(f"Running with options: model={args.model}, force_cpu={args.force_cpu}, limit={args.limit}, "
         f"batch_size={args.batch_size}, use_chrf={args.use_chrf}, template={args.template}, "
         f"only_chrf={args.only_chrf}, custom_model_path={args.custom_model_path}")
    
    # Ensure use_chrf is enabled when only_chrf is set
    if args.only_chrf and not args.use_chrf:
        print("Warning: --only-chrf requires --use-chrf. Enabling --use-chrf automatically.")
        args.use_chrf = True
    
    check_hf_token()
    
    models_to_test = list(MODEL_OPTIONS.keys()) if args.model == 'all' else [args.model]
    
    for model in models_to_test:
        # Run translation tests
        if not run_translation_tests(
            model, 
            force_cpu=args.force_cpu, 
            limit=args.limit, 
            batch_size=args.batch_size,
            use_chrf=args.use_chrf,
            template=args.template,
            only_chrf=args.only_chrf,
            custom_model_path=args.custom_model_path
        ):
            print(f"Translation tests for {model} failed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 