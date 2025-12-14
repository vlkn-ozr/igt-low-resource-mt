#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

# Available models
MODEL_OPTIONS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
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

def run_translation_tests(model_option="llama", force_cpu=False, limit=None, batch_size=1, use_chrf=False, templates=["few_shot"], only_chrf=False, gloss_file='eval_gloss.txt', translation_file='eval_translation.txt', train_gloss_file='train_gloss.txt', train_translation_file='train_translation.txt', prompt_dir='prompts', results_dir='results'):
    """Run the translation tests for the specified model."""
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
    
    if templates:
        cmd.append('--templates')
        for template in templates:
            if template in PROMPT_TEMPLATES:
                cmd.append(template)
    
    if only_chrf:
        cmd.append('--only-chrf')
    
    if gloss_file != 'eval_gloss.txt':
        cmd.extend(['--gloss-file', gloss_file])
    
    if translation_file != 'eval_translation.txt':
        cmd.extend(['--translation-file', translation_file])
    
    if train_gloss_file != 'train_gloss.txt':
        cmd.extend(['--train-gloss-file', train_gloss_file])
    
    if train_translation_file != 'train_translation.txt':
        cmd.extend(['--train-translation-file', train_translation_file])
    
    if prompt_dir != 'prompts':
        cmd.extend(['--prompt-dir', prompt_dir])
    
    if results_dir != 'results':
        cmd.extend(['--results-dir', results_dir])
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Test translation prompts')
    parser.add_argument('--model', choices=list(MODEL_OPTIONS.keys()) + ['all'], default='llama')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage for model loading')
    parser.add_argument('--limit', type=int, help='Limit number of test examples')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing examples')
    parser.add_argument('--use-chrf', action='store_true', help='Use chrF++ retrieval for few-shot examples')
    parser.add_argument('--templates', nargs='+', choices=PROMPT_TEMPLATES, default=['few_shot'],
                        help='Prompt templates to use for translation')
    parser.add_argument('--only-chrf', action='store_true', 
                       help='Only run zero-shot and chrF++ tests (both standard and advanced templates)')
    parser.add_argument('--gloss-file', default='eval_gloss.txt', 
                       help='Path to the test gloss file (default: eval_gloss.txt)')
    parser.add_argument('--translation-file', default='eval_translation.txt', 
                       help='Path to the test translation file (default: eval_translation.txt)')
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
    
    check_hf_token()
    
    models_to_test = list(MODEL_OPTIONS.keys()) if args.model == 'all' else [args.model]
    
    for model in models_to_test:
        if not run_translation_tests(
            model, 
            force_cpu=args.force_cpu, 
            limit=args.limit, 
            batch_size=args.batch_size,
            use_chrf=args.use_chrf,
            templates=args.templates,
            only_chrf=args.only_chrf,
            gloss_file=args.gloss_file,
            translation_file=args.translation_file,
            train_gloss_file=args.train_gloss_file,
            train_translation_file=args.train_translation_file,
            prompt_dir=args.prompt_dir,
            results_dir=args.results_dir
        ):
            print(f"Translation tests for {model} failed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 