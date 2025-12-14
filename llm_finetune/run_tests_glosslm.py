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
    #"finetuned_qwen": "./finetuned_qwen25_7b_model",  # Add path to finetuned model
    #"finetuned_qwen3": "./finetuned_qwen3_8b_model",  # Add path to finetuned model
    "finetuned_qwen_glosslm": "./output_qwen25_glosslm",  # Add path to finetuned model
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

def run_translation_tests(model_option="llama", force_cpu=False, limit=None, batch_size=1, use_embeddings=False, templates=["few_shot"], only_embeddings=False, custom_model_path=None, gloss_file='eval_gloss.txt', translation_file='eval_translation.txt', train_gloss_file='train_gloss.txt', train_translation_file='train_translation.txt', prompt_dir='prompts', results_dir='results', iso_code='en', data_dir='vera'):
    """Run the translation tests for the specified model.
    
    Args:
        model_option: Which model to use
        force_cpu: Whether to force CPU usage
        limit: Limit number of test examples
        batch_size: Batch size for processing examples
        use_embeddings: Whether to use embedding-based retrieval
        templates: List of prompt templates to use (zero_shot, few_shot, advanced_few_shot)
        only_embeddings: Only run embedding-based tests
        custom_model_path: Path to a custom model
        gloss_file: Path to gloss file
        translation_file: Path to translation file
        train_gloss_file: Path to training gloss file
        train_translation_file: Path to training translation file
        prompt_dir: Directory for saving prompts
        results_dir: Directory for saving results
    """
    print(f"Running Translation Tests for {MODEL_OPTIONS[model_option]}")
    cmd = [sys.executable, 'test_prompts_glosslm.py', '--model', model_option]
    
    if force_cpu:
        cmd.append('--force-cpu')
    
    if limit is not None:
        cmd.extend(['--limit', str(limit)])
    
    if batch_size != 1:
        cmd.extend(['--batch-size', str(batch_size)])
    
    if use_embeddings:
        cmd.append('--use-embeddings')
    
    if templates:
        cmd.append('--templates')
        for template in templates:
            cmd.append(template)
    
    if only_embeddings:
        cmd.append('--only-embeddings')
    
    cmd.extend(['--data-dir', data_dir])
    
    if custom_model_path:
        cmd.extend(['--custom-model-path', custom_model_path])
    
    cmd.extend(['--gloss-file', gloss_file])
    cmd.extend(['--translation-file', translation_file])
    cmd.extend(['--train-gloss-file', train_gloss_file])
    cmd.extend(['--train-translation-file', train_translation_file])
    cmd.extend(['--prompt-dir', prompt_dir])
    cmd.extend(['--results-dir', results_dir])
    
    cmd.extend(['--iso-code', iso_code])
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Test translation prompts')
    parser.add_argument('--model', choices=list(MODEL_OPTIONS.keys()) + ['all'], default='llama')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage for model loading')
    parser.add_argument('--limit', type=int, help='Limit number of test examples')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing examples')
    parser.add_argument('--use-embeddings', action='store_true', help='Use embedding-based retrieval for few-shot examples')
    parser.add_argument('--templates', nargs='+', choices=PROMPT_TEMPLATES, default=['few_shot'],
                        help='Prompt templates to use for translation')
    parser.add_argument('--only-embeddings', action='store_true', 
                       help='Only run zero-shot and embedding-based tests (both standard and advanced templates)')
    parser.add_argument('--data-dir', default='vera', help='Directory containing training data files (default: vera)')
    parser.add_argument('--custom-model-path', type=str, help='Custom path to a finetuned model or adapter')
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
    parser.add_argument('--iso-code', default='en', 
                       help='ISO code for the language being tested (default: en)')
    args = parser.parse_args()
    
    # Print arguments
    print(f"Running with options: model={args.model}, force_cpu={args.force_cpu}, limit={args.limit}, "
         f"batch_size={args.batch_size}, use_embeddings={args.use_embeddings}, templates={args.templates}, "
         f"only_embeddings={args.only_embeddings}, custom_model_path={args.custom_model_path}, "
         f"gloss_file={args.gloss_file}, translation_file={args.translation_file}, "
         f"train_gloss_file={args.train_gloss_file}, train_translation_file={args.train_translation_file}, "
         f"prompt_dir={args.prompt_dir}, results_dir={args.results_dir}, iso_code={args.iso_code}, data_dir={args.data_dir}")
    
    # Ensure use_embeddings is enabled when only_embeddings is set
    if args.only_embeddings and not args.use_embeddings:
        print("Warning: --only-embeddings requires --use-embeddings. Enabling --use-embeddings automatically.")
        args.use_embeddings = True
    
    check_hf_token()
    
    models_to_test = list(MODEL_OPTIONS.keys()) if args.model == 'all' else [args.model]
    
    for model in models_to_test:
        # Run translation tests
        if not run_translation_tests(
            model, 
            force_cpu=args.force_cpu, 
            limit=args.limit, 
            batch_size=args.batch_size,
            use_embeddings=args.use_embeddings,
            templates=args.templates,
            only_embeddings=args.only_embeddings,
            custom_model_path=args.custom_model_path,
            gloss_file=args.gloss_file,
            translation_file=args.translation_file,
            train_gloss_file=args.train_gloss_file,
            train_translation_file=args.train_translation_file,
            prompt_dir=args.prompt_dir,
            results_dir=args.results_dir,
            iso_code=args.iso_code,
            data_dir=args.data_dir
        ):
            print(f"Translation tests for {model} failed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
