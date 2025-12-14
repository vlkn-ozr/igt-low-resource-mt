#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys
import re
import yaml
from nmt_logger import NMTLogger
import sentencepiece as spm
from detokenization_utils import detokenize_with_fallback, check_detokenization_consistency

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Translate using OpenNMT-py NMT model with BPE support')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input file (gloss)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output file (translation)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--log_dir', type=str, default=os.path.join(project_root, 'logs_rnn_small_multi_100k_plus_1200_sampled_setimes_no_pos_preset_eval_tur'),
                        help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default='onmt_translation_bpe',
                        help='Name of the experiment for logging')
    parser.add_argument('--config', type=str, default=os.path.join(project_root, 'configs', 'config_bpe_rnn_small_multi_100k_plus_1200_sampled_setimes_no_pos_preset_tur.yaml'),
                        help='Path to the configuration file with BPE settings')
    parser.add_argument('--src_bpe_model', type=str, default=None,
                        help='Path to source BPE model (overrides config file)')
    parser.add_argument('--tgt_bpe_model', type=str, default=None,
                        help='Path to target BPE model (overrides config file)')
    parser.add_argument('--language', type=str, default='en',
                        help='Language code for language-specific detokenization rules')
    parser.add_argument('--normalize_unicode', action='store_true',
                        help='Apply Unicode normalization')
    parser.add_argument('--fix_entities', action='store_true', default=True,
                        help='Apply named entity fixes')
    parser.add_argument('--fix_numbers', action='store_true', default=True,
                        help='Fix numbers and dates')
    parser.add_argument('--consistency_check', action='store_true',
                        help='Perform consistency check on detokenization')
    return parser.parse_args()

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def write_file(data, file_path):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')

def apply_bpe(sentences, model_path):
    """Apply BPE tokenization to the sentences."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    tokenized_sentences = []
    for sent in sentences:
        tokens = sp.encode_as_pieces(sent)
        tokenized_sentences.append(' '.join(tokens))
    
    return tokenized_sentences

def parse_onmt_translate_output(output_line):
    """Parse OpenNMT-py translation output line to extract metrics."""
    metrics = {}
    
    sent_match = re.search(r'Translated: (\d+)', output_line)
    if sent_match:
        metrics['sentences_processed'] = int(sent_match.group(1))
    
    speed_match = re.search(r'(\d+\.\d+) sent/s', output_line)
    if speed_match:
        metrics['translation_speed'] = float(speed_match.group(1))
    
    return metrics

def main():
    args = parse_args()
    
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = NMTLogger(log_dir=log_dir, experiment_name=args.experiment_name)
    logger.log_info(f"Starting OpenNMT translation with BPE: {args.experiment_name}")
    logger.log_info(f"Model: {args.model}")
    logger.log_info(f"Input file: {args.input}")
    logger.log_info(f"Output file: {args.output}")
    
    config_path = args.config
    logger.log_info(f"Using config file: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.log_error(f"Error loading config file: {e}")
        sys.exit(1)
    
    src_bpe_model = args.src_bpe_model or config.get('src_subword_model')
    tgt_bpe_model = args.tgt_bpe_model or config.get('tgt_subword_model')
    
    if src_bpe_model and not os.path.isabs(src_bpe_model):
        src_bpe_model = os.path.join(project_root, src_bpe_model)
    
    if tgt_bpe_model and not os.path.isabs(tgt_bpe_model):
        tgt_bpe_model = os.path.join(project_root, tgt_bpe_model)
    
    if not src_bpe_model or not os.path.exists(src_bpe_model):
        logger.log_error(f"Source BPE model not found: {src_bpe_model}")
        sys.exit(1)
    
    if not tgt_bpe_model or not os.path.exists(tgt_bpe_model):
        logger.log_error(f"Target BPE model not found: {tgt_bpe_model}")
        sys.exit(1)
    
    logger.log_info(f"Source BPE model: {src_bpe_model}")
    logger.log_info(f"Target BPE model: {tgt_bpe_model}")
    
    try:
        input_sentences = read_file(args.input)
        logger.log_info(f"Read {len(input_sentences)} sentences from input file")
    except Exception as e:
        logger.log_error(f"Error reading input file: {e}")
        sys.exit(1)
    
    logger.log_info("Applying BPE tokenization to input...")
    tokenized_input = apply_bpe(input_sentences, src_bpe_model)
    
    temp_input_file = f"{args.input}.bpe"
    write_file(tokenized_input, temp_input_file)
    logger.log_info(f"Tokenized input saved to {temp_input_file}")
    
    temp_output_file = f"{args.output}.bpe"
    
    cmd = [
        'onmt_translate',
        '-model', args.model,
        '-src', temp_input_file,
        '-output', temp_output_file,
        '-beam_size', '5',
        '-batch_size', '32',
        '-gpu', str(args.gpu),
        '-verbose'
    ]
    
    logger.log_info("Running command: " + ' '.join(cmd))
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            
            metrics = parse_onmt_translate_output(line)
            if metrics:
                logger.log_metrics(metrics)
        
        return_code = process.wait()
        if return_code != 0:
            logger.log_error(f"Translation process exited with code {return_code}")
            sys.exit(return_code)
        
        logger.log_info(f"Translation completed successfully! BPE output saved to {temp_output_file}")
        
        tokenized_output = read_file(temp_output_file)
        logger.log_info(f"Read {len(tokenized_output)} tokenized sentences from output file")
        
        original_tokenized = tokenized_output.copy() if args.consistency_check else None
        
        logger.log_info("Detokenizing BPE output...")
        detokenized_output = detokenize_with_fallback(
            tokenized_output, 
            model_path=tgt_bpe_model,
            lang=args.language,
            normalize=args.normalize_unicode
        )
        logger.log_info("Detokenization completed")
        
        if args.consistency_check:
            logger.log_info("Performing consistency check on detokenization...")
            inconsistent_count = 0
            for i, (orig, detok) in enumerate(zip(original_tokenized[:10], detokenized_output[:10])):
                consistency = check_detokenization_consistency(orig, detok)
                if not consistency['is_consistent']:
                    inconsistent_count += 1
                    logger.log_warning(f"Inconsistency in sentence {i+1}:")
                    logger.log_warning(f"  Original: {orig}")
                    logger.log_warning(f"  Detokenized: {detok}")
                    logger.log_warning(f"  Issues: {consistency}")
            
            if inconsistent_count > 0:
                logger.log_warning(f"Found {inconsistent_count} inconsistencies in the first 10 sentences.")
        
        write_file(detokenized_output, args.output)
        logger.log_info(f"Detokenized output saved to {args.output}")
        
        os.remove(temp_input_file)
        os.remove(temp_output_file)
        logger.log_info("Temporary files cleaned up")
        
    except subprocess.CalledProcessError as e:
        logger.log_error(f"Error running command: {e}")
        sys.exit(1)
    except Exception as e:
        logger.log_error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 