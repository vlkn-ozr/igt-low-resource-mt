#!/usr/bin/env python3
import os
import yaml
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
configs_dir = os.path.join(project_root, 'configs')


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def write_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Create OpenNMT configuration with BPE settings')
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help='Vocabulary size for SentencePiece model')
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram'],
                        help='SentencePiece model type (bpe or unigram)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(data_dir, exist_ok=True)
    
    config = {
        'data': {
            'corpus_1': {
                'path_src': os.path.join(data_dir, 'train.gloss'),
                'path_tgt': os.path.join(data_dir, 'train.translation'),
                'weight': 1
            },
            'valid': {
                'path_src': os.path.join(data_dir, 'valid.gloss'),
                'path_tgt': os.path.join(data_dir, 'valid.translation')
            }
        },
        'src_vocab': os.path.join(data_dir, 'vocab.gloss'),
        'tgt_vocab': os.path.join(data_dir, 'vocab.translation'),
        'save_model': os.path.join(project_root, 'models', f'rnn_gloss_nmt_bpe_{args.vocab_size}'),
        'save_checkpoint_steps': 500,
        'keep_checkpoint': 2,
        'seed': 3435,
        'train_steps': 100000,
        'valid_steps': 250,
        'report_every': 10,
        'tensorboard': True,
        'tensorboard_log_dir': os.path.join(project_root, 'models', 'logs'),
        'model_dtype': 'fp16',
        'optim': 'sgd',
        'learning_rate': 0.8,
        'learning_rate_decay': 0.7,
        'start_decay_at': 9,
        'max_grad_norm': 1.0,
        'batch_size': 64,
        'batch_type': 'sents',
        'normalization': 'sents',
        'accum_count': 2,
        'world_size': 1,
        'gpu_ranks': [0],
        'num_workers': 2,
        'bucket_size': 8192,
        'prefetch': 1,
        'dropout': 0.3,
        'label_smoothing': 0.1,
        'encoder_type': 'rnn',
        'decoder_type': 'rnn',
        'rnn_type': 'LSTM',
        'rnn_size': 1000,
        'word_vec_size': 600,
        'enc_layers': 4,
        'dec_layers': 4,
        'global_attention': 'general',
        'early_stopping': 9,
        'early_stopping_criteria': 'ppl',
        'param_init': 0.1,
        'train_from': '',
        'max_epochs': 13,
        'src_subword_model': os.path.join(data_dir, 'spm.gloss.model'),
        'tgt_subword_model': os.path.join(data_dir, 'spm.translation.model'),
        'src_subword_type': args.model_type,
        'tgt_subword_type': args.model_type,
        'src_subword_nbest': 1,
        'tgt_subword_nbest': 1,
        'src_subword_alpha': 0.0,
        'tgt_subword_alpha': 0.0
    }
    
    config_path = os.path.join(configs_dir, 'config_bpe_rnn_big.yaml')
    os.makedirs(configs_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"RNN Configuration with BPE prepared and saved to {config_path}")

if __name__ == '__main__':
    main() 