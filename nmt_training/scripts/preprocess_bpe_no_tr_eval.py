#!/usr/bin/env python3
import os
import random
import sys
import argparse
import sentencepiece as spm
from typing import List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def read_file(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def write_file(data: List[str], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')

def split_data(source: List[str], target: List[str], train_ratio=0.8, valid_ratio=0.1) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]], List[str]]:
    assert len(source) == len(target), "Source and target must have the same length"
    
    pairs = list(zip(source, target))
    turkish_pairs = [pair for pair in pairs if pair[0].startswith('tur')]
    non_turkish_pairs = [pair for pair in pairs if not pair[0].startswith('tur')]
    
    lang_counts = {}
    for src, _ in pairs:
        lang_code = src.split()[0] if src.split() else "unknown"
        if lang_code in lang_counts:
            lang_counts[lang_code] += 1
        else:
            lang_counts[lang_code] = 1
    
    random.shuffle(turkish_pairs)
    random.shuffle(non_turkish_pairs)
    
    non_turkish_total = len(non_turkish_pairs)
    non_turkish_train_size = int(non_turkish_total * train_ratio)
    non_turkish_valid_size = int(non_turkish_total * valid_ratio)
    
    non_turkish_train = non_turkish_pairs[:non_turkish_train_size]
    non_turkish_valid = non_turkish_pairs[non_turkish_train_size:non_turkish_train_size + non_turkish_valid_size]
    non_turkish_test = non_turkish_pairs[non_turkish_train_size + non_turkish_valid_size:]
    
    turkish_total = len(turkish_pairs)
    turkish_train_ratio = train_ratio / (train_ratio + valid_ratio)
    turkish_valid_ratio = valid_ratio / (train_ratio + valid_ratio)
    
    turkish_train_size = int(turkish_total * turkish_train_ratio)
    
    turkish_train = turkish_pairs[:turkish_train_size]
    turkish_valid = turkish_pairs[turkish_train_size:]
    
    train_pairs = non_turkish_train + turkish_train
    valid_pairs = non_turkish_valid + turkish_valid
    test_pairs = non_turkish_test
    
    random.shuffle(train_pairs)
    random.shuffle(valid_pairs)
    
    train_lang_counts = {}
    valid_lang_counts = {}
    test_lang_counts = {}
    
    for src, _ in train_pairs:
        lang_code = src.split()[0] if src.split() else "unknown"
        if lang_code in train_lang_counts:
            train_lang_counts[lang_code] += 1
        else:
            train_lang_counts[lang_code] = 1
            
    for src, _ in valid_pairs:
        lang_code = src.split()[0] if src.split() else "unknown"
        if lang_code in valid_lang_counts:
            valid_lang_counts[lang_code] += 1
        else:
            valid_lang_counts[lang_code] = 1
            
    for src, _ in test_pairs:
        lang_code = src.split()[0] if src.split() else "unknown"
        if lang_code in test_lang_counts:
            test_lang_counts[lang_code] += 1
        else:
            test_lang_counts[lang_code] = 1
    
    stats = [
        f"Total samples: {len(pairs)}",
        f"Turkish samples: {len(turkish_pairs)}",
        f"Non-Turkish samples: {len(non_turkish_pairs)}",
        f"Training set: {len(train_pairs)} samples ({sum(1 for p in train_pairs if p[0].startswith('tur'))} Turkish)",
        f"Validation set: {len(valid_pairs)} samples ({sum(1 for p in valid_pairs if p[0].startswith('tur'))} Turkish)",
        f"Test set: {len(test_pairs)} samples ({sum(1 for p in test_pairs if p[0].startswith('tur'))} Turkish)",
        "",
        "Language Distribution in Dataset:",
    ]
    
    # Add overall language distribution
    for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(pairs)) * 100
        stats.append(f"  {lang}: {count} samples ({percentage:.2f}%)")
    
    # Add language distribution in splits
    stats.append("")
    stats.append("Language Distribution in Training Set:")
    for lang, count in sorted(train_lang_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(train_pairs)) * 100
        stats.append(f"  {lang}: {count} samples ({percentage:.2f}%)")
        
    stats.append("")
    stats.append("Language Distribution in Validation Set:")
    for lang, count in sorted(valid_lang_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(valid_pairs)) * 100
        stats.append(f"  {lang}: {count} samples ({percentage:.2f}%)")
        
    stats.append("")
    stats.append("Language Distribution in Test Set:")
    for lang, count in sorted(test_lang_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_pairs)) * 100
        stats.append(f"  {lang}: {count} samples ({percentage:.2f}%)")
    
    for stat in stats:
        print(stat)
    
    return train_pairs, valid_pairs, test_pairs, stats

def train_sentencepiece(sentences: List[str], model_prefix: str, vocab_size: int = 8000, model_type: str = 'bpe'):
    """Train a SentencePiece model on the given sentences."""
    temp_file = f"{model_prefix}.tmp"
    write_file(sentences, temp_file)
    
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        normalization_rule_name='nmt_nfkc',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    
    os.remove(temp_file)
    print(f"SentencePiece model trained and saved to {model_prefix}.model and {model_prefix}.vocab")
    return f"{model_prefix}.model"

def apply_sentencepiece(sentences: List[str], model_path: str) -> List[str]:
    """Apply SentencePiece tokenization to the sentences."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    tokenized_sentences = []
    for sent in sentences:
        tokens = sp.encode_as_pieces(sent)
        tokenized_sentences.append(' '.join(tokens))
    
    return tokenized_sentences

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data with SentencePiece BPE tokenization')
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help='Vocabulary size for SentencePiece model')
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram'],
                        help='SentencePiece model type (bpe or unigram)')
    parser.add_argument('--save_raw', action='store_true', default=True,
                        help='Save raw untokenized versions of the test and validation data')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define paths
    raw_data_dir = os.path.join(project_root, 'data_multi_100k_plus_600_sampled_setimes_preset_eval_tur', 'raw')
    processed_data_dir = os.path.join(project_root, 'data_multi_100k_plus_600_sampled_setimes_preset_eval_tur', 'processed')
    
    # Ensure directories exist
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Read raw data
    gloss_path = os.path.join(raw_data_dir, 'gloss.txt')
    translation_path = os.path.join(raw_data_dir, 'translation.txt')
    
    try:
        gloss = read_file(gloss_path)
        translation = read_file(translation_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure the files exist in {raw_data_dir}")
        sys.exit(1)
    
    print(f"Read {len(gloss)} gloss sentences and {len(translation)} translation sentences")
    
    train_pairs, valid_pairs, test_pairs, stats = split_data(gloss, translation)
    
    stats_path = os.path.join(processed_data_dir, 'stats.txt')
    write_file(stats, stats_path)
    print(f"Statistics saved to {stats_path}")
    
    train_gloss, train_translation = zip(*train_pairs)
    valid_gloss, valid_translation = zip(*valid_pairs)
    test_gloss, test_translation = zip(*test_pairs)
    
    print(f"Split data into {len(train_gloss)} training, {len(valid_gloss)} validation, and {len(test_gloss)} test samples")
    
    if args.save_raw:
        print("Saving raw untokenized versions of test and validation data...")
        write_file(test_gloss, os.path.join(processed_data_dir, 'test.gloss.raw'))
        write_file(test_translation, os.path.join(processed_data_dir, 'test.translation.raw'))
        write_file(valid_gloss, os.path.join(processed_data_dir, 'valid.gloss.raw'))
        write_file(valid_translation, os.path.join(processed_data_dir, 'valid.translation.raw'))
    
    gloss_model_prefix = os.path.join(processed_data_dir, 'spm.gloss')
    translation_model_prefix = os.path.join(processed_data_dir, 'spm.translation')
    
    print(f"Training SentencePiece model for gloss with vocab size {args.vocab_size}...")
    gloss_model_path = train_sentencepiece(list(train_gloss), gloss_model_prefix, args.vocab_size, args.model_type)
    
    print(f"Training SentencePiece model for translation with vocab size {args.vocab_size}...")
    translation_model_path = train_sentencepiece(list(train_translation), translation_model_prefix, args.vocab_size, args.model_type)
    
    print("Applying SentencePiece tokenization to training data...")
    train_gloss_tokenized = apply_sentencepiece(train_gloss, gloss_model_path)
    train_translation_tokenized = apply_sentencepiece(train_translation, translation_model_path)
    
    print("Applying SentencePiece tokenization to validation data...")
    valid_gloss_tokenized = apply_sentencepiece(valid_gloss, gloss_model_path)
    valid_translation_tokenized = apply_sentencepiece(valid_translation, translation_model_path)
    
    print("Applying SentencePiece tokenization to test data...")
    test_gloss_tokenized = apply_sentencepiece(test_gloss, gloss_model_path)
    test_translation_tokenized = apply_sentencepiece(test_translation, translation_model_path)
    
    write_file(train_gloss_tokenized, os.path.join(processed_data_dir, 'train.gloss'))
    write_file(train_translation_tokenized, os.path.join(processed_data_dir, 'train.translation'))
    write_file(valid_gloss_tokenized, os.path.join(processed_data_dir, 'valid.gloss'))
    write_file(valid_translation_tokenized, os.path.join(processed_data_dir, 'valid.translation'))
    write_file(test_gloss_tokenized, os.path.join(processed_data_dir, 'test.gloss'))
    write_file(test_translation_tokenized, os.path.join(processed_data_dir, 'test.translation'))
    
    gloss_vocab = read_file(f"{gloss_model_prefix}.vocab")
    translation_vocab = read_file(f"{translation_model_prefix}.vocab")
    
    gloss_tokens = [line.split('\t')[0] for line in gloss_vocab]
    translation_tokens = [line.split('\t')[0] for line in translation_vocab]
    
    write_file(gloss_tokens, os.path.join(processed_data_dir, 'vocab.gloss'))
    write_file(translation_tokens, os.path.join(processed_data_dir, 'vocab.translation'))
    
    print("Preprocessing with SentencePiece BPE completed successfully!")

if __name__ == '__main__':
    random.seed(42)
    main() 