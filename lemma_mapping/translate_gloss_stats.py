#!/usr/bin/env python3
import os
import json
import argparse
import re

def turkish_lower(text):
    """
    Properly convert Turkish text to lowercase, handling special characters like 'İ' -> 'i'
    """
    text = text.replace('İ', 'i')
    return text.lower()

def load_tr_en_dictionary(dict_path, skipped_entries_path=None, duplicates_path=None):
    """
    Reads the dictionary file and returns a mapping:
      { tr_word: en_word, ... }
    Expected dictionary file line format:
      tr_word -> en_word
    Lines that don't match the expected format are skipped and optionally saved to a file.
    Duplicate entries (same Turkish word) are also tracked.
    """
    mapping = {}
    skipped_entries = []
    duplicate_entries = []
    seen_words = set()
    
    with open(dict_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            original_line = line
            line = line.strip()
            
            # Track skipped entries with reason
            if not line:
                skipped_entries.append((line_num, original_line, "Empty line"))
                continue
                
            if "->" not in line:
                skipped_entries.append((line_num, original_line, "Missing separator"))
                continue
                
            tr_word, en_word = line.split("->", 1)
            tr_word = tr_word.strip()
            en_word = en_word.strip()
            
            if not tr_word:
                skipped_entries.append((line_num, original_line, "Empty source word"))
                continue
            
            # Check for duplicates
            if tr_word in seen_words:
                duplicate_entries.append((line_num, original_line, tr_word, en_word))
            else:
                seen_words.add(tr_word)
                
            # Last entry for each word will be kept in the mapping
            mapping[tr_word] = en_word
    
    # Save skipped entries if path is provided
    if skipped_entries_path and skipped_entries:
        ensure_dir_exists(skipped_entries_path)
        with open(skipped_entries_path, "w", encoding="utf-8") as f:
            f.write(f"Total skipped entries: {len(skipped_entries)}\n")
            f.write("-" * 50 + "\n")
            for line_num, line, reason in skipped_entries:
                f.write(f"Line {line_num}: '{line.strip()}' - Reason: {reason}\n")
    
    # Save duplicate entries if path is provided
    if duplicates_path and duplicate_entries:
        ensure_dir_exists(duplicates_path)
        with open(duplicates_path, "w", encoding="utf-8") as f:
            f.write(f"Total duplicate entries: {len(duplicate_entries)}\n")
            f.write("-" * 50 + "\n")
            for line_num, line, tr_word, en_word in duplicate_entries:
                f.write(f"Line {line_num}: '{line.strip()}' - Duplicate key: '{tr_word}' -> '{en_word}'\n")
    
    stats = {
        "total_lines": line_num if 'line_num' in locals() else 0,
        "skipped_entries": len(skipped_entries),
        "duplicate_entries": len(duplicate_entries),
        "unique_entries": len(mapping)
    }
    
    return mapping, stats

def ensure_dir_exists(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def is_number(s):
    """Check if a string is a number or contains only digits"""
    return s.isdigit() or (s.replace('.', '', 1).isdigit() and s.count('.') <= 1)

def translate_gloss_with_stats(gloss_path, tr_en_mapping, output_path, 
                              replaced_words_path, not_replaced_words_path):
    """
    Reads the gloss file, translates Turkish words to English using the mapping,
    and generates statistics about the translation process.
    """
    total_words = 0
    replaced_words_count = 0
    used_dict_entries = set()
    replaced_words = {}
    not_replaced_words = set()
    
    # Create a lowercase version of the mapping for case-insensitive lookups
    lowercase_mapping = {turkish_lower(k): v for k, v in tr_en_mapping.items()}
    
    # Ensure output directories exist
    ensure_dir_exists(output_path)
    ensure_dir_exists(replaced_words_path)
    ensure_dir_exists(not_replaced_words_path)
    
    with open(gloss_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            tokens = line.strip().split()
            new_tokens = []
            for token in tokens:
                total_words += 1
                
                # If there's a hyphen, split into base and tags
                if "-" in token:
                    base, rest = token.split("-", 1)
                    
                    # Try exact match first
                    if base in tr_en_mapping:
                        new_token = tr_en_mapping[base] + "-" + rest
                        replaced_words_count += 1
                        used_dict_entries.add(base)
                        replaced_words[base] = tr_en_mapping[base]
                    # Try lowercase version
                    elif turkish_lower(base) in lowercase_mapping:
                        # Preserve original case pattern for the translation if possible
                        en_word = lowercase_mapping[turkish_lower(base)]
                        if base.isupper():
                            en_word = en_word.upper()
                        elif base[0].isupper():
                            en_word = en_word.capitalize()
                        new_token = en_word + "-" + rest
                        replaced_words_count += 1
                        used_dict_entries.add(turkish_lower(base))
                        replaced_words[base] = en_word
                    else:
                        # Check if it's a number
                        if is_number(base):
                            new_token = base + "-" + rest  # Keep numbers as is
                            replaced_words_count += 1  # Count numbers as replaced
                        else:
                            new_token = token
                            not_replaced_words.add(base)
                else:
                    # If the token does not have a hyphen, try to translate it directly
                    if token in tr_en_mapping:
                        new_token = tr_en_mapping[token]
                        replaced_words_count += 1
                        used_dict_entries.add(token)
                        replaced_words[token] = tr_en_mapping[token]
                    # Try lowercase version
                    elif turkish_lower(token) in lowercase_mapping:
                        # Preserve original case pattern for the translation if possible
                        en_word = lowercase_mapping[turkish_lower(token)]
                        if token.isupper():
                            en_word = en_word.upper()
                        elif token[0].isupper():
                            en_word = en_word.capitalize()
                        new_token = en_word
                        replaced_words_count += 1
                        used_dict_entries.add(turkish_lower(token))
                        replaced_words[token] = en_word
                    else:
                        # Check if it's a number
                        if is_number(token):
                            new_token = token  # Keep numbers as is
                            replaced_words_count += 1  # Count numbers as replaced
                        else:
                            new_token = token
                            not_replaced_words.add(token)
                
                new_tokens.append(new_token)
            fout.write(" ".join(new_tokens) + "\n")
    
    # Save replaced words
    with open(replaced_words_path, "w", encoding="utf-8") as f:
        for tr_word, en_word in replaced_words.items():
            f.write(f"{tr_word} -> {en_word}\n")
    
    # Save not replaced words (excluding numbers)
    with open(not_replaced_words_path, "w", encoding="utf-8") as f:
        for word in sorted(not_replaced_words):
            f.write(f"{word}\n")
    
    # Calculate statistics
    replacement_rate = (replaced_words_count / total_words) * 100 if total_words > 0 else 0
    dict_usage_rate = (len(used_dict_entries) / len(tr_en_mapping)) * 100 if tr_en_mapping else 0
    
    stats = {
        "total_words": total_words,
        "replaced_words_count": replaced_words_count,
        "replacement_rate": replacement_rate,
        "total_dict_entries": len(tr_en_mapping),
        "used_dict_entries": len(used_dict_entries),
        "dict_usage_rate": dict_usage_rate,
        "unique_replaced_words": len(replaced_words),
        "unique_not_replaced_words": len(not_replaced_words)
    }
    
    return stats

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Translate Turkish glosses to English and generate statistics")
    parser.add_argument("--dict-path", default="dict_llm_5k_lemma_assesment.txt",
                        help="Path to the Turkish-English dictionary file")
    parser.add_argument("--gloss-path", default="glosslm_disambiguated_glosslm_eval_tur_200_fixed.txt",
                        help="Path to the Turkish gloss file")  
    parser.add_argument("--output-path", default="glosses_en_llm_5k_lemma_assesment_glosslm_tur_200.txt",
                        help="Path to save the translated English gloss file") 
    parser.add_argument("--stats-path", default="translation_stats_llm_5k_lemma_assesment_glosslm_tur_200/translation_stats.json",
                        help="Path to save the translation statistics")
    parser.add_argument("--replaced-words-path", default="translation_stats_llm_5k_lemma_assesment_glosslm_tur_200/replaced_words.txt",
                        help="Path to save the replaced words")
    parser.add_argument("--not-replaced-words-path", default="translation_stats_llm_5k_lemma_assesment_glosslm_tur_200/not_replaced_words.txt",
                        help="Path to save the words that were not replaced")
    parser.add_argument("--skipped-entries-path", default="translation_stats_llm_5k_lemma_assesment_glosslm_tur_200/skipped_entries.txt",
                        help="Path to save skipped dictionary entries")
    parser.add_argument("--duplicates-path", default="translation_stats_llm_5k_lemma_assesment_glosslm_tur_200/duplicate_entries.txt",
                        help="Path to save duplicate dictionary entries")
    
    args = parser.parse_args()
    
    print(f"Loading dictionary from {args.dict_path}...")
    tr_en_mapping, dict_stats = load_tr_en_dictionary(
        args.dict_path, 
        args.skipped_entries_path,
        args.duplicates_path
    )
    
    print(f"Dictionary statistics:")
    print(f"- Total lines: {dict_stats['total_lines']}")
    print(f"- Skipped entries: {dict_stats['skipped_entries']}")
    print(f"- Duplicate entries: {dict_stats['duplicate_entries']}")
    print(f"- Unique entries loaded: {dict_stats['unique_entries']}")
    
    print(f"Translating glosses from {args.gloss_path} to {args.output_path}...")
    stats = translate_gloss_with_stats(
        args.gloss_path, tr_en_mapping, args.output_path,
        args.replaced_words_path, args.not_replaced_words_path
    )
    
    # Ensure stats directory exists
    ensure_dir_exists(args.stats_path)
    
    # Combine dictionary stats with translation stats
    stats.update({
        "dictionary": dict_stats
    })
    
    # Save statistics to a JSON file
    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    # Print statistics summary
    print("\nTranslation Statistics:")
    print(f"Total words processed: {stats['total_words']}")
    print(f"Words replaced (including numbers): {stats['replaced_words_count']} ({stats['replacement_rate']:.2f}%)")
    print(f"Dictionary entries: {stats['total_dict_entries']}")
    print(f"Dictionary entries used: {stats['used_dict_entries']} ({stats['dict_usage_rate']:.2f}%)")
    print(f"Unique replaced words from dictionary: {stats['unique_replaced_words']}")
    print(f"Unique non-replaced words: {stats['unique_not_replaced_words']}")
    print(f"\nReplaced words saved to: {args.replaced_words_path}")
    print(f"Non-replaced words saved to: {args.not_replaced_words_path}")
    print(f"Skipped entries saved to: {args.skipped_entries_path}")
    print(f"Duplicate entries saved to: {args.duplicates_path}")
    print(f"Full statistics saved to: {args.stats_path}") 