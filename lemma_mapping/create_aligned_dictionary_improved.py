#!/usr/bin/env python3
import collections
import re
import argparse
from typing import Dict, List, Tuple, Set

# List of common English function words to filter out
FUNCTION_WORDS = {
    # Articles
    'a', 'an', 'the',
    
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'around',
    'between', 'among', 'through', 'throughout', 'during', 'until', 'till', 'since',
    'before', 'after', 'above', 'below', 'over', 'under', 'behind', 'beside', 'beneath',
    'across', 'along', 'into', 'onto', 'within', 'without',
    
    # Conjunctions
    'and', 'but', 'or', 'nor', 'so', 'yet', 'because', 'although', 'since', 'unless',
    'while', 'where', 'if', 'that', 'than',
    
    # Auxiliary verbs and common forms of "to be"
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must',
    
    # Pronouns
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those',
    'who', 'whom', 'whose', 'which', 'what',
    
    # Other common function words
    'as', 'not', 'no', 'yes', 'all', 'any', 'each', 'every', 'some', 'many', 'much',
    'more', 'most', 'few', 'little', 'other', 'another', 'such', 'only', 'very',
    'too', 'just', 'even', 'still', 'rather', 'quite', 'somewhat'
}

def clean_word(word: str) -> str:
    """Clean a word by removing punctuation and lowercasing."""
    # Remove punctuation and lowercase
    return re.sub(r'[^\w\s]', '', word.lower())

def is_function_word(word: str) -> bool:
    """Check if a word is a function word."""
    clean = clean_word(word)
    return clean in FUNCTION_WORDS or len(clean) < 2

def process_alignments(
    tr_file: str, 
    en_file: str, 
    alignment_file: str, 
    min_freq: int = 3,
    max_translations: int = 3,
    filter_function_words: bool = True
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Process the alignments to create a dictionary mapping Turkish words to English translations.
    
    Args:
        tr_file: Path to the Turkish text file
        en_file: Path to the English text file
        alignment_file: Path to the alignment file
        min_freq: Minimum frequency for an alignment to be considered
        max_translations: Maximum number of translations to include per Turkish word
        filter_function_words: Whether to filter out function words
        
    Returns:
        A dictionary mapping Turkish words to a list of (English word, frequency) tuples
    """
    # This dictionary will map each Turkish word to a dictionary of candidate English words and their counts
    aligned_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    
    # Track statistics
    total_alignments = 0
    filtered_alignments = 0
    
    # Process the sentence pairs and their alignments together
    with open(tr_file, "r", encoding="utf-8") as f_tr, \
         open(en_file, "r", encoding="utf-8") as f_en, \
         open(alignment_file, "r", encoding="utf-8") as f_align:
        
        for line_num, (tr_line, en_line, align_line) in enumerate(zip(f_tr, f_en, f_align), 1):
            try:
                # Tokenize the sentences (assumes words are whitespace-separated)
                tr_tokens = tr_line.strip().split()
                en_tokens = en_line.strip().split()
                
                # Each alignment token is like "3-9"
                alignments = align_line.strip().split()
                
                # Process each alignment
                for alignment in alignments:
                    try:
                        idx_tr, idx_en = alignment.split("-")
                        idx_tr = int(idx_tr)
                        idx_en = int(idx_en)
                    except ValueError:
                        # Skip if the alignment token is not formatted as expected
                        continue
                    
                    total_alignments += 1
                    
                    # Make sure the indices are within the tokens list bounds
                    if idx_tr < len(tr_tokens) and idx_en < len(en_tokens):
                        tr_word = clean_word(tr_tokens[idx_tr])
                        en_word = clean_word(en_tokens[idx_en])
                        
                        # Skip empty words or very short words
                        if not tr_word or not en_word:
                            continue
                            
                        # Skip function words if requested
                        if filter_function_words and is_function_word(en_word):
                            filtered_alignments += 1
                            continue
                        
                        # Add to the counts
                        aligned_counts[tr_word][en_word] += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Filter alignments by frequency and build the final dictionary
    filtered_dict = {}
    for tr_word, en_word_dict in aligned_counts.items():
        # Filter out low-frequency alignments
        frequent_alignments = [(en, count) for en, count in en_word_dict.items() if count >= min_freq]
        
        # Sort by frequency (descending)
        frequent_alignments.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top N translations
        if frequent_alignments:
            filtered_dict[tr_word] = frequent_alignments[:max_translations]
    
    print(f"Total alignments processed: {total_alignments}")
    print(f"Function words filtered: {filtered_alignments}")
    print(f"Turkish words with alignments: {len(aligned_counts)}")
    print(f"Turkish words after frequency filtering: {len(filtered_dict)}")
    
    return filtered_dict

def write_dictionary(dictionary: Dict[str, List[Tuple[str, int]]], output_file: str, include_counts: bool = False):
    """
    Write the dictionary to a file.
    
    Args:
        dictionary: Dictionary mapping Turkish words to a list of (English word, frequency) tuples
        output_file: Path to the output file
        include_counts: Whether to include the alignment counts in the output
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for tr_word, translations in sorted(dictionary.items()):
            if include_counts:
                # Include all translations with their counts
                translations_str = ", ".join([f"{en} ({count})" for en, count in translations])
                f.write(f"{tr_word} -> {translations_str}\n")
            else:
                # Include only the translations without counts
                translations_str = ", ".join([en for en, _ in translations])
                f.write(f"{tr_word} -> {translations_str}\n")

def main():
    parser = argparse.ArgumentParser(description="Create an aligned Turkish-English dictionary from parallel corpus.")
    parser.add_argument("--tr-file", default="preprocessed_data/tr_corpus.txt",
                        help="Path to the Turkish text file")
    parser.add_argument("--en-file", default="preprocessed_data/en_corpus.txt",
                        help="Path to the English text file")
    parser.add_argument("--alignment-file", default="awesome_align_finetuned_long_output.txt",
                        help="Path to the alignment file")
    parser.add_argument("--output-file", default="tr_en_dictionary_awesome_align_finetuned_long_detailed.txt",
                        help="Path to the output dictionary file")
    parser.add_argument("--min-freq", type=int, default=1, 
                        help="Minimum frequency for an alignment to be considered")
    parser.add_argument("--max-translations", type=int, default=3,
                        help="Maximum number of translations to include per Turkish word")
    parser.add_argument("--include-counts", action="store_true",
                        help="Include alignment counts in the output")
    parser.add_argument("--keep-function-words", action="store_true",
                        help="Keep function words instead of filtering them out")
    
    args = parser.parse_args()
    
    print("Processing alignments...")
    dictionary = process_alignments(
        args.tr_file,
        args.en_file,
        args.alignment_file,
        min_freq=args.min_freq,
        max_translations=args.max_translations,
        filter_function_words=not args.keep_function_words
    )
    
    print(f"Writing dictionary to {args.output_file}...")
    write_dictionary(dictionary, args.output_file, include_counts=args.include_counts)
    print("Done!")

if __name__ == "__main__":
    main() 