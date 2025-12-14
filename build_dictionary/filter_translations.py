#!/usr/bin/env python3

import re
import argparse
from collections import defaultdict

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

def filter_translations(input_file, output_file, remove_function_words=True, min_length=2):
    """
    Filter the translations file by removing function words.
    
    Args:
        input_file (str): Path to the input translations file
        output_file (str): Path to the output filtered translations file
        remove_function_words (bool): Whether to remove function words
        min_length (int): Minimum length of words to keep
    """
    total_lines = 0
    total_words_before = 0
    total_words_after = 0
    removed_words = defaultdict(int)
    
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_lines += 1
            line = line.strip()
            
            # Skip empty lines
            if not line:
                fout.write('\n')
                continue
            
            # Handle lines with line numbers (e.g., "1| text")
            if '|' in line:
                parts = line.split('|', 1)
                line_num = parts[0].strip()
                text = parts[1].strip()
                has_line_num = True
            else:
                text = line
                has_line_num = False
            
            # Split into words and filter
            words = text.split()
            total_words_before += len(words)
            
            filtered_words = []
            for word in words:
                # Check if the word is a function word (ignoring case)
                word_lower = word.lower()
                
                # Handle words with punctuation (e.g., "and," -> "and")
                word_clean = re.sub(r'[^\w\s]', '', word_lower)
                
                if (remove_function_words and word_clean in FUNCTION_WORDS) or len(word_clean) < min_length:
                    removed_words[word_lower] += 1
                else:
                    filtered_words.append(word)
            
            total_words_after += len(filtered_words)
            
            # Write the filtered line
            filtered_text = ' '.join(filtered_words)
            if has_line_num:
                fout.write(f"{line_num}| {filtered_text}\n")
            else:
                fout.write(f"{filtered_text}\n")
    
    # Generate a report of removed words
    report_file = output_file + ".report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Translation Filtering Report\n\n")
        f.write("## Statistics\n")
        f.write(f"Total lines processed: {total_lines}\n")
        f.write(f"Total words before filtering: {total_words_before}\n")
        f.write(f"Total words after filtering: {total_words_after}\n")
        f.write(f"Words removed: {total_words_before - total_words_after}\n")
        f.write(f"Percentage of words removed: {(total_words_before - total_words_after) / total_words_before * 100:.2f}%\n\n")
        
        f.write("## Most Frequently Removed Words\n")
        for word, count in sorted(removed_words.items(), key=lambda x: x[1], reverse=True)[:100]:
            f.write(f"{word}: {count}\n")
    
    print(f"Filtered translations saved to: {output_file}")
    print(f"Report saved to: {report_file}")
    print(f"Total lines processed: {total_lines}")
    print(f"Words before filtering: {total_words_before}")
    print(f"Words after filtering: {total_words_after}")
    print(f"Words removed: {total_words_before - total_words_after} ({(total_words_before - total_words_after) / total_words_before * 100:.2f}%)")
    
    return total_words_before, total_words_after

def main():
    parser = argparse.ArgumentParser(description='Filter English translations by removing function words.')
    parser.add_argument('input_file', help='Path to the input translations file')
    parser.add_argument('output_file', help='Path to the output filtered translations file')
    parser.add_argument('--keep-function-words', action='store_true', 
                        help='Keep function words instead of removing them')
    parser.add_argument('--min-length', type=int, default=2,
                        help='Minimum length of words to keep (default: 2)')
    
    args = parser.parse_args()
    
    before, after = filter_translations(
        args.input_file, 
        args.output_file,
        remove_function_words=not args.keep_function_words,
        min_length=args.min_length
    )
    
    print(f"Translation filtering complete.")

if __name__ == "__main__":
    main() 