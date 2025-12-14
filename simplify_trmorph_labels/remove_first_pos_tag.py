#!/usr/bin/env python3
import argparse
import os

def remove_first_pos_tag(input_file, output_dir=None, remove_all_pos=False):
    """
    Reads the gloss file and removes only the first POS tag in a gloss if there are multiple,
    or optionally removes all POS tags.
    
    Args:
        input_file (str): Path to the input file.
        output_dir (str): Directory to save output files. If None, creates a directory based on the input filename.
        remove_all_pos (bool): If True, removes all POS tags. If False, removes only the first POS tag.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return 0, 0
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    if output_dir is None:
        output_dir = f"{base_name}_removed_pos_tags"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
        
    if remove_all_pos:
        output_file = os.path.join(output_dir, f"{base_name}_without_all_pos_tags.txt")
        stats_file_name = os.path.join(output_dir, f"{base_name}_removed_all_pos_tags_stats.txt")
    else:
        output_file = os.path.join(output_dir, f"{base_name}_without_first_pos_tag.txt")
        stats_file_name = os.path.join(output_dir, f"{base_name}_removed_first_pos_tag_stats.txt")
    
    pos_tags = ['NOM', 'V', 'ADJ', 'ADV', 'CNJ', 'PROP', 'DET', 'PRN', 'POSTP', 'NUM']
    
    removed_pos_counts = {tag: 0 for tag in pos_tags}
    total_glosses = 0
    removed_glosses = 0
    
    print(f"Processing file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Statistics will be saved to: {stats_file_name}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if not line.strip():
                    outfile.write('\n')
                    continue
                
                tokens = line.strip().split()
                new_tokens = [tokens[0]]
                
                for token in tokens[1:]:
                    parts = token.split('-')
                    word = parts[0]
                    glosses = parts[1:]
                    
                    total_glosses += len(glosses)
                    
                    new_glosses = []
                    pos_removed = False
                    
                    for gloss in glosses:
                        if gloss in pos_tags:
                            if remove_all_pos or not pos_removed:
                                removed_glosses += 1
                                removed_pos_counts[gloss] += 1
                                pos_removed = True
                                continue
                        
                        new_glosses.append(gloss)
                    
                    new_token = word
                    if new_glosses:
                        new_token += '-' + '-'.join(new_glosses)
                    
                    new_tokens.append(new_token)
                
                outfile.write(' '.join(new_tokens) + '\n')
        
        with open(stats_file_name, 'w', encoding='utf-8') as stats_file:
            stats_file.write(f"{'POS TAG':<10}{'REMOVED COUNT':<15}{'PERCENTAGE':<10}\n")
            stats_file.write("-" * 35 + "\n")
            
            sorted_removals = sorted(removed_pos_counts.items(), key=lambda x: x[1], reverse=True)
            
            for tag, count in sorted_removals:
                if count > 0:
                    percentage = (count / removed_glosses) * 100 if removed_glosses > 0 else 0
                    stats_file.write(f"{tag:<10}{count:<15}{percentage:.2f}%\n")
            
            stats_file.write("\n--- Summary ---\n")
            stats_file.write(f"Input file: {input_file}\n")
            stats_file.write(f"Output file: {output_file}\n")
            stats_file.write(f"Total glosses: {total_glosses}\n")
            stats_file.write(f"POS tags removed: {removed_glosses} ({removed_glosses/total_glosses:.2%})\n")
            if remove_all_pos:
                stats_file.write("Mode: All POS tags removed\n")
            else:
                stats_file.write("Mode: Only first POS tag removed\n")
        
        print(f"Processed gloss file: {input_file}")
        print(f"Total glosses: {total_glosses}")
        print(f"POS tags removed: {removed_glosses} ({removed_glosses/total_glosses:.2%})")
        print(f"File saved to: {output_file}")
        print(f"Removal statistics saved to: {stats_file_name}")
        
        return total_glosses, removed_glosses
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return 0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove POS tags from gloss file')
    parser.add_argument('input', type=str, help='Input gloss file')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files (defaults to input_filename_removed_pos_tags)')
    parser.add_argument('--all', action='store_true', help='Remove all POS tags instead of just the first one')
    args = parser.parse_args()
    
    remove_first_pos_tag(args.input, args.output_dir, args.all) 