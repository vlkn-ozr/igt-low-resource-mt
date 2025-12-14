#!/usr/bin/env python3
import argparse

def create_aligned_dictionary(input_file, output_file, most_frequent_only=False, show_counts=True, ignore_case=True):
    """Create an aligned dictionary from aligned word pairs."""
    aligned_dict = {}
    translation_counts = {}
    lowercase_map = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        pairs = content.split()
        
        for pair in pairs:
            if '<sep>' in pair:
                source, target = pair.split('<sep>', 1)
                
                if ignore_case:
                    source_lower = source.lower()
                    if source_lower in lowercase_map:
                        canonical_source = lowercase_map[source_lower]
                    else:
                        lowercase_map[source_lower] = source
                        canonical_source = source
                else:
                    canonical_source = source
                
                count_key = (canonical_source, target)
                
                if count_key in translation_counts:
                    translation_counts[count_key] += 1
                else:
                    translation_counts[count_key] = 1
                
                if canonical_source in aligned_dict:
                    if target not in aligned_dict[canonical_source]:
                        aligned_dict[canonical_source].append(target)
                else:
                    aligned_dict[canonical_source] = [target]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for source, targets in sorted(aligned_dict.items(), key=lambda x: x[0].lower()):
            if most_frequent_only:
                max_count = 0
                best_target = targets[0]
                
                for target in targets:
                    count = translation_counts.get((source, target), 0)
                    if count > max_count:
                        max_count = count
                        best_target = target
                
                if show_counts:
                    f.write(f"{source} -> {best_target} ({max_count})\n")
                else:
                    f.write(f"{source} -> {best_target}\n")
            else:
                if show_counts:
                    targets_with_counts = []
                    for target in targets:
                        count = translation_counts.get((source, target), 0)
                        targets_with_counts.append(f"{target} ({count})")
                    targets_str = ', '.join(targets_with_counts)
                else:
                    targets_str = ', '.join(targets)
                
                f.write(f"{source} -> {targets_str}\n")
    
    print(f"Dictionary created successfully in {output_file}")
    print(f"Total entries: {len(aligned_dict)}")
    if ignore_case:
        print(f"Note: Duplicates with different capitalization were merged")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an aligned dictionary with translation counts')
    parser.add_argument('--input', required=True, help='Input file containing aligned pairs') 
    parser.add_argument('--output', required=True, help='Output file for the aligned dictionary')
    parser.add_argument('--most-frequent', action='store_true', default=True,
                        help='Only keep the most frequent translation for each source word')
    parser.add_argument('--show-counts', action='store_true', default=False,
                        help='Show counts for each translation')
    parser.add_argument('--ignore-case', action='store_true', default=True,
                        help='Ignore case when identifying duplicates')

    args = parser.parse_args()
    
    create_aligned_dictionary(args.input, args.output, args.most_frequent, args.show_counts, args.ignore_case)  