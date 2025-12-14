#!/usr/bin/env python3
import argparse

def count_gloss_types(input_file='gloss.txt', output_file='gloss_counts.txt'):
    """Counts the frequency of gloss types in a gloss file."""
    gloss_counts = {}
    
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            
            tokens = line.strip().split()
            
            for token in tokens[1:]:
                parts = token.split('-')
                
                for gloss in parts[1:]:
                    if gloss in ['', 'PUNC']:
                        continue
                    
                    if gloss in gloss_counts:
                        gloss_counts[gloss] += 1
                    else:
                        gloss_counts[gloss] = 1
    
    sorted_counts = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"{'GLOSS':<20}{'COUNT':<10}\n")
        out_file.write("-" * 30 + "\n")
        
        for gloss, count in sorted_counts:
            out_file.write(f"{gloss:<20}{count:<10}\n")
    
    print(f"Found {len(gloss_counts)} unique gloss types.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count gloss types in a file')
    parser.add_argument('--input', type=str, default='gloss.txt', help='Input gloss file')
    parser.add_argument('--output', type=str, default='gloss_counts.txt', help='Output file')
    args = parser.parse_args()
    
    count_gloss_types(args.input, args.output)

if __name__ == "__main__":
    count_gloss_types() 