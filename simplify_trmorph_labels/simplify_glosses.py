#!/usr/bin/env python3

def get_simplified_gloss(gloss):
    """
    Maps a gloss to its simplified version based on the defined rules.
    """
    # Step 1: Keep high-frequency labels untouched
    high_freq_labels = {
        'NOM', 'V', 'ADJ', 'PROP', 'ADV', 'PL', 'LOC', 'GEN', 'NUM', 'CNJ', 
        'DAT', 'ACC', 'ARA', 'DET', 'POSTP', 'COO', 'PST', 'ABL'
    }
    
    # Person/number markers to keep
    person_number = {
        '1S', '2S', '3S', '1P', '2P', '3P', '1SG', '2SG', '3SG', '1PL', '2PL', '3PL'
    }
    
    # Step 2: Keep essential grammatical categories
    essential_grammar = {
        'PROG', 'FUT', 'EVID', 'AOR', 'INS', 'NEG', 'Q', 'PRN'
    }
    
    # Step 3: Merge similar tags
    # PART variants
    if gloss.startswith('PART:'):
        return 'PART'
    
    # CV variants
    if gloss.startswith('CV:'):
        return 'CV'
    
    # CPL variants
    if gloss.startswith('CPL:'):
        return 'CPL'
    
    # VN variants
    if gloss.startswith('VN:'):
        return 'VN'
    
    # Step 4: Drop very low-frequency tags (< 10 occurrences)
    low_freq_tags = {
        'COND', 'REFL', 'IMPF', 'POSTP:ADV:NOMC', 'MREDUP', 'ALPHA', 'RCP', 
        'CV:INCE', 'SYM', 'ROM', 'CV:DIKCE', 'DIST', 'SI', 'DIM', 'TYPO', 
        'ISE', 'CNJ:ADV', 'CV:YE', 'IK', 'ACCC', 'CPL:EVID'
    }
    if gloss in low_freq_tags:
        return None  # This will be filtered out
    
    # Keep if it's in our high-frequency or essential sets
    if gloss in high_freq_labels or gloss in person_number or gloss in essential_grammar:
        return gloss
    
    # By default, keep the gloss as is (we can review these later)
    return gloss

def simplify_gloss_file(input_file='gloss.txt', output_file='simplified_gloss.txt', stats_file='gloss_mapping_stats.txt'):
    """Reads a gloss file, simplifies the gloss labels, and saves the result."""
    mapping_stats = {}
    
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
                
                new_glosses = []
                for gloss in parts[1:]:
                    if gloss in ('', 'PUNC'):
                        new_glosses.append(gloss)
                        continue
                    
                    simplified = get_simplified_gloss(gloss)
                    
                    if gloss != simplified:
                        pair = (gloss, simplified)
                        mapping_stats[pair] = mapping_stats.get(pair, 0) + 1
                    
                    if simplified is not None:
                        new_glosses.append(simplified)
                
                new_token = word
                if new_glosses:
                    new_token += '-' + '-'.join(new_glosses)
                
                new_tokens.append(new_token)
            
            outfile.write(' '.join(new_tokens) + '\n')
    
    with open(stats_file, 'w', encoding='utf-8') as stats:
        stats.write("Original Gloss\tSimplified Gloss\tOccurrences\n")
        stats.write("-" * 60 + "\n")
        
        sorted_mappings = sorted(mapping_stats.items(), key=lambda x: x[1], reverse=True)
        
        for (original, simplified), count in sorted_mappings:
            stats.write(f"{original}\t{simplified if simplified else 'REMOVED'}\t{count}\n")
    
    unique_glosses = set()
    with open(output_file, 'r', encoding='utf-8') as simplified_file:
        for line in simplified_file:
            tokens = line.strip().split()
            for token in tokens[1:]:
                parts = token.split('-')
                for gloss in parts[1:]:
                    if gloss and gloss != 'PUNC':
                        unique_glosses.add(gloss)
    
    print(f"Original gloss file processed.")
    print(f"Number of unique glosses after simplification: {len(unique_glosses)}")
    print(f"Simplified glosses saved to: {output_file}")
    print(f"Mapping statistics saved to: {stats_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simplify gloss labels in a file')
    parser.add_argument('--input', type=str, default='gloss.txt', help='Input gloss file')
    parser.add_argument('--output', type=str, default='simplified_gloss.txt', help='Output file')
    parser.add_argument('--stats', type=str, default='gloss_mapping_stats.txt', help='Statistics file')
    args = parser.parse_args()
    
    simplify_gloss_file(args.input, args.output, args.stats)
