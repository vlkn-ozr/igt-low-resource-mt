#!/usr/bin/env python3

def get_label_priority(label):
    """
    Determine the priority of a label for keeping the most important ones.
    Lower numbers = higher priority (more important to keep).
    """
    # POS tags - highest priority
    if label in ['NOM', 'V', 'ADJ', 'ADV', 'CNJ', 'PROP', 'DET', 'PRN', 'POSTP', 'NUM']:
        return 1
    
    # Inflectional morphology - second priority
    if label in ['PL', '3SG', '1PL', '2PL', '3PL', '1SG', '2SG', '3S', '1S', '2S', '1P', '2P', '3P', 
                'PST', 'FUT', 'PROG', 'NEG', 'PASS', 'CAUS', 'ABIL', 'AOR', 'EVID']:
        return 2
    
    # Case markers - third priority
    if label in ['GEN', 'DAT', 'ACC', 'LOC', 'ABL', 'INS']:
        return 3
    
    # Derivational morphology - fourth priority
    if label in ['VN', 'PART', 'CV', 'CPL', 'LIK', 'LI', 'DIR', 'OPT', 'IMP']:
        return 4
    
    # The rest - lowest priority
    return 5

def simplify_to_max_2_labels(input_file, output_file):
    """
    Reads the input file and limits each word to at most 2 gloss labels.
    """
    # Dictionary to track which glosses were kept vs dropped
    dropped_stats = {}
    total_glosses = 0
    dropped_glosses = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Skip empty lines
            if not line.strip():
                outfile.write('\n')
                continue
            
            tokens = line.strip().split()
            new_tokens = []
            
            # Process each token
            for token in tokens:
                parts = token.split('-')
                word = parts[0]
                glosses = parts[1:]
                
                # Skip processing if it's already 2 or fewer glosses
                if len(glosses) <= 2:
                    new_tokens.append(token)
                    continue
                
                # Count total glosses for statistics
                total_glosses += len(glosses)
                
                # Sort glosses by priority
                sorted_glosses = sorted([(g, get_label_priority(g)) for g in glosses 
                                        if g not in ['', 'PUNC']], 
                                        key=lambda x: x[1])
                
                # Keep only the top 2 most important glosses
                kept_glosses = [g[0] for g in sorted_glosses[:2]]
                
                # Track dropped glosses
                for g in sorted_glosses[2:]:
                    dropped_glosses += 1
                    if g[0] in dropped_stats:
                        dropped_stats[g[0]] += 1
                    else:
                        dropped_stats[g[0]] = 1
                
                # Reconstruct the token with max 2 glosses
                new_token = word
                if kept_glosses:
                    new_token += '-' + '-'.join(kept_glosses)
                
                new_tokens.append(new_token)
            
            # Write the modified line
            outfile.write(' '.join(new_tokens) + '\n')
    
    # Generate and save dropped gloss statistics
    stats_file_name = output_file.replace('.txt', '_stats.txt')
    with open(stats_file_name, 'w', encoding='utf-8') as stats_file:
        stats_file.write(f"{'GLOSS':<20}{'DROPPED COUNT':<15}{'PERCENTAGE':<10}\n")
        stats_file.write("-" * 45 + "\n")
        
        # Sort by frequency
        sorted_drops = sorted(dropped_stats.items(), key=lambda x: x[1], reverse=True)
        
        for gloss, count in sorted_drops:
            percentage = (count / dropped_glosses) * 100 if dropped_glosses > 0 else 0
            stats_file.write(f"{gloss:<20}{count:<15}{percentage:.2f}%\n")
    
    # Count unique glosses in the new file
    unique_glosses = set()
    with open(output_file, 'r', encoding='utf-8') as max2_file:
        for line in max2_file:
            tokens = line.strip().split()
            for token in tokens:
                parts = token.split('-')
                for gloss in parts[1:]:
                    if gloss and gloss != 'PUNC':
                        unique_glosses.add(gloss)
    
    print(f"File processed: {input_file}")
    print(f"Total glosses before limiting: {total_glosses}")
    if dropped_glosses > 0:
        print(f"Glosses dropped: {dropped_glosses} ({dropped_glosses/total_glosses:.2%})")
    else:
        print("No glosses were dropped")
    print(f"Number of unique glosses remaining: {len(unique_glosses)}")
    print(f"Max 2 labels per word file saved to: {output_file}")
    print(f"Dropped glosses statistics saved to: {stats_file_name}")
    
    # Analyze final gloss distribution
    analyze_glosses(output_file)

def analyze_glosses(file_path):
    """
    Analyzes the distribution of glosses in the max 2 labels file.
    """
    # Dictionary to store the counts
    gloss_counts = {}
    total_glosses = 0
    
    # Process the file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            
            tokens = line.strip().split()
            
            for token in tokens:
                # Split on hyphens to get the glosses
                parts = token.split('-')
                
                # Skip the first part as it's the word itself
                for gloss in parts[1:]:
                    # Handle special cases
                    if gloss in ['', 'PUNC']:
                        continue
                    
                    # Count the gloss
                    total_glosses += 1
                    if gloss in gloss_counts:
                        gloss_counts[gloss] += 1
                    else:
                        gloss_counts[gloss] = 1
    
    # Save the results to a file
    counts_file_name = file_path.replace('.txt', '_counts.txt')
    with open(counts_file_name, 'w', encoding='utf-8') as out_file:
        out_file.write(f"{'GLOSS':<20}{'COUNT':<10}{'PERCENTAGE':<10}\n")
        out_file.write("-" * 40 + "\n")
        
        # Sort by frequency (descending)
        sorted_counts = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)
        
        for gloss, count in sorted_counts:
            percentage = (count / total_glosses) * 100 if total_glosses > 0 else 0
            out_file.write(f"{gloss:<20}{count:<10}{percentage:.2f}%\n")
    
    # Display top 10 glosses
    print("\nTop 10 most frequent glosses in max 2 labels file:")
    print(f"{'GLOSS':<10}{'COUNT':<10}{'PERCENTAGE':<10}")
    print("-" * 30)
    
    for gloss, count in sorted_counts[:10]:
        percentage = (count / total_glosses) * 100 if total_glosses > 0 else 0
        print(f"{gloss:<10}{count:<10}{percentage:.2f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simplify_max_2_odin.py input_file [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Generate output filename based on input filename
        output_file = input_file.replace('.txt', '_max_2.txt')
        if output_file == input_file:  # No .txt extension found
            output_file = input_file + '_max_2'
    
    simplify_to_max_2_labels(input_file, output_file) 