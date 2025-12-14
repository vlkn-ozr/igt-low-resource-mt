#!/usr/bin/env python3
"""
Script to create a mapping file from morphological tags to GlossLM format tags.
"""

import sys
from pathlib import Path

def create_mapping(output_file='gloss_mapping.txt'):
    # Core morphological tag to GlossLM mapping
    core_mapping = {
        # Part of Speech
        'N': 'NOM',
        'N:prop': 'PROP',
        'Adj': 'ADJ',
        'Adv': 'ADV',
        'V': 'V',
        'Prn': 'PRN',
        'Det': 'DET',
        'Num': 'NUM',
        'Postp': 'POSTP',
        'Cnj': 'CNJ',
        
        # Cases
        'nom': 'NOM',
        'acc': 'ACC',
        'dat': 'DAT',
        'gen': 'GEN',
        'loc': 'LOC',
        'abl': 'ABL',
        'ins': 'INS',
        
        # Verb features
        'past': 'PST',
        'pres': 'PRES',
        'fut': 'FUT',
        'aor': 'AOR',
        'prog': 'PROG',
        'evid': 'EVID',
        'cond': 'COND',
        'opt': 'OPT',
        'imp': 'IMP',
        'pass': 'PASS',
        'caus': 'CAUS',
        'abil': 'ABIL',
        'neg': 'NEG',
        
        # Person/Number
        'p1s': '1SG',
        'p2s': '2SG',
        'p3s': '3SG',
        'p1p': '1PL',
        'p2p': '2PL',
        'p3p': '3PL',
        
        # Other features
        'pl': 'PL',
        'Q': 'Q',
        'rfl': 'REFL',
        'dir': 'DIR',
        'part': 'PART',
        'vn': 'VN',
 
    }
    
    # Write the mapping to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Morphological Tag to GlossLM Mapping\n")
        f.write("# Format: morph_tag -> GlossLM_tag\n\n")
        
        # Write core mappings
        f.write("## Core Mappings\n")
        for morph_tag, gloss_tag in sorted(core_mapping.items()):
            f.write(f"{morph_tag:<15} -> {gloss_tag}\n")
        
        # Write special cases and compound tags
        f.write("\n## Special Cases and Compound Tags\n")
        f.write("# These are examples of how compound tags should be handled:\n")
        f.write("V<past><3s>      -> V.PST.3SG\n")
        f.write("N<pl><acc>       -> N.PL.ACC\n")
        f.write("Adj<0><N><p3s>   -> ADJ.NMLZ.3SG.POSS\n")
        f.write("V<abil><neg>     -> V.ABIL.NEG\n")
        
        # Write notes about handling special cases
        f.write("\n## Notes\n")
        f.write("1. Multiple tags are joined with dots (.)\n")
        f.write("2. Preserve order: POS -> Number -> Case -> Other features\n")
        f.write("3. For verbs: POS -> TAM -> Polarity -> Person/Number\n")

def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'gloss_mapping.txt'
    create_mapping(output_file)
    print(f"Mapping file created: {output_file}")

if __name__ == "__main__":
    main() 