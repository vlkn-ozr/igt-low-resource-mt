import pandas as pd
import os
import csv
from tqdm import tqdm
import re
import pycountry
import iso639

def fix_alignment():
    """Rebuild source and target files from original subset data with perfect alignment."""
    def get_language_code(language_name):
        essential_mappings = {
            "Unknown language": "und",
            "Japhug": "jya",
            "Sanzhi": "xzr",
            "Pichi": "fpe",
            "Tsez": "ddo"
        }
        
        if language_name in essential_mappings:
            return essential_mappings[language_name]
        
        try:
            lang = pycountry.languages.lookup(language_name)
            if hasattr(lang, 'alpha_3'):
                return lang.alpha_3
        except (LookupError, AttributeError):
            pass
        
        try:
            for lang in iso639.iter_langs():
                if language_name.lower() in lang.name.lower() or lang.name.lower() in language_name.lower():
                    return lang.pt3
        except (KeyError, AttributeError):
            pass
        
        language_parts = language_name.lower().split()
        for part in language_parts:
            if len(part) > 3:
                try:
                    lang = pycountry.languages.get(name=part)
                    if lang and hasattr(lang, 'alpha_3'):
                        return lang.alpha_3
                except (LookupError, AttributeError):
                    pass
                
                try:
                    for lang in iso639.iter_langs():
                        if part.lower() in lang.name.lower():
                            return lang.pt3
                except (KeyError, AttributeError):
                    pass
        
        for country in pycountry.countries:
            if hasattr(country, 'name') and country.name.lower() in language_name.lower():
                for lang in pycountry.languages:
                    if hasattr(lang, 'alpha_3') and hasattr(lang, 'name'):
                        if country.name.lower() in lang.name.lower():
                            return lang.alpha_3
        
        cleaned_name = re.sub(r'[^a-zA-Z]', '', language_name)
        return cleaned_name[:3].lower() if cleaned_name else "und"
    
    print("Loading GlossLM subset...")
    if not os.path.exists('glosslm_subset.csv'):
        print("Error: glosslm_subset.csv not found!")
        return
    
    subset_df = pd.read_csv('glosslm_subset.csv')
    print(f"Loaded {len(subset_df)} examples")
    
    print("Creating language code mapping...")
    unique_languages = subset_df['language'].unique()
    language_to_code = {}
    for language in tqdm(unique_languages, desc="Mapping languages"):
        language_to_code[language] = get_language_code(language)
    
    os.makedirs('training_data_100k_tr_odin', exist_ok=True)
    with open('training_data_100k_tr_odin/language_codes.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['language', 'iso_code'])
        for language, code in sorted(language_to_code.items()):
            writer.writerow([language, code])
    
    print(f"Created language_codes.csv with mappings for {len(language_to_code)} languages")
    
    print("Creating clean paired examples...")
    valid_pairs = []
    skipped = 0
    
    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        language = row['language']
        iso_code = language_to_code.get(language, 'und')
        
        glosses = row['glosses']
        translation = row['translation']
        
        if pd.isna(glosses) or pd.isna(translation) or not str(glosses).strip() or not str(translation).strip():
            skipped += 1
            continue
        
        glosses = ' '.join(str(glosses).replace('\n', ' ').split())
        translation = ' '.join(str(translation).replace('\n', ' ').split())
        
        source_line = f"{iso_code} {glosses}"
        target_line = translation
        
        valid_pairs.append((source_line, target_line))
    
    print(f"Writing {len(valid_pairs)} aligned pairs to files...")
    
    with open('training_data_100k_tr_odin/source.txt', 'w', encoding='utf-8') as source_file, \
         open('training_data_100k_tr_odin/target.txt', 'w', encoding='utf-8') as target_file, \
         open('training_data_100k_tr_odin/aligned_pairs.tsv', 'w', encoding='utf-8') as tsv_file:
        
        tsv_file.write("line_num\tsource\ttarget\n")
        
        for i, (source, target) in enumerate(tqdm(valid_pairs, desc="Writing files")):
            source_file.write(f"{source}\n")
            target_file.write(f"{target}\n")
            tsv_file.write(f"{i+1}\t{source}\t{target}\n")
    
    print("\nVerifying alignment...")
    with open('training_data_100k_tr_odin/source.txt', 'r', encoding='utf-8') as source_file, \
         open('training_data_100k_tr_odin/target.txt', 'r', encoding='utf-8') as target_file:
        source_lines = source_file.readlines()
        target_lines = target_file.readlines()
        
        if len(source_lines) == len(target_lines):
            print(f"✓ ALIGNMENT VERIFIED: Both files have exactly {len(source_lines)} lines.")
            
            check_positions = [
                (0, "First line"),
                (len(source_lines)//4, "25% position"),
                (len(source_lines)//2, "Middle"),
                (3*len(source_lines)//4, "75% position"),
                (len(source_lines)-1, "Last line")
            ]
            
            if len(source_lines) > 18500:
                for i in range(18557, min(18562, len(source_lines))):
                    check_positions.append((i, f"Line {i+1} (problematic range)"))
            
            print("\nSample alignment checks:")
            print("-" * 60)
            for pos, desc in check_positions:
                print(f"{desc}:")
                print(f"  Source: {source_lines[pos].strip()}")
                print(f"  Target: {target_lines[pos].strip()}")
                print("-" * 60)
        else:
            print(f"× ALIGNMENT ERROR: source.txt has {len(source_lines)} lines, but target.txt has {len(target_lines)} lines.")
    
    print(f"\nDone! Created perfectly aligned training data files:")
    print(f"- source.txt: contains {len(valid_pairs)} examples with format: 'iso_code glosses'")
    print(f"- target.txt: contains {len(valid_pairs)} examples with translations")
    print(f"- aligned_pairs.tsv: contains both source and target in a single file (for verification)")
    print(f"- language_codes.csv: contains mappings from language names to ISO codes")
    print(f"Skipped {skipped} examples due to missing data")
    
    with open('training_data_100k_tr_odin/alignment_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("ALIGNMENT REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total examples in original subset: {len(subset_df)}\n")
        f.write(f"Valid aligned pairs: {len(valid_pairs)}\n")
        f.write(f"Skipped examples: {skipped}\n\n")
        f.write("Files created:\n")
        f.write("- source.txt: Contains source sentences with ISO language tags in format 'iso_code glosses'\n")
        f.write("- target.txt: Contains target translations\n")
        f.write("- aligned_pairs.tsv: Contains both source and target in a tab-separated format\n")
        f.write("- language_codes.csv: Contains mappings from language names to ISO codes\n\n")
        f.write("Format explanation:\n")
        f.write("The source.txt file contains lines in the format 'iso_code glosses', where:\n")
        f.write("- iso_code: 3-letter ISO 639-3 code for the language (e.g., 'tur' for Turkish)\n")
        f.write("- glosses: The glosses for the example\n\n")
        f.write("The target.txt file contains the corresponding translations in English.\n\n")
        f.write("This format follows the paper's specification, with examples such as:\n")
        f.write("  - 'tur 1SG yesterday evening PRO eat'\n")
        f.write("  - 'deu 1SG.ACC be_thirsty'\n\n")
        f.write("The target file contains the corresponding English translations.\n")

def create_from_deduplicated_tsv(input_tsv, output_dir):
    """Create source.txt and target.txt files from a deduplicated aligned_pairs.tsv file."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_tsv):
        print(f"Error: {input_tsv} not found!")
        return
    
    print(f"Reading deduplicated pairs from {input_tsv}...")
    
    pairs = []
    with open(input_tsv, 'r', encoding='utf-8') as f:
        header = f.readline()
        
        for line in tqdm(f, desc="Reading pairs"):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pairs.append((parts[1], parts[2]))
    
    print(f"Found {len(pairs)} deduplicated pairs")
    
    print(f"Writing {len(pairs)} pairs to source and target files...")
    with open(f"{output_dir}/source.txt", 'w', encoding='utf-8') as source_file, \
         open(f"{output_dir}/target.txt", 'w', encoding='utf-8') as target_file:
        
        for source, target in tqdm(pairs, desc="Writing files"):
            source_file.write(f"{source}\n")
            target_file.write(f"{target}\n")
    
    print("\nVerifying alignment...")
    with open(f"{output_dir}/source.txt", 'r', encoding='utf-8') as source_file, \
         open(f"{output_dir}/target.txt", 'r', encoding='utf-8') as target_file:
        source_lines = source_file.readlines()
        target_lines = target_file.readlines()
        
        if len(source_lines) == len(target_lines):
            print(f"✓ ALIGNMENT VERIFIED: Both files have exactly {len(source_lines)} lines.")
            
            check_positions = [
                (0, "First line"),
                (len(source_lines)//4, "25% position"),
                (len(source_lines)//2, "Middle"),
                (3*len(source_lines)//4, "75% position"),
                (len(source_lines)-1, "Last line")
            ]
            
            print("\nSample alignment checks:")
            print("-" * 60)
            for pos, desc in check_positions:
                print(f"{desc}:")
                print(f"  Source: {source_lines[pos].strip()}")
                print(f"  Target: {target_lines[pos].strip()}")
                print("-" * 60)
        else:
            print(f"× ALIGNMENT ERROR: source.txt has {len(source_lines)} lines, but target.txt has {len(target_lines)} lines.")
    
    orig_lang_codes = os.path.dirname(input_tsv) + '/language_codes.csv'
    if os.path.exists(orig_lang_codes):
        import shutil
        shutil.copy2(orig_lang_codes, f"{output_dir}/language_codes.csv")
        print(f"Copied language_codes.csv to {output_dir}")
    
    print(f"\nDone! Created deduplicated training data files:")
    print(f"- {output_dir}/source.txt: contains {len(pairs)} examples with format: 'iso_code glosses'")
    print(f"- {output_dir}/target.txt: contains {len(pairs)} examples with translations")
    
    with open(f"{output_dir}/alignment_report.txt", 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("DEDUPLICATED ALIGNMENT REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total deduplicated pairs: {len(pairs)}\n\n")
        f.write("Files created:\n")
        f.write("- source.txt: Contains source sentences with ISO language tags in format 'iso_code glosses'\n")
        f.write("- target.txt: Contains target translations\n\n")
        f.write("Input files used:\n")
        f.write(f"- {input_tsv}: Contains deduplicated source-target pairs\n\n")
        f.write("Format explanation:\n")
        f.write("The source.txt file contains lines in the format 'iso_code glosses', where:\n")
        f.write("- iso_code: 3-letter ISO 639-3 code for the language (e.g., 'tur' for Turkish)\n")
        f.write("- glosses: The glosses for the example\n\n")
        f.write("The target.txt file contains the corresponding translations in English.\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create source and target files for GlossLM training')
    parser.add_argument('--from-tsv', action='store_true', 
                        help='Create files from a deduplicated aligned_pairs.tsv')
    parser.add_argument('--input', type=str, default='training_data_100k_tr_odin/aligned_pairs.tsv',
                        help='Path to the deduplicated aligned_pairs.tsv file')
    parser.add_argument('--output', type=str, default='training_data_100k_tr_odin_dedup',
                        help='Directory to save the output files')
    
    args = parser.parse_args()
    
    if args.from_tsv:
        create_from_deduplicated_tsv(args.input, args.output)
    else:
        # Original functionality
        fix_alignment() 