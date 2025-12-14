import pandas as pd
import sys
from datetime import datetime

# Define the languages to keep
LANGUAGES_TO_KEEP = {'arp', 'cmn'}  # Arapaho and Mandarin Chinese

# Define fieldwork languages to remove (from the analysis)
FIELDWORK_LANGUAGES_TO_REMOVE = {
    'ain',  # Ainu
    'rmn',  # Balkan Romani  
    'bej',  # Beja
    'boa',  # Bora
    'dol',  # Dolgan (using dol instead of dolg1241)
    'evn',  # Evenki
    'kha',  # Jinghpaw
    'klf',  # Kalamang
    'pnr',  # Pnar
    'sanz', # Sanzhi Dargwa (using sanz instead of sanz1248)
    'sso',  # Savosavo
    'sel',  # Selkup
    'smw',  # Sumbawa
    'sme',  # Sümi
    'mpo',  # Texistepec Popoluca
    'vre'   # Vera'a
}

def filter_source_target_files():
    """Filter source and target files to keep only Arapaho and Mandarin samples"""
    
    print("Loading source and target files...")
    
    # Read source and target files
    source_file = "training_data_100k_glosslm_dedup/source.txt"
    target_file = "training_data_100k_glosslm_dedup/target.txt"
    
    with open(source_file, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()
    
    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()
    
    print(f"Original files have {len(source_lines)} lines each")
    
    # Check alignment
    if len(source_lines) != len(target_lines):
        print("Error: Source and target files have different numbers of lines!")
        return
    
    # Filter lines
    filtered_source = []
    filtered_target = []
    kept_count = 0
    removed_count = 0
    
    print("Filtering lines...")
    for i, (source_line, target_line) in enumerate(zip(source_lines, target_lines)):
        source_line = source_line.strip()
        target_line = target_line.strip()
        
        # Skip empty lines
        if not source_line or not target_line:
            continue
        
        # Extract ISO code from source line (first word)
        parts = source_line.split()
        if not parts:
            continue
        
        iso_code = parts[0].lower()
        
        # Check if this is a language we want to keep
        if iso_code in LANGUAGES_TO_KEEP:
            filtered_source.append(source_line)
            filtered_target.append(target_line)
            kept_count += 1
        # Check if this is a fieldwork language we want to remove
        elif iso_code in FIELDWORK_LANGUAGES_TO_REMOVE:
            removed_count += 1
            # Skip this line (don't add to filtered lists)
        else:
            # Keep non-fieldwork languages
            filtered_source.append(source_line)
            filtered_target.append(target_line)
            kept_count += 1
    
    print(f"Kept {kept_count} lines")
    print(f"Removed {removed_count} fieldwork language lines")
    
    # Create output directory
    output_dir = "training_data_100k_glosslm_filtered"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Write filtered files
    output_source = f"{output_dir}/source.txt"
    output_target = f"{output_dir}/target.txt"
    
    with open(output_source, 'w', encoding='utf-8') as f:
        for line in filtered_source:
            f.write(line + '\n')
    
    with open(output_target, 'w', encoding='utf-8') as f:
        for line in filtered_target:
            f.write(line + '\n')
    
    print(f"Filtered files saved to:")
    print(f"  {output_source}")
    print(f"  {output_target}")
    
    # Create a summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"filtering_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("FIELDWORK LANGUAGE FILTERING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original lines: {len(source_lines)}\n")
        f.write(f"Kept lines: {kept_count}\n")
        f.write(f"Removed fieldwork lines: {removed_count}\n")
        f.write(f"Remaining lines: {len(filtered_source)}\n\n")
        
        f.write("Languages kept:\n")
        f.write("- Arapaho (arp)\n")
        f.write("- Mandarin Chinese (cmn)\n")
        f.write("- All non-fieldwork languages\n\n")
        
        f.write("Fieldwork languages removed:\n")
        for lang in sorted(FIELDWORK_LANGUAGES_TO_REMOVE):
            f.write(f"- {lang}\n")
    
    print(f"Filtering report saved to: {report_file}")

if __name__ == "__main__":
    filter_source_target_files() 