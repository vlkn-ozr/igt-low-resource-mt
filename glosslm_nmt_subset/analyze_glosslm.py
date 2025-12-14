import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm

def analyze_glosslm():
    # Load the dataset from parquet file
    print("Loading GlossLM dataset from raw_data/glosslm.parquet...")
    if not os.path.exists('raw_data/glosslm.parquet'):
        print("Parquet file not found. Run download_glosslm_via_api.py first.")
        return
    
    dataset = pd.read_parquet('raw_data/glosslm.parquet')
    print(f"Dataset loaded with {len(dataset)} examples")
    
    language_stats = defaultdict(lambda: {
        'total': 0,
        'unsegmented': 0,
        'segmented': 0
    })
    
    total_stats = {
        'total': 0,
        'unsegmented': 0,
        'segmented': 0,
        'languages': 0
    }
    
    # Analyze the dataset
    print("Analyzing dataset...")
    for _, item in tqdm(dataset.iterrows(), total=len(dataset)):
        lang = item['language']
        # Update language-specific stats
        language_stats[lang]['total'] += 1
        if item['is_segmented'] == 'yes':
            language_stats[lang]['segmented'] += 1
            total_stats['segmented'] += 1
        else:  # empty string or 'no' means unsegmented
            language_stats[lang]['unsegmented'] += 1
            total_stats['unsegmented'] += 1
        total_stats['total'] += 1
    
    total_stats['languages'] = len(language_stats)
    
    print("\n" + "="*50)
    print("GLOSSLM DATASET SUMMARY")
    print("="*50)
    print(f"Total number of languages: {total_stats['languages']}")
    print(f"Total number of examples: {total_stats['total']}")
    print(f"Total segmented examples: {total_stats['segmented']}")
    print(f"Total unsegmented examples: {total_stats['unsegmented']} (includes both empty string and 'no' values)")
    print("="*50 + "\n")
    
    with open('glosslm_total_stats.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write("GLOSSLM DATASET SUMMARY\n")
        f.write("="*50 + "\n")
        f.write("Note: The 'is_segmented' field in GlossLM can contain three values:\n")
        f.write("      - 'yes': indicates the example is segmented\n")
        f.write("      - '': empty string, indicates the example is unsegmented\n") 
        f.write("      - 'no': also indicates the example is unsegmented\n\n")
        f.write(f"Total number of languages: {total_stats['languages']}\n")
        f.write(f"Total number of examples: {total_stats['total']}\n")
        f.write(f"Total segmented examples: {total_stats['segmented']}\n")
        f.write(f"Total unsegmented examples: {total_stats['unsegmented']} (includes both empty string and 'no' values)\n")
        f.write("="*50 + "\n")
    print("Total statistics have been saved to 'glosslm_total_stats.txt'")
    
    df = pd.DataFrame.from_dict(language_stats, orient='index')
    df = df.sort_values(['total', 'segmented'], ascending=[False, False])
    
    print("Per-Language Statistics:")
    print("=" * 80)
    print(f"{'Language':<30} {'Total':<10} {'Unsegmented':<15} {'Segmented':<15}")
    print("-" * 80)
    for lang, stats in df.iterrows():
        print(f"{lang:<30} {stats['total']:<10} {stats['unsegmented']:<15} {stats['segmented']:<15}")
    
    df.to_csv('glosslm_language_stats.csv')
    print("\nDetailed statistics have been saved to 'glosslm_language_stats.csv'")

if __name__ == "__main__":
    analyze_glosslm() 