import requests
import pandas as pd
import random
import json
import os
from tqdm import tqdm
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)

# Define Turkic languages
turkic_languages = {
    "Turkish", "Northern Uzbek", "Tatar", "Chuvash", "Kazakh", "Uzbek", 
    "Bashkir", "Khakas", "Salar", "Nogai", "Karachay-Balkar", "Mrassu Shor-Tutal",
    "Khalaj Turkic", "Northern Altai", "Lower Chulym Turkic", "Dolgan", "Ili Turki",
    "Taimyr Pidgin Russian", "Tofa", "Turkmen", "Kur-Urmi", "Upper Ussuri", "Southern Altai"
}

# Function to download dataset
def download_glosslm_dataset():
    """Download the GlossLM dataset from HuggingFace using the datasets library"""
    print("Downloading GlossLM corpus-split dataset from HuggingFace...")
    
    # Create a directory to store the raw data
    os.makedirs('raw_data', exist_ok=True)
    
    try:
        # Load the train split from the glosslm-corpus-split dataset
        dataset = load_dataset("lecslab/glosslm-corpus-split", split="train")
        print(f"Dataset loaded with {len(dataset)} examples in train split")
        
        # Convert to pandas DataFrame
        data_df = dataset.to_pandas()
        
        # Save the train split as parquet
        parquet_file = 'raw_data/glosslm_train.parquet'
        data_df.to_parquet(parquet_file, index=False)
        print(f"Saved train split to {parquet_file}")
        
        return data_df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

# Try to download via datasets library
try:
    data_df = download_glosslm_dataset()
    if data_df is None:
        raise ValueError("Failed to download dataset")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Trying to use cached parquet file if it exists...")
    
    # Check if we already have a downloaded parquet file
    if os.path.exists('raw_data/glosslm_train.parquet'):
        data_df = pd.read_parquet('raw_data/glosslm_train.parquet')
        print(f"Loaded cached train split with {len(data_df)} examples")
    else:
        print("No cached data available. Please check your internet connection and try again.")
        exit(1)

# Initialize counters and data structures
turkic_unsegmented = []
non_turkic_unsegmented_sample = []
lang_stats = {}

# Process each example in the dataset
print("Processing dataset...")
for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
    # Convert row to dictionary for easier processing
    example = row.to_dict()
    
    # Check if example is unsegmented (is_segmented is empty or 'no')
    is_segmented = example.get('is_segmented', '')
    is_unsegmented = (is_segmented == '' or is_segmented == 'no')
    language = example.get('language', '')
    
    # Track language statistics
    if language not in lang_stats:
        lang_stats[language] = {
            'total': 0,
            'unsegmented': 0,
            'segmented': 0
        }
    
    lang_stats[language]['total'] += 1
    if is_unsegmented:
        lang_stats[language]['unsegmented'] += 1
    else:
        lang_stats[language]['segmented'] += 1
    
    # Filter for unsegmented examples
    if is_unsegmented:
        # For Turkic languages, keep all unsegmented examples
        if language in turkic_languages:
            turkic_unsegmented.append(example)
        # For non-Turkic languages, use random sampling (42%)
        elif random.random() < 0.42:
            non_turkic_unsegmented_sample.append(example)

# Combine the Turkic and non-Turkic samples
combined_data = turkic_unsegmented + non_turkic_unsegmented_sample

# Create DataFrame from the combined data
print("Creating DataFrame...")
required_fields = ['transcription', 'glosses', 'translation', 'source', 'language']
subset_data = []

for item in combined_data:
    example = {field: item.get(field, '') for field in required_fields}
    subset_data.append(example)

df_subset = pd.DataFrame(subset_data)

# Calculate statistics
turkic_count = len(turkic_unsegmented)
non_turkic_count = len(non_turkic_unsegmented_sample)
total_count = turkic_count + non_turkic_count

# Calculate per-language statistics
language_counts = {}
for example in combined_data:
    lang = example.get('language', '')
    if lang not in language_counts:
        language_counts[lang] = 0
    language_counts[lang] += 1

# Save the subset
print(f"Saving subset with {len(df_subset)} examples...")
df_subset.to_csv('glosslm_subset.csv', index=False)

# Save statistics
print("Saving statistics...")
with open('subset_statistics.txt', 'w') as f:
    f.write("==================================================\n")
    f.write("GLOSSLM SUBSET STATISTICS (TRAIN SPLIT)\n")
    f.write("==================================================\n\n")
    
    f.write("Overall Statistics:\n")
    f.write("------------------\n")
    f.write(f"Total examples in train split: {len(data_df)}\n")
    f.write(f"Total examples in subset: {total_count}\n")
    f.write(f"Total Turkic examples: {turkic_count}\n")
    f.write(f"Total non-Turkic examples (42% sample): {non_turkic_count}\n")
    f.write(f"Number of languages: {len(language_counts)}\n\n")
    
    f.write("Turkic Language Statistics:\n")
    f.write("-------------------------\n")
    for lang in sorted(turkic_languages):
        if lang in language_counts:
            f.write(f"{lang}:\n")
            f.write(f"  Total unsegmented examples in subset: {language_counts[lang]}\n")
            f.write(f"  Original unsegmented examples: {lang_stats.get(lang, {}).get('unsegmented', 'N/A')}\n")
            f.write("\n")
    
    f.write("Non-Turkic Language Statistics (42% sample):\n")
    f.write("------------------------------------------\n")
    for lang, count in sorted(
        [(l, c) for l, c in language_counts.items() if l not in turkic_languages], 
        key=lambda x: x[1], 
        reverse=True
    )[:20]:  # Show top 20 non-Turkic languages
        f.write(f"{lang}:\n")
        f.write(f"  Total examples: {count}\n")
        f.write(f"  Original unsegmented examples: {lang_stats.get(lang, {}).get('unsegmented', 'N/A')}\n")
        f.write("\n")
    f.write("... and more languages (truncated for brevity)\n")

print("Done! Files created:")
print("1. glosslm_subset.csv - Contains the actual subset data from GlossLM train split")
print("2. subset_statistics.txt - Contains detailed statistics")
print("3. raw_data/glosslm_train.parquet - Contains the full train split") 