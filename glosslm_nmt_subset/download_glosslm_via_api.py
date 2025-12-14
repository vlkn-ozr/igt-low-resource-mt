import requests
import pandas as pd
import random
import json
import os
from tqdm import tqdm

random.seed(42)

turkic_languages = {
    "Turkish", "Northern Uzbek", "Tatar", "Chuvash", "Kazakh", "Uzbek", 
    "Bashkir", "Khakas", "Salar", "Nogai", "Karachay-Balkar", "Mrassu Shor-Tutal",
    "Khalaj Turkic", "Northern Altai", "Lower Chulym Turkic", "Dolgan", "Ili Turki",
    "Taimyr Pidgin Russian", "Tofa", "Turkmen", "Kur-Urmi", "Upper Ussuri", "Southern Altai"
}

def download_glosslm_dataset():
    """Download the GlossLM dataset from HuggingFace using the HTTP API"""
    print("Downloading GlossLM dataset from HuggingFace...")
    
    os.makedirs('raw_data', exist_ok=True)
    
    dataset_url = "https://huggingface.co/api/datasets/lecslab/glosslm-corpus"
    response = requests.get(dataset_url)
    
    if response.status_code != 200:
        print(f"Error downloading dataset info: {response.status_code}")
        return None
    
    dataset_info = response.json()
    
    download_url = None
    for file in dataset_info.get('siblings', []):
        if file.get('rfilename', '').endswith('.parquet'):
            download_url = f"https://huggingface.co/datasets/lecslab/glosslm-corpus/resolve/main/{file['rfilename']}"
            break
    
    if not download_url:
        print("Could not find parquet file in dataset")
        return None
    
    parquet_file = 'raw_data/glosslm.parquet'
    print(f"Downloading from {download_url} to {parquet_file}")
    
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(parquet_file, 'wb') as file:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    
    print("Loading parquet file...")
    data = pd.read_parquet(parquet_file)
    print(f"Dataset loaded with {len(data)} examples")
    
    return data

try:
    data_df = download_glosslm_dataset()
    if data_df is None:
        raise ValueError("Failed to download dataset")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Trying to use cached parquet file if it exists...")
    
    if os.path.exists('raw_data/glosslm.parquet'):
        data_df = pd.read_parquet('raw_data/glosslm.parquet')
        print(f"Loaded cached dataset with {len(data_df)} examples")
    else:
        print("No cached data available. Please check your internet connection and try again.")
        exit(1)

turkic_unsegmented = []
non_turkic_unsegmented_sample = []
lang_stats = {}

print("Processing dataset...")
for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
    example = row.to_dict()
    
    is_unsegmented = (example.get('is_segmented', '') == '')
    language = example.get('language', '')
    
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
    
    if is_unsegmented:
        if language in turkic_languages:
            turkic_unsegmented.append(example)
        elif random.random() < 0.25:
            non_turkic_unsegmented_sample.append(example)

combined_data = turkic_unsegmented + non_turkic_unsegmented_sample

print("Creating DataFrame...")
required_fields = ['transcription', 'glosses', 'translation', 'source', 'language']
subset_data = []

for item in combined_data:
    example = {field: item.get(field, '') for field in required_fields}
    subset_data.append(example)

df_subset = pd.DataFrame(subset_data)

turkic_count = len(turkic_unsegmented)
non_turkic_count = len(non_turkic_unsegmented_sample)
total_count = turkic_count + non_turkic_count

language_counts = {}
for example in combined_data:
    lang = example.get('language', '')
    if lang not in language_counts:
        language_counts[lang] = 0
    language_counts[lang] += 1

print(f"Saving subset with {len(df_subset)} examples...")
df_subset.to_csv('glosslm_subset.csv', index=False)

print("Saving statistics...")
with open('subset_statistics.txt', 'w') as f:
    f.write("==================================================\n")
    f.write("GLOSSLM SUBSET STATISTICS\n")
    f.write("==================================================\n\n")
    
    f.write("Overall Statistics:\n")
    f.write("------------------\n")
    f.write(f"Total examples in subset: {total_count}\n")
    f.write(f"Total Turkic examples: {turkic_count}\n")
    f.write(f"Total non-Turkic examples (25% sample): {non_turkic_count}\n")
    f.write(f"Number of languages: {len(language_counts)}\n\n")
    
    f.write("Turkic Language Statistics:\n")
    f.write("-------------------------\n")
    for lang in sorted(turkic_languages):
        if lang in language_counts:
            f.write(f"{lang}:\n")
            f.write(f"  Total unsegmented examples in subset: {language_counts[lang]}\n")
            f.write(f"  Original unsegmented examples: {lang_stats.get(lang, {}).get('unsegmented', 'N/A')}\n")
            f.write("\n")
    
    f.write("Non-Turkic Language Statistics (25% sample):\n")
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
print("1. glosslm_subset.csv - Contains the actual subset data from GlossLM")
print("2. subset_statistics.txt - Contains detailed statistics") 