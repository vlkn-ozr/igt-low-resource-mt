import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define Turkic languages
turkic_languages = {
    "Turkish", "Northern Uzbek", "Tatar", "Chuvash", "Kazakh", "Uzbek", 
    "Bashkir", "Khakas", "Salar", "Nogai", "Karachay-Balkar", "Mrassu Shor-Tutal",
    "Khalaj Turkic", "Northern Altai", "Lower Chulym Turkic", "Dolgan", "Ili Turki",
    "Taimyr Pidgin Russian", "Tofa", "Turkmen", "Kur-Urmi", "Upper Ussuri", "Southern Altai"
}

# Load the dataset
print("Loading dataset from raw_data/glosslm_train.parquet...")
if not os.path.exists('raw_data/glosslm_train.parquet'):
    print("Parquet file not found. Run download_glosslm_via_api.py first.")
    exit(1)

data_df = pd.read_parquet('raw_data/glosslm_train.parquet')
print(f"Dataset loaded with {len(data_df)} examples")

# Define what counts as unsegmented
print("Identifying unsegmented examples...")
# Consider both empty string and 'no' as unsegmented
data_df['is_unsegmented'] = data_df['is_segmented'].apply(lambda x: x == '' or x == 'no')

# Count unsegmented examples
total_unsegmented = data_df['is_unsegmented'].sum()
print(f"Total unsegmented examples in train split: {total_unsegmented}")

# Group by language
print("Grouping examples by language...")
language_groups = {}
language_stats = defaultdict(lambda: {'total': 0, 'unsegmented': 0, 'segmented': 0})

# Process each example
for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing examples"):
    language = row['language']
    is_unsegmented = row['is_unsegmented']
    
    # Update language statistics
    language_stats[language]['total'] += 1
    if is_unsegmented:
        language_stats[language]['unsegmented'] += 1
    else:
        language_stats[language]['segmented'] += 1
    
    # Group by language
    if language not in language_groups:
        language_groups[language] = {'unsegmented': [], 'segmented': []}
    
    if is_unsegmented:
        language_groups[language]['unsegmented'].append(row.to_dict())
    else:
        language_groups[language]['segmented'].append(row.to_dict())

# Create the subset
print("Creating subset...")
subset_data = []

# For each language, include examples in the subset
for language, group in tqdm(language_groups.items(), desc="Selecting examples"):
    unsegmented_examples = group['unsegmented']
    
    if language in turkic_languages:
        # For Turkic languages, keep all unsegmented examples
        subset_data.extend(unsegmented_examples)
        print(f"{language}: Added {len(unsegmented_examples)} unsegmented examples (100%)")
    else:
        # For non-Turkic languages, keep 42% of unsegmented examples
        sample_size = max(1, int(len(unsegmented_examples) * 0.42))
        sampled = random.sample(unsegmented_examples, min(sample_size, len(unsegmented_examples)))
        subset_data.extend(sampled)

# Create DataFrame from the subset data
print("Preparing final dataset...")
required_fields = ['transcription', 'glosses', 'translation', 'source', 'language']
final_subset = []

for item in subset_data:
    example = {field: item.get(field, '') for field in required_fields}
    final_subset.append(example)

df_subset = pd.DataFrame(final_subset)

# Calculate statistics for the report
print("Calculating statistics...")
subset_lang_counts = df_subset['language'].value_counts().to_dict()
turkic_count = sum(subset_lang_counts.get(lang, 0) for lang in turkic_languages)
non_turkic_count = len(df_subset) - turkic_count

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
    f.write(f"Total examples in subset: {len(df_subset)}\n")
    f.write(f"Total Turkic examples: {turkic_count}\n")
    f.write(f"Total non-Turkic examples (42% sample): {non_turkic_count}\n")
    f.write(f"Number of languages: {len(subset_lang_counts)}\n\n")
    
    f.write("Turkic Language Statistics:\n")
    f.write("-------------------------\n")
    for lang in sorted(turkic_languages):
        if lang in subset_lang_counts:
            f.write(f"{lang}:\n")
            f.write(f"  Total unsegmented examples in subset: {subset_lang_counts[lang]}\n")
            f.write(f"  Original unsegmented examples: {language_stats.get(lang, {}).get('unsegmented', 'N/A')}\n")
            f.write(f"  Percentage of original: {100.0 if language_stats.get(lang, {}).get('unsegmented', 0) == 0 else subset_lang_counts[lang] / language_stats.get(lang, {}).get('unsegmented', 1) * 100:.1f}%\n")
            f.write("\n")
    
    f.write("Non-Turkic Language Statistics (42% sample):\n")
    f.write("------------------------------------------\n")
    for lang, count in sorted(
        [(l, c) for l, c in subset_lang_counts.items() if l not in turkic_languages], 
        key=lambda x: x[1], 
        reverse=True
    )[:20]:  # Show top 20 non-Turkic languages
        f.write(f"{lang}:\n")
        f.write(f"  Total examples: {count}\n")
        f.write(f"  Original unsegmented examples: {language_stats.get(lang, {}).get('unsegmented', 'N/A')}\n")
        f.write(f"  Percentage of original: {42.0 if language_stats.get(lang, {}).get('unsegmented', 0) == 0 else count / language_stats.get(lang, {}).get('unsegmented', 1) * 100:.1f}%\n")
        f.write("\n")
    f.write("... and more languages (truncated for brevity)\n")

print("Done! Files created:")
print("1. glosslm_subset.csv - Contains the subset with all unsegmented Turkic examples")
print("2. subset_statistics.txt - Contains detailed statistics") 