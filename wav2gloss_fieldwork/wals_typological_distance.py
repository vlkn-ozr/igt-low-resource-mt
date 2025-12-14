#!/usr/bin/env python3
"""
Script to calculate typological similarity between languages in the fieldwork dataset
and English using WALS (World Atlas of Language Structures) features.
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO

def download_wals_data():
    print("Downloading WALS data...")
    # Download language data
    languages_url = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/languages.csv"
    values_url = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/values.csv"
    parameters_url = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/parameters.csv"
    
    languages_response = requests.get(languages_url)
    values_response = requests.get(values_url)
    parameters_response = requests.get(parameters_url)
    
    languages_df = pd.read_csv(StringIO(languages_response.text))
    values_df = pd.read_csv(StringIO(values_response.text))
    parameters_df = pd.read_csv(StringIO(parameters_response.text))
    
    return languages_df, values_df, parameters_df

def match_language_to_wals(languages_df, language_names):
    name_to_wals = {}
    
    # Create mapping from language name to WALS ID
    for _, row in languages_df.iterrows():
        if pd.notna(row['Name']):
            name_to_wals[row['Name'].lower()] = row['ID']
    
    # Map our dataset's language names to WALS IDs
    dataset_wals_ids = {}
    unmatched_languages = []
    
    for lang_code, lang_name in language_names.items():
        # Try exact match
        if lang_name.lower() in name_to_wals:
            dataset_wals_ids[lang_code] = name_to_wals[lang_name.lower()]
            continue
        
        # Try partial match
        matched = False
        for wals_name in name_to_wals:
            # Check if the language name is contained in the WALS name or vice versa
            if lang_name.lower() in wals_name or wals_name in lang_name.lower():
                dataset_wals_ids[lang_code] = name_to_wals[wals_name]
                print(f"Matched '{lang_name}' to WALS language '{wals_name}'")
                matched = True
                break
        
        if not matched:
            unmatched_languages.append((lang_code, lang_name))
    
    if unmatched_languages:
        print("\nWarning: Could not match the following languages to WALS:")
        for lang_code, lang_name in unmatched_languages:
            print(f"  {lang_name} ({lang_code})")
    
    return dataset_wals_ids

def calculate_typological_similarity(values_df, lang1_id, lang2_id, min_features=1):
    """
    Calculate typological similarity between two languages based on WALS features.
    
    Parameters:
    - values_df: DataFrame with WALS feature values
    - lang1_id: WALS ID for first language (e.g., English)
    - lang2_id: WALS ID for second language
    - min_features: Minimum number of shared features required for reliable comparison
    
    Returns:
    - similarity: Similarity score (0-1, higher = more similar)
    - shared_features: Number of shared features
    - total_features: Total number of features annotated in either language
    """
    # Get features for both languages
    lang1_features = values_df[values_df['Language_ID'] == lang1_id][['Parameter_ID', 'Value']]
    lang2_features = values_df[values_df['Language_ID'] == lang2_id][['Parameter_ID', 'Value']]
    
    # Convert to dictionaries for easier comparison
    lang1_dict = dict(zip(lang1_features['Parameter_ID'], lang1_features['Value']))
    lang2_dict = dict(zip(lang2_features['Parameter_ID'], lang2_features['Value']))
    
    # Find shared features (features annotated in both languages)
    shared_features = set(lang1_dict.keys()) & set(lang2_dict.keys())
    
    # Find total possible features (features annotated in either language)
    total_possible_features = set(lang1_dict.keys()) | set(lang2_dict.keys())
    
    if len(shared_features) < min_features:
        return np.nan, len(shared_features), len(total_possible_features)
    
    # Count matching features (features with the same value)
    matches = sum(lang1_dict[feature] == lang2_dict[feature] for feature in shared_features)
    
    # Calculate similarity as matches divided by total possible features
    similarity = matches / len(total_possible_features)
    
    return similarity, len(shared_features), len(total_possible_features)

def get_language_data():
    iso_codes = []
    glotto_to_iso = {}
    language_names = {}
    
    try:
        with open('fieldwork_iso.txt', 'r') as f:
            lines = f.readlines()
            
        # Skip the header rows (first 2 lines)
        for line in lines[2:]:
            # Extract data from the line
            parts = [part.strip() for part in line.strip().split('|')]
            if len(parts) >= 4:  # Should have at least 4 parts: empty, glottocode, language name, iso code, empty
                glottocode = parts[1].strip()
                lang_name = parts[2].strip()
                iso_code = parts[3].strip()
                
                if iso_code and lang_name:
                    iso_codes.append(iso_code)
                    glotto_to_iso[glottocode] = iso_code
                    language_names[iso_code] = lang_name
    except Exception as e:
        print(f"Error reading fieldwork_iso.txt: {e}")
        return [], {}, {}
    
    return iso_codes, glotto_to_iso, language_names

def normalize_language_name(name):
    # Extract main language name from parentheses
    if '(' in name:
        main_name = name.split('(')[0].strip()
        return main_name
    return name

def main():
    # Get language data from the fieldwork_iso.txt file
    iso_codes, glotto_to_iso, language_names = get_language_data()
    if not language_names:
        print("No language data found. Please check the fieldwork_iso.txt file.")
        return
    
    print(f"Found {len(language_names)} languages in the dataset")
    
    # Normalize language names
    normalized_names = {code: normalize_language_name(name) for code, name in language_names.items()}
    
    # Download WALS data
    languages_df, values_df, parameters_df = download_wals_data()
    print(f"Downloaded WALS data: {len(languages_df)} languages, {len(parameters_df)} features")
    
    # Find English in WALS
    english_rows = languages_df[languages_df['Name'] == 'English']
    if len(english_rows) == 0:
        print("Error: English not found in WALS data")
        return
    english_wals_id = english_rows.iloc[0]['ID']
    print(f"English WALS ID: {english_wals_id}")
    
    # Match language names to WALS IDs
    dataset_wals_ids = match_language_to_wals(languages_df, normalized_names)
    print(f"Matched {len(dataset_wals_ids)} languages to WALS IDs")
    
    # Calculate typological similarity between English and each language
    results = []
    for iso_code, wals_id in dataset_wals_ids.items():
        similarity, shared_features, total_features = calculate_typological_similarity(values_df, english_wals_id, wals_id)
        # Use the language name from WALS, but fall back to our dataset's name if needed
        wals_language_name = languages_df[languages_df['ID'] == wals_id]['Name'].iloc[0]
        dataset_language_name = language_names.get(iso_code, "Unknown")
        
        results.append({
            'ISO_Code': iso_code,
            'WALS_ID': wals_id,
            'WALS_Language': wals_language_name,
            'Dataset_Language': dataset_language_name,
            'Similarity': similarity,
            'Shared_Features': shared_features,
            'Total_Features': total_features
        })
    
    # Convert to DataFrame and sort by similarity (higher = more similar to English)
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Similarity', ascending=False, na_position='last')
        
        # Save results to CSV
        results_df.to_csv('wals_typological_distances.csv', index=False)
        print(f"Results saved to wals_typological_distances.csv")
        
        # Print summary
        print("\nTypological similarities to English (higher = more similar to English):")
        for _, row in results_df.iterrows():
            if pd.isna(row['Similarity']):
                print(f"{row['Dataset_Language']} ({row['ISO_Code']}): insufficient data ({row['Shared_Features']} shared features out of {row['Total_Features']} total)")
            else:
                print(f"{row['Dataset_Language']} ({row['ISO_Code']}): {row['Similarity']:.4f} ({row['Shared_Features']} shared features out of {row['Total_Features']} total)")
    else:
        print("No results found. Check if the WALS IDs were matched correctly.")

if __name__ == "__main__":
    main() 