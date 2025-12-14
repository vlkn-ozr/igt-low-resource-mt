#!/usr/bin/env python3
"""
Script to download the wav2gloss/fieldwork dataset from Hugging Face.
Downloads only transcription, gloss, and translation columns (no audio files).
"""

import os
from datasets import load_dataset

def download_wav2gloss_dataset():
    """Download the wav2gloss/fieldwork dataset with only text columns."""
    dataset_name = "wav2gloss/fieldwork"
    print(f"Downloading dataset: {dataset_name}")
    print("Downloading only text columns: transcription, gloss, translation")
    print("(Excluding audio files to reduce size)\n")
    
    try:
        print("Loading dataset...")
        dataset = load_dataset(dataset_name)
        
        print("Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        columns_to_keep = ['transcription', 'gloss', 'translation', 'language']
        
        filtered_dataset = {}
        for split_name, split_data in dataset.items():
            print(f"\nProcessing {split_name.upper()} split...")
            print(f"  - Original columns: {list(split_data.features.keys())}")
            print(f"  - Number of examples: {len(split_data)}")
            
            columns_to_remove = [col for col in split_data.column_names if col not in columns_to_keep]
            filtered_split = split_data.remove_columns(columns_to_remove)
            
            print(f"  - Kept columns: {list(filtered_split.features.keys())}")
            filtered_dataset[split_name] = filtered_split
        
        save_path = "./wav2gloss_fieldwork_text_only"
        print(f"\nSaving filtered dataset to: {save_path}")
        
        for split_name, split_data in filtered_dataset.items():
            split_path = f"{save_path}/{split_name}"
            os.makedirs(split_path, exist_ok=True)
            split_data.save_to_disk(split_path)
        
        print(f"\nDataset successfully downloaded and saved to: {save_path}")
        
        print("\nExample data structure (first 3 items from train split):")
        if 'train' in filtered_dataset:
            for i in range(min(3, len(filtered_dataset['train']))):
                print(f"\nExample {i+1}:")
                example = filtered_dataset['train'][i]
                for key, value in example.items():
                    print(f"  {key}: {str(value)[:150]}{'...' if len(str(value)) > 150 else ''}")
        
        print(f"\nDataset statistics:")
        for split_name, split_data in filtered_dataset.items():
            print(f"{split_name}: {len(split_data)} examples")
        
        return filtered_dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have the 'datasets' library installed:")
        print("pip install datasets")
        return None

if __name__ == "__main__":
    download_wav2gloss_dataset() 