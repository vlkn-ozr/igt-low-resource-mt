#!/usr/bin/env python3
import json
import sys
import os
import numpy as np
import torch
from comet import download_model, load_from_checkpoint

def main():
    # Check if input file is provided
    if len(sys.argv) != 2:
        print("Usage: python calculate_xcomet.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Generate output filename
    basename = os.path.basename(input_file)
    output_file = f"xcomet_scored_{basename}"
    
    print(f"Processing file: {input_file}")
    
    # Load the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Calculate XCOMET scores
    try:
        calculate_xcomet(data)
        print("\nXCOMET scores have been calculated and added to each item.")
    except Exception as e:
        print(f"\nError calculating XCOMET scores: {e}")
        print("Failed to calculate XCOMET scores.")
        sys.exit(1)

    # Calculate average XCOMET scores
    calculate_averages(data)
    
    # Reorganize data to put summary at the beginning
    reorganized_data = {}
    
    # First add summary if it exists
    if 'summary' in data:
        reorganized_data['summary'] = data['summary']
    
    # Add system-level scores if they exist
    for key in ['zero_shot_system_comet', 'few_shot_system_comet', 'advanced_few_shot_system_comet']:
        if key in data:
            reorganized_data[key] = data[key]
    
    # Add any other keys except results
    for key in data:
        if key not in reorganized_data and key != 'results':
            reorganized_data[key] = data[key]
    
    # Finally add results
    if 'results' in data:
        reorganized_data['results'] = data['results']
    
    # Write the updated data back to the output file
    with open(output_file, 'w') as f:
        json.dump(reorganized_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

def calculate_xcomet(data):
    """Calculate XCOMET scores for translations"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    print("Downloading and loading XCOMET-XL model...")
    try:
        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)
        print("Successfully loaded XCOMET-XL model")
    except Exception as e:
        print(f"Error loading XCOMET-XL model: {e}")
        print("Falling back to standard COMET model...")
        model_path = download_model("wmt20-comet-da")
        model = load_from_checkpoint(model_path)
        print("Successfully loaded standard COMET model")
    
    zero_shot_data = []
    few_shot_data = []
    advanced_few_shot_data = []
    
    for i, item in enumerate(data['results']):
        source = item.get('input', '')
        reference = item.get('expected', '')
        
        if 'zero_shot' in item:
            zero_shot_data.append({
                "src": source,
                "mt": item['zero_shot'],
                "ref": reference
            })
            
        if 'few_shot' in item:
            few_shot_data.append({
                "src": source,
                "mt": item['few_shot'],
                "ref": reference
            })
            
        if 'advanced_few_shot' in item:
            advanced_few_shot_data.append({
                "src": source,
                "mt": item['advanced_few_shot'],
                "ref": reference
            })
    
    batch_size = 6
    
    print(f"Calculating scores for {len(zero_shot_data)} zero-shot translations...")
    if zero_shot_data:
        model_output = model.predict(zero_shot_data, batch_size=batch_size, gpus=0 if device == "cpu" else 1)
        
        zero_shot_scores = None
        
        if hasattr(model_output, 'scores'):
            zero_shot_scores = model_output.scores
        elif isinstance(model_output, dict) and 'scores' in model_output:
            zero_shot_scores = model_output['scores']
        else:
            zero_shot_scores = model_output
        
        for i in range(len(zero_shot_scores)):
            item_idx = i
            if item_idx < len(data['results']) and 'zero_shot' in data['results'][item_idx]:
                score_value = zero_shot_scores[i]
                if isinstance(score_value, list):
                    score_value = score_value[0]
                data['results'][item_idx]['zero_shot_comet'] = round(float(score_value), 4)
                
                if hasattr(model_output, 'metadata') and hasattr(model_output.metadata, 'error_spans') and i < len(model_output.metadata.error_spans):
                    data['results'][item_idx]['zero_shot_error_spans'] = model_output.metadata.error_spans[i]
        
        if hasattr(model_output, 'system_score'):
            data['zero_shot_system_comet'] = round(float(model_output.system_score), 4)
        else:
            flat_scores = [s[0] if isinstance(s, list) else s for s in zero_shot_scores]
            data['zero_shot_system_comet'] = round(float(np.mean(flat_scores)), 4)
    
    print(f"Calculating scores for {len(few_shot_data)} few-shot translations...")
    if few_shot_data:
        model_output = model.predict(few_shot_data, batch_size=batch_size, gpus=0 if device == "cpu" else 1)
        
        few_shot_scores = None
        
        if hasattr(model_output, 'scores'):
            few_shot_scores = model_output.scores
        elif isinstance(model_output, dict) and 'scores' in model_output:
            few_shot_scores = model_output['scores']
        else:
            few_shot_scores = model_output
        
        for i in range(len(few_shot_scores)):
            item_idx = i
            if item_idx < len(data['results']) and 'few_shot' in data['results'][item_idx]:
                score_value = few_shot_scores[i]
                if isinstance(score_value, list):
                    score_value = score_value[0]
                data['results'][item_idx]['few_shot_comet'] = round(float(score_value), 4)
                
                if hasattr(model_output, 'metadata') and hasattr(model_output.metadata, 'error_spans') and i < len(model_output.metadata.error_spans):
                    data['results'][item_idx]['few_shot_error_spans'] = model_output.metadata.error_spans[i]
        
        if hasattr(model_output, 'system_score'):
            data['few_shot_system_comet'] = round(float(model_output.system_score), 4)
        else:
            flat_scores = [s[0] if isinstance(s, list) else s for s in few_shot_scores]
            data['few_shot_system_comet'] = round(float(np.mean(flat_scores)), 4)
    
    print(f"Calculating scores for {len(advanced_few_shot_data)} advanced few-shot translations...")
    if advanced_few_shot_data:
        model_output = model.predict(advanced_few_shot_data, batch_size=batch_size, gpus=0 if device == "cpu" else 1)
        
        advanced_few_shot_scores = None
        
        if hasattr(model_output, 'scores'):
            advanced_few_shot_scores = model_output.scores
        elif isinstance(model_output, dict) and 'scores' in model_output:
            advanced_few_shot_scores = model_output['scores']
        else:
            advanced_few_shot_scores = model_output
        
        for i in range(len(advanced_few_shot_scores)):
            item_idx = i
            if item_idx < len(data['results']) and 'advanced_few_shot' in data['results'][item_idx]:
                score_value = advanced_few_shot_scores[i]
                if isinstance(score_value, list):
                    score_value = score_value[0]
                data['results'][item_idx]['advanced_few_shot_comet'] = round(float(score_value), 4)
                
                if hasattr(model_output, 'metadata') and hasattr(model_output.metadata, 'error_spans') and i < len(model_output.metadata.error_spans):
                    data['results'][item_idx]['advanced_few_shot_error_spans'] = model_output.metadata.error_spans[i]
        
        if hasattr(model_output, 'system_score'):
            data['advanced_few_shot_system_comet'] = round(float(model_output.system_score), 4)
        else:
            flat_scores = [s[0] if isinstance(s, list) else s for s in advanced_few_shot_scores]
            data['advanced_few_shot_system_comet'] = round(float(np.mean(flat_scores)), 4)

def calculate_averages(data):
    """Calculate average COMET scores"""
    zero_shot_comet_scores = [item.get('zero_shot_comet', 0) for item in data['results'] if 'zero_shot_comet' in item]
    few_shot_comet_scores = [item.get('few_shot_comet', 0) for item in data['results'] if 'few_shot_comet' in item]
    advanced_few_shot_comet_scores = [item.get('advanced_few_shot_comet', 0) for item in data['results'] if 'advanced_few_shot_comet' in item]
    
    zero_shot_comet_avg = np.mean(zero_shot_comet_scores) if zero_shot_comet_scores else 0
    few_shot_comet_avg = np.mean(few_shot_comet_scores) if few_shot_comet_scores else 0
    advanced_few_shot_comet_avg = np.mean(advanced_few_shot_comet_scores) if advanced_few_shot_comet_scores else 0
    
    summary = {
        'sample_count': len(data['results']),
        'zero_shot_avg_comet': round(zero_shot_comet_avg, 4),
        'few_shot_avg_comet': round(few_shot_comet_avg, 4),
        'advanced_few_shot_avg_comet': round(advanced_few_shot_comet_avg, 4)
    }
    
    if 'summary' in data:
        data['summary'].update(summary)
    else:
        data['summary'] = summary
    
    print(f"\nNumber of samples with scores: {len(data['results'])}")
    
    print("\nAverage COMET scores (segment-level):")
    print(f"Zero-shot: {zero_shot_comet_avg:.4f}")
    print(f"Few-shot: {few_shot_comet_avg:.4f}")
    print(f"Advanced few-shot: {advanced_few_shot_comet_avg:.4f}")
    
    if 'zero_shot_system_comet' in data:
        print("\nSystem-level COMET scores:")
        print(f"Zero-shot: {data['zero_shot_system_comet']:.4f}")
        print(f"Few-shot: {data['few_shot_system_comet']:.4f}")
        print(f"Advanced few-shot: {data['advanced_few_shot_system_comet']:.4f}")

if __name__ == "__main__":
    main() 
