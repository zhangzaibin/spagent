#!/usr/bin/env python3
"""
Extract Error Cases from BLINK Dataset

This script extracts BLINK dataset entries that correspond to incorrect tool-assisted predictions
from a CSV results file and saves them as a new JSONL file containing the original BLINK data.
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_blink_dataset(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load BLINK dataset from JSONL file
    
    Args:
        jsonl_path: Path to the BLINK dataset JSONL file
        
    Returns:
        Dictionary mapping image paths to dataset entries
    """
    blink_data = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            # Use the first image path as the key for mapping
            if 'image' in entry and entry['image']:
                # Create a key from the first image path
                first_image = entry['image'][0]
                # Extract the filename without extension for matching
                image_key = Path(first_image).stem
                blink_data[image_key] = entry
    
    return blink_data


def extract_error_cases(csv_path: str, blink_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract error cases from CSV results
    
    Args:
        csv_path: Path to the CSV results file
        blink_data: BLINK dataset dictionary
        
    Returns:
        List of BLINK entries corresponding to incorrect predictions
    """
    # Read CSV file with error handling
    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip', engine='python')
    
    # Filter for incorrect predictions (is_correct = 0)
    error_cases = df[df['is_correct'] == 0]
    
    extracted_entries = []
    missing_entries = []
    
    for _, row in error_cases.iterrows():
        # Parse the path column (it contains a list of image paths as string)
        try:
            # The path column contains a string representation of a list
            path_str = row['path']
            # Remove brackets and quotes, then split
            if path_str.startswith('[') and path_str.endswith(']'):
                path_str = path_str[1:-1]  # Remove brackets
            
            # Split by comma and clean up
            image_paths = [p.strip().strip("'\"") for p in path_str.split(',')]
            
            # Get the first image path for matching
            first_image = image_paths[0]
            
            # Extract filename pattern for matching with BLINK data
            # For example: 'dataset/BLINK_images/Art_Style_val_000044_img1.jpg' 
            # should match with entries that have this pattern
            filename = Path(first_image).name
            
            # Try different matching strategies
            matched_entry = None
            
            # Strategy 1: Match by exact filename pattern
            for key, entry in blink_data.items():
                if any(filename in img_path for img_path in entry['image']):
                    matched_entry = entry
                    break
            
            # Strategy 2: Match by task and index pattern if Strategy 1 fails
            if matched_entry is None:
                # Extract task and index from filename
                # Example: Art_Style_val_000044_img1.jpg -> Art_Style, 000044
                parts = filename.split('_')
                if len(parts) >= 4:
                    task_name = parts[0] + '_' + parts[1]  # e.g., "Art_Style"
                    
                    for key, entry in blink_data.items():
                        if entry.get('task') == task_name:
                            # Check if any image path contains the same index
                            if any(parts[3] in img_path for img_path in entry['image']):
                                matched_entry = entry
                                break
            
            if matched_entry:
                # Add CSV row information to the entry for context
                enhanced_entry = matched_entry.copy()
                enhanced_entry['csv_info'] = {
                    'question': row['question'],
                    'predicted_answer': row.get('normalized_prediction', ''),
                    'correct_answer': row.get('normalized_ground_truth', ''),
                    'used_tools': row.get('used_tools', ''),
                    'analysis': row.get('analysis', '')
                }
                extracted_entries.append(enhanced_entry)
            else:
                missing_entries.append({
                    'first_image': first_image,
                    'question': row['question'][:100] + '...' if len(row['question']) > 100 else row['question']
                })
        
        except Exception as e:
            print(f"Error processing row: {e}")
            missing_entries.append({
                'error': str(e),
                'row_data': str(row.to_dict())
            })
    
    print(f"Successfully extracted {len(extracted_entries)} error cases")
    if missing_entries:
        print(f"Could not match {len(missing_entries)} entries:")
        for missing in missing_entries[:5]:  # Show first 5 missing entries
            print(f"  - {missing}")
        if len(missing_entries) > 5:
            print(f"  ... and {len(missing_entries) - 5} more")
    
    return extracted_entries


def save_error_cases(error_cases: List[Dict[str, Any]], output_path: str):
    """
    Save error cases to JSONL file
    
    Args:
        error_cases: List of error case entries
        output_path: Path to save the JSONL file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSONL file (one JSON object per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for case in error_cases:
            # Remove the csv_info we added, keep only original BLINK data
            clean_case = case.copy()
            if 'csv_info' in clean_case:
                del clean_case['csv_info']
            
            json.dump(clean_case, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved {len(error_cases)} error cases to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract error cases from BLINK dataset based on CSV results')
    parser.add_argument('csv_file', help='Path to the CSV results file')
    parser.add_argument('--blink_jsonl', default='dataset/BLINK_All_Tasks.jsonl', 
                       help='Path to BLINK dataset JSONL file (default: dataset/BLINK_All_Tasks.jsonl)')
    parser.add_argument('--output_dir', default='dataset', 
                       help='Output directory for the JSONL file (default: dataset)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    if not os.path.exists(args.blink_jsonl):
        print(f"Error: BLINK JSONL file not found: {args.blink_jsonl}")
        return
    
    # Extract filename from CSV path for output naming
    csv_filename = Path(args.csv_file).stem
    output_filename = f"{csv_filename}_error_cases.jsonl"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Loading BLINK dataset from {args.blink_jsonl}...")
    blink_data = load_blink_dataset(args.blink_jsonl)
    print(f"Loaded {len(blink_data)} BLINK entries")
    
    print(f"Extracting error cases from {args.csv_file}...")
    error_cases = extract_error_cases(args.csv_file, blink_data)
    
    if error_cases:
        save_error_cases(error_cases, output_path)
    else:
        print("No error cases found in the CSV file.")


if __name__ == "__main__":
    main()