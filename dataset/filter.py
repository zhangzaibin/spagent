#!/usr/bin/env python3
"""
Filter script to extract Relative_Depth_BLINK entries from BLINK_All_Tasks.jsonl
"""

import json
import os
from pathlib import Path

def filter_relative_depth_blink(input_file, output_file):
    """
    Filter the JSONL file to extract only entries with 'Relative_Depth_BLINK' in their ID.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
    """
    filtered_count = 0
    total_count = 0
    
    print(f"Filtering {input_file} for Relative_Depth_BLINK entries...")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            total_count += 1
            
            try:
                # Parse JSON line
                data = json.loads(line)
                
                # Check if ID contains 'Relative_Depth_BLINK'
                if 'id' in data and 'Multi-view_Reasoning_BLINK' in data['id']:
                    # Write the filtered entry to output file
                    outfile.write(line + '\n')
                    filtered_count += 1
                    
                    # Print progress every 10 entries
                    if filtered_count % 10 == 0:
                        print(f"  Found {filtered_count} Multi-view_Reasoning_BLINK entries...")
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                continue
    
    print(f"\nFiltering completed!")
    print(f"Total entries processed: {total_count}")
    print(f"Multi-view_Reasoning_BLINK entries found: {filtered_count}")
    print(f"Output saved to: {output_file}")

def main():
    """Main function to run the filter."""
    # File paths
    current_dir = Path(__file__).parent
    input_file = current_dir / "BLINK_All_Tasks.jsonl"
    output_file = current_dir / "Multi-view_Reasoning_BLINK_subset.jsonl"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Filter the data
    filter_relative_depth_blink(str(input_file), str(output_file))
    
    # Show file size comparison
    if output_file.exists():
        input_size = input_file.stat().st_size / (1024 * 1024)  # MB
        output_size = output_file.stat().st_size / (1024 * 1024)  # MB
        
        print(f"\nFile size comparison:")
        print(f"Input file: {input_size:.2f} MB")
        print(f"Output file: {output_size:.2f} MB")
        print(f"Size reduction: {((input_size - output_size) / input_size * 100):.1f}%")

if __name__ == "__main__":
    main()

