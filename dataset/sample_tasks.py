#!/usr/bin/env python3
"""
Sampling script to extract N random samples from each task in BLINK_All_Tasks.jsonl
"""

import json
import os
import random
import argparse
from collections import defaultdict
from pathlib import Path

def sample_tasks(input_file, output_file, samples_per_task):
    """
    Sample N entries from each task type in the JSONL file.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
        samples_per_task (int): Number of samples to take from each task
    """
    # Dictionary to store entries by task type
    task_entries = defaultdict(list)
    total_count = 0
    
    print(f"Reading {input_file} and grouping by task...")
    
    # First pass: Read and group entries by task type
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            total_count += 1
            
            try:
                # Parse JSON line
                data = json.loads(line)
                
                # Extract task type from id
                if 'id' in data:
                    task_type = data['id'].split('_')[0]  # Assuming format like "TaskType_XXX"
                    task_entries[task_type].append(line)
                    
                if total_count % 1000 == 0:
                    print(f"  Processed {total_count} entries...")
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                continue
    
    print(f"\nFound {len(task_entries)} different task types.")
    
    # Second pass: Sample and write to output
    with open(output_file, 'w', encoding='utf-8') as outfile:
        total_sampled = 0
        
        for task_type, entries in task_entries.items():
            # Calculate how many samples to take (minimum of available entries or requested samples)
            n_samples = min(samples_per_task, len(entries))
            sampled_entries = random.sample(entries, n_samples)
            
            # Write sampled entries
            for entry in sampled_entries:
                outfile.write(entry + '\n')
                total_sampled += 1
                
            print(f"Sampled {n_samples} entries from task type: {task_type}")
    
    print(f"\nSampling completed!")
    print(f"Total entries processed: {total_count}")
    print(f"Total entries sampled: {total_sampled}")
    print(f"Output saved to: {output_file}")

def main():
    """Main function to run the sampler."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sample N entries from each task type in a JSONL file.')
    parser.add_argument('--input', type=str, default='BLINK_All_Tasks.jsonl',
                      help='Input JSONL file path')
    parser.add_argument('--output', type=str, default='Sampled_Tasks_3.jsonl',
                      help='Output JSONL file path')
    parser.add_argument('--samples', type=int, default=3,
                      help='Number of samples to take from each task type')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute using script location
    current_dir = Path(__file__).parent
    input_file = current_dir / args.input
    output_file = current_dir / args.output
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Sample the data
    sample_tasks(str(input_file), str(output_file), args.samples)
    
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