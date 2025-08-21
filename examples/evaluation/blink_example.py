import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm
import cv2
import numpy as np
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent import SPAgent
from spagent.models import GPTModel, QwenModel
from spagent.tools import (
    DepthEstimationTool,
    SegmentationTool,
    ObjectDetectionTool,
    SupervisionTool,
    YOLOETool
)
from spagent.utils.utils import (
    load_blink_data, 
    extract_question_and_answer, 
    normalize_answer, 
    print_evaluation_results, 
    validate_sample_paths,
    save_error_to_tsv
)
from spagent_evaluation import evaluate_tool_config, evaluate_single_sample
# Define server URLs
TOOL_SERVERS = {
    "depth": "http://10.8.131.51:30750",
    "segmentation": "http://10.8.131.51:30646",
    "detection": "http://10.8.131.51:30969"
}

TOOL_CONFIGS = {
    # "baseline_no_tools": [
    #     # Empty tool list - pure LLM baseline
    # ],
    "depth_detection_segmentation": [
        DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"]),
        ObjectDetectionTool(use_mock=False, server_url=TOOL_SERVERS["detection"]),
        SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["segmentation"])
    ]
}

def main():
    """Main function"""
    # Configure paths
    parser = argparse.ArgumentParser(description='Depth Anything V2 Server')

    parser.add_argument('--data_path', type=str, default='dataset/BLINK_All_Tasks.jsonl',
                        help='Path to the data file (default: dataset/BLINK_All_Tasks.jsonl)')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='Maximum number of samples to process (default: 5), Set to None for full evaluation')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--image_base_path', type=str, default='dataset',
                        help='Path to the image base directory (default: dataset)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model to use for evaluation (default: gpt-4o-mini)')

    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        return

    # Run evaluation for each tool configuration
    all_results = {}
    for config_name, tools in TOOL_CONFIGS.items():
        results = evaluate_tool_config(
            config_name=config_name,
            tools=tools,
            data_path=args.data_path,
            image_base_path=args.image_base_path,
            model=args.model,
            max_samples=args.max_samples,
            max_workers=args.max_workers
        )
        all_results[config_name] = results
        
        # Print individual config results
        print(f"\nResults for {config_name}:")
        print_evaluation_results(results)
    
    # Save all results to file
    output_file = f"spagent_evaluation_results_{model.replace('-', '_')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    main()