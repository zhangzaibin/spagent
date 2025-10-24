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
from spagent.core import DataCollector  # NEW: Import DataCollector
from spagent.models import GPTModel, QwenModel
from spagent.tools import (
    DepthEstimationTool,
    SegmentationTool,
    ObjectDetectionTool,
    SupervisionTool,
    YOLOETool,
    MoondreamTool,
    Pi3Tool
)
from spagent.utils.utils import (
    load_json_data, 
    extract_question_and_answer, 
    normalize_answer, 
    print_evaluation_results, 
    validate_sample_paths,
    save_result_to_csv
)
from spagent_evaluation import evaluate_tool_config, evaluate_single_sample
from datetime import datetime  # NEW: For timestamp
# Define server URLs
TOOL_SERVERS = {
    "depth": "http://0.0.0.0:20019",  # depth-anything-v2
    "segmentation": "http://0.0.0.0:20020",  # sam
    "detection": "http://10.7.8.94:20022",  # dino
    "pi3": "http://0.0.0.0:20030"  # pi3
}

TOOL_CONFIGS = {
    # "baseline_no_tools": [
    #     # Empty tool list - pure LLM baseline
    # ],
    "depth_detection_segmentation": [
        # DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"]),
        # ObjectDetectionTool(use_mock=False, server_url=TOOL_SERVERS["detection"]),
        # SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["segmentation"]),
        Pi3Tool(use_mock=False, server_url=TOOL_SERVERS["pi3"])
    ]
}

def main():
    """Main function"""
    # Configure paths
    parser = argparse.ArgumentParser(description='Depth Anything V2 Server')

    parser.add_argument('--data_path', type=str, default='dataset/cvbench_data.jsonl',
                        help='Path to the data file (default: dataset/cvbench_data.jsonl)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: 5), Set to None for full evaluation')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--image_base_path', type=str, default='dataset',
                        help='Path to the image base directory (default: dataset)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model to use for evaluation (default: gpt-4o-mini)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum number of tool-call iterations (default: 3)')
    
    # NEW: Data collection arguments
    parser.add_argument('--enable_data_collection', action='store_true',
                        help='Enable training data collection')
    parser.add_argument('--data_output_dir', type=str, default=None,
                        help='Directory for training data (auto-generated if not specified)')

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
        # NEW: Create DataCollector if enabled
        data_collector = None
        if args.enable_data_collection:
            if args.data_output_dir is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                data_output_dir = f"training_data/{config_name}_{args.model.replace('-', '_')}_{timestamp}"
            else:
                data_output_dir = args.data_output_dir
            
            data_collector = DataCollector(
                output_dir=data_output_dir,
                save_images=True,
                auto_save=True
            )
            print(f"✓ Data collection enabled: {data_output_dir}")
        
        results = evaluate_tool_config(
            config_name=config_name,
            tools=tools,
            data_path=args.data_path,
            image_base_path=args.image_base_path,
            model=args.model,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            max_iterations=args.max_iterations,
            data_collector=data_collector  # NEW: Pass DataCollector
        )
        all_results[config_name] = results
        
        # Print individual config results
        print(f"\nResults for {config_name}:")
        print_evaluation_results(results)
    
    # Save all results to file
    output_file = f"spagent_evaluation_results_{args.model.replace('-', '_')}_{args.max_iterations}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    main()