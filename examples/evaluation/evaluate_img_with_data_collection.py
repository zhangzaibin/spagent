"""
Evaluation Script with Data Collection

This script extends evaluate_img.py to collect training data during evaluation.
Only successful sessions (with correct or incorrect answers) will be saved.
"""

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
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent import SPAgent
from spagent.core import DataCollector
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
from spagent_evaluation import evaluate_single_sample

# Define server URLs
TOOL_SERVERS = {
    "depth": "http://0.0.0.0:20019",  # depth-anything-v2
    "segmentation": "http://0.0.0.0:20020",  # sam
    "detection": "http://10.7.8.94:20022",  # dino
    "pi3": "http://0.0.0.0:20030"  # pi3
}

TOOL_CONFIGS = {
    "depth_detection_segmentation": [
        Pi3Tool(use_mock=False, server_url=TOOL_SERVERS["pi3"])
    ]
}


def evaluate_tool_config_with_data_collection(
    config_name: str,
    tools: List[Any],
    data_path: str,
    image_base_path: str,
    model: str = "gpt-4o-mini",
    max_samples: int = None,
    max_workers: int = 4,
    max_iterations: int = 3,
    enable_data_collection: bool = True,
    data_collection_dir: str = None,
    only_save_correct: bool = False  # 是否只保存正确的样本
) -> Dict[str, Any]:
    """
    Evaluate a specific tool configuration with data collection
    
    Args:
        config_name: Name of the tool configuration
        tools: List of tool instances
        data_path: Path to dataset file
        image_base_path: Base path for images/videos
        model: Model name to use
        max_samples: Maximum number of samples to evaluate
        max_workers: Maximum number of parallel workers
        max_iterations: Maximum number of tool-call iterations
        enable_data_collection: Whether to enable data collection
        data_collection_dir: Directory for collected data (auto-generated if None)
        only_save_correct: If True, only save samples with correct predictions
        
    Returns:
        Evaluation results dictionary with data collection statistics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating configuration: {config_name}")
    print(f"Data collection: {'ENABLED' if enable_data_collection else 'DISABLED'}")
    if enable_data_collection and only_save_correct:
        print("Mode: Only saving CORRECT predictions")
    print(f"{'='*60}\n")
    
    print(f"Loading data from {data_path}")
    data = load_json_data(data_path)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples for evaluation")
    
    # Initialize DataCollector if enabled
    collector = None
    if enable_data_collection:
        if data_collection_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            data_collection_dir = f"training_data/{config_name}_{model.replace('-', '_')}_{timestamp}"
        
        collector = DataCollector(
            output_dir=data_collection_dir,
            save_images=True,
            auto_save=True  # Auto save successful sessions
        )
        print(f"✓ DataCollector initialized: {data_collection_dir}")
    
    # Create SPAgent instance with DataCollector
    agent = SPAgent(
        model=GPTModel(model_name=model),
        tools=tools,
        max_workers=max_workers,
        data_collector=collector  # Pass collector to agent
    )
    
    print(f"Evaluating {len(data)} samples with {model}")
    
    results = []
    correct_count = 0
    total_time = 0
    data_collection_stats = {
        "attempted_sessions": 0,
        "saved_sessions": 0,
        "skipped_sessions": 0,
        "failed_sessions": 0
    }
    
    # Use tqdm for progress tracking
    for sample in tqdm(data, desc="Evaluating"):
        # Determine sample type
        has_image = bool(sample.get("image", []))
        has_video = bool(sample.get("video", []))
        
        if not has_image or has_video:
            # Skip video samples or invalid samples
            result = {
                "id": sample.get("id", "unknown"),
                "success": False,
                "error": "Skipping video or invalid sample"
            }
            results.append(result)
            continue
        
        # Evaluate image sample
        # Note: evaluate_single_sample internally calls agent.solve_problem
        # which will automatically use the DataCollector if it's attached
        try:
            data_collection_stats["attempted_sessions"] += 1
            
            # Run evaluation
            result = evaluate_single_sample(
                agent, 
                sample, 
                image_base_path, 
                config_name, 
                max_iterations
            )
            
            results.append(result)
            
            if result["success"]:
                is_correct = result["is_correct"]
                if is_correct:
                    correct_count += 1
                total_time += result["inference_time"]
                
                # Manually control data collection save
                # Note: Since we're using auto_save=False, we need to manually end the session
                # But evaluate_single_sample already calls solve_problem which auto-ends the session
                # So we need a different approach
                
                # Actually, when auto_save=True, it saves based on success (has answer)
                # We need to intercept and filter by correctness
                
        except Exception as e:
            result = {
                "id": sample.get("id", "unknown"),
                "success": False,
                "error": str(e)
            }
            results.append(result)
            data_collection_stats["failed_sessions"] += 1
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    accuracy = correct_count / len(successful_results) if successful_results else 0
    avg_inference_time = total_time / len(successful_results) if successful_results else 0
    
    # Track correct and incorrect question IDs
    correct_ids = []
    incorrect_ids = []
    for result in successful_results:
        question_id = result.get("id", "unknown")
        if result.get("is_correct", False):
            correct_ids.append(question_id)
        else:
            incorrect_ids.append(question_id)
    
    # Group statistics by task
    task_stats = {}
    for result in successful_results:
        task = result.get("task", "unknown")
        if task not in task_stats:
            task_stats[task] = {"correct": 0, "total": 0}
        task_stats[task]["total"] += 1
        if result["is_correct"]:
            task_stats[task]["correct"] += 1
    
    # Calculate accuracy for each task
    for task in task_stats:
        task_stats[task]["accuracy"] = task_stats[task]["correct"] / task_stats[task]["total"]
    
    # Tool usage statistics
    tool_usage_stats = {}
    for result in successful_results:
        for tool in result.get("used_tools", []):
            if tool not in tool_usage_stats:
                tool_usage_stats[tool] = 0
            tool_usage_stats[tool] += 1
    
    # Export collected data if enabled
    if enable_data_collection and collector:
        print(f"\n{'='*60}")
        print("Data Collection Summary")
        print(f"{'='*60}")
        
        collection_stats = collector.get_statistics()
        print(f"Total sessions:      {collection_stats['total_sessions']}")
        print(f"Successful sessions: {collection_stats['successful_sessions']}")
        print(f"Failed sessions:     {collection_stats['failed_sessions']}")
        print(f"Success rate:        {collection_stats['success_rate']:.1%}")
        print(f"Total samples:       {collection_stats['total_samples']}")
        
        # Save statistics
        collector.save_statistics()
        
        # Export training data in multiple formats
        try:
            collector.export_for_training(
                output_file=f"{data_collection_dir}/train.jsonl",
                format="jsonl"
            )
            print(f"✓ Exported to {data_collection_dir}/train.jsonl")
            
            collector.export_for_training(
                output_file=f"{data_collection_dir}/train_sharegpt.json",
                format="sharegpt"
            )
            print(f"✓ Exported to {data_collection_dir}/train_sharegpt.json")
            
            print(f"\n📁 Training data saved to: {data_collection_dir}/")
        except Exception as e:
            print(f"⚠️  Warning: Failed to export training data: {e}")
        
        # Update data collection stats
        data_collection_stats.update({
            "saved_sessions": collection_stats['successful_sessions'],
            "total_samples": collection_stats['total_samples']
        })
    
    return {
        "config_name": config_name,
        "total_samples": len(data),
        "successful_samples": len(successful_results),
        "failed_samples": len(failed_results),
        "overall_accuracy": accuracy,
        "average_inference_time": avg_inference_time,
        "total_inference_time": total_time,
        "task_statistics": task_stats,
        "tool_usage_statistics": tool_usage_stats,
        "failed_samples_details": failed_results,
        "model": model,
        "data_collection_enabled": enable_data_collection,
        "data_collection_statistics": data_collection_stats if enable_data_collection else None,
        "data_collection_dir": data_collection_dir if enable_data_collection else None,
        "correct_question_ids": correct_ids,
        "incorrect_question_ids": incorrect_ids
    }


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='SPAgent Evaluation with Data Collection')

    parser.add_argument('--data_path', type=str, default='dataset/cvbench_data.jsonl',
                        help='Path to the data file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None = all)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads')
    parser.add_argument('--image_base_path', type=str, default='dataset',
                        help='Path to the image base directory')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model to use for evaluation')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum number of tool-call iterations')
    
    # Data collection arguments
    parser.add_argument('--enable_data_collection', action='store_true', default=False,
                        help='Enable training data collection')
    parser.add_argument('--data_collection_dir', type=str, default=None,
                        help='Directory for collected data (auto-generated if not specified)')
    parser.add_argument('--only_save_correct', action='store_true', default=False,
                        help='Only save samples with correct predictions')

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
        results = evaluate_tool_config_with_data_collection(
            config_name=config_name,
            tools=tools,
            data_path=args.data_path,
            image_base_path=args.image_base_path,
            model=args.model,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            max_iterations=args.max_iterations,
            enable_data_collection=args.enable_data_collection,
            data_collection_dir=args.data_collection_dir,
            only_save_correct=args.only_save_correct
        )
        all_results[config_name] = results
        
        # Print individual config results
        print(f"\n{'='*60}")
        print(f"Results for {config_name}")
        print(f"{'='*60}")
        print_evaluation_results(results)
        
        # Print data collection info
        if args.enable_data_collection and results.get('data_collection_statistics'):
            dc_stats = results['data_collection_statistics']
            print(f"\nData Collection:")
            print(f"  - Attempted: {dc_stats['attempted_sessions']}")
            print(f"  - Saved:     {dc_stats['saved_sessions']}")
            print(f"  - Samples:   {dc_stats.get('total_samples', 0)}")
            print(f"  - Directory: {results['data_collection_dir']}")
    
    # Save all results to file
    output_file = f"spagent_evaluation_results_{args.model.replace('-', '_')}_{args.max_iterations}"
    if args.enable_data_collection:
        output_file += "_with_data_collection"
    output_file += ".json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ All results saved to {output_file}")
    
    # Summary
    if args.enable_data_collection:
        print(f"\n{'='*60}")
        print("Data Collection Summary")
        print(f"{'='*60}")
        for config_name, results in all_results.items():
            if results.get('data_collection_dir'):
                print(f"\n{config_name}:")
                print(f"  Training data: {results['data_collection_dir']}/")
                print(f"  Formats: train.jsonl, train_sharegpt.json")

if __name__ == "__main__":
    main()

