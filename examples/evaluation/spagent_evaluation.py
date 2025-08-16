import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm
import cv2
import numpy as np

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
from utils.utils import (
    load_blink_data, 
    extract_question_and_answer, 
    normalize_answer, 
    print_evaluation_results, 
    validate_sample_paths,
    save_error_to_tsv
)

# Define server URLs
TOOL_SERVERS = {
    "depth": "http://10.8.131.51:30750",
    "segmentation": "http://10.8.131.51:30646",
    "detection": "http://10.8.131.51:30969"
}

    # "baseline_no_tools": [
    #     # Empty tool list - pure LLM baseline
    # ],

# Define tool combinations for evaluation

TOOL_CONFIGS = {
    "baseline_no_tools": [
        # Empty tool list - pure LLM baseline
    ],
    "depth_detection_segmentation": [
        DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"]),
        ObjectDetectionTool(use_mock=False, server_url=TOOL_SERVERS["detection"]),
        SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["segmentation"])
    ]
}


def extract_video_frames(video_path: str, target_fps: float = 1.0) -> List[str]:
    """Extract frames from video
    
    Args:
        video_path: Path to video file
        target_fps: Target frame rate, default 1 fps
        
    Returns:
        List of paths to extracted frame images
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / original_fps
    
    # Calculate frames to extract based on target fps
    num_frames = int(total_duration * target_fps)
    frame_interval = total_frames / num_frames
    
    frame_paths = []
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    
    # Extract frames evenly
    for i in range(num_frames):
        frame_idx = int(i * frame_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = temp_dir / f"frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
    
    cap.release()
    print(f"Extracted {len(frame_paths)} frames from video (duration: {total_duration:.2f}s, original fps: {original_fps:.2f}, target fps: {target_fps})")
    return frame_paths

def evaluate_single_video(
    agent: SPAgent,
    sample: Dict[str, Any], 
    video_base_path: str,
    target_fps: float = 1.0
) -> Dict[str, Any]:
    """Evaluate a single video sample
    
    Args:
        agent: SPAgent instance
        sample: Data sample
        video_base_path: Base path for videos
        target_fps: Target frame rate, default 1 fps
        
    Returns:
        Evaluation result dictionary
    """
    # Validate paths
    is_valid, result = validate_sample_paths(sample, video_base_path, "video")
    if not is_valid:
        return result
    
    try:
        # Extract video frames
        frame_paths = extract_video_frames(result["path"], target_fps)
        
        # Run inference using SPAgent
        start_time = time.time()
        agent_result = agent.solve_problem(
            frame_paths,
            f"Based on these {len(frame_paths)} frames from a video (sampled at {target_fps} fps), please answer: {result['question']}"
        )
        inference_time = time.time() - start_time
        
        prediction = agent_result["answer"]
        
        # Normalize answers
        _, normalized_prediction = normalize_answer(prediction)
        _, normalized_ground_truth = normalize_answer(result["ground_truth"])
        
        # Check correctness
        is_correct = normalized_prediction == normalized_ground_truth
        
        # Save errors to TSV
        if not is_correct:
            error_data = {
                'question': result["question"],
                'path': result["path"],
                'analysis': prediction,
                'normalized_prediction': normalized_prediction,
                'normalized_ground_truth': normalized_ground_truth
            }
            save_error_to_tsv(error_data)
        
        return {
            "id": sample.get("id", "unknown"),
            "success": True,
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "prediction": prediction,
            "normalized_prediction": normalized_prediction,
            "normalized_ground_truth": normalized_ground_truth,
            "is_correct": is_correct,
            "inference_time": inference_time,
            "task": sample.get("task", "unknown"),
            "num_frames": len(frame_paths),
            "target_fps": target_fps,
            "used_tools": agent_result.get("used_tools", [])
        }
        
    except Exception as e:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": str(e)
        }
    finally:
        # Cleanup temporary files
        if 'frame_paths' in locals():
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            try:
                os.rmdir("temp_frames")
            except:
                pass

def evaluate_single_sample(
    agent: SPAgent,
    sample: Dict[str, Any], 
    image_base_path: str
) -> Dict[str, Any]:
    """Evaluate a single image sample
    
    Args:
        agent: SPAgent instance
        sample: Data sample
        image_base_path: Base path for images
        
    Returns:
        Evaluation result dictionary
    """
    is_valid, result = validate_sample_paths(sample, image_base_path, "image")
    if not is_valid:
        return result
    
    try:
        # Run inference using SPAgent
        start_time = time.time()
        agent_result = agent.solve_problem(
            result["path"],
            result["question"]
        )
        inference_time = time.time() - start_time
        
        prediction = agent_result["answer"]
        
        # Normalize answers
        _, normalized_prediction = normalize_answer(prediction)
        _, normalized_ground_truth = normalize_answer(result["ground_truth"])
        
        # Check correctness
        is_correct = normalized_prediction == normalized_ground_truth
        
        # Save errors to TSV
        if not is_correct:
            error_data = {
                'question': result["question"],
                'path': result["path"],
                'analysis': prediction,
                'normalized_prediction': normalized_prediction,
                'normalized_ground_truth': normalized_ground_truth
            }
            save_error_to_tsv(error_data)
        
        return {
            "id": sample.get("id", "unknown"),
            "success": True,
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "prediction": prediction,
            "normalized_prediction": normalized_prediction,
            "normalized_ground_truth": normalized_ground_truth,
            "is_correct": is_correct,
            "inference_time": inference_time,
            "task": sample.get("task", "unknown"),
            "used_tools": agent_result.get("used_tools", [])
        }
        
    except Exception as e:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": str(e)
        }

def evaluate_tool_config(
    config_name: str,
    tools: List[Any],
    data_path: str,
    image_base_path: str,
    model: str = "gpt-4o-mini",
    max_samples: int = None,
    max_workers: int = 4
) -> Dict[str, Any]:
    """Evaluate a specific tool configuration
    
    Args:
        config_name: Name of the tool configuration
        tools: List of tool instances
        data_path: Path to dataset file
        image_base_path: Base path for images/videos
        model: Model name to use
        max_samples: Maximum number of samples to evaluate
        max_workers: Maximum number of parallel workers
        
    Returns:
        Evaluation results dictionary
    """
    print(f"\nEvaluating configuration: {config_name}")
    print(f"Loading data from {data_path}")
    data = load_blink_data(data_path)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples for evaluation")
    
    # Create SPAgent instance
    agent = SPAgent(
        model=GPTModel(model_name=model),
        tools=tools,
        max_workers=max_workers
    )
    
    print(f"Evaluating {len(data)} samples with {model}")
    
    results = []
    correct_count = 0
    total_time = 0
    
    # Use tqdm for progress tracking
    for sample in tqdm(data, desc="Evaluating"):
        # Determine sample type and choose evaluation function
        has_image = bool(sample.get("image", []))
        has_video = bool(sample.get("video", []))
        
        if has_image and not has_video:
            # Image sample
            result = evaluate_single_sample(agent, sample, image_base_path)
        elif has_video and not has_image:
            # Video sample
            result = evaluate_single_video(agent, sample, image_base_path, target_fps=0.5)
        else:
            # Invalid sample
            result = {
                "id": sample.get("id", "unknown"),
                "success": False,
                "error": "Invalid sample: must contain either image or video, but not both"
            }
        
        results.append(result)
        
        if result["success"]:
            if result["is_correct"]:
                correct_count += 1
            total_time += result["inference_time"]
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    accuracy = correct_count / len(successful_results) if successful_results else 0
    avg_inference_time = total_time / len(successful_results) if successful_results else 0
    
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
    
    # Add sample type statistics
    sample_type_stats = {
        "image_samples": len([r for r in results if "image" in r.get("error", "") or (r["success"] and not "num_frames" in r)]),
        "video_samples": len([r for r in results if "video" in r.get("error", "") or (r["success"] and "num_frames" in r)]),
        "invalid_samples": len([r for r in results if "Invalid sample" in r.get("error", "")])
    }
    
    # Tool usage statistics
    tool_usage_stats = {}
    for result in successful_results:
        for tool in result.get("used_tools", []):
            if tool not in tool_usage_stats:
                tool_usage_stats[tool] = 0
            tool_usage_stats[tool] += 1
    
    return {
        "config_name": config_name,
        "total_samples": len(data),
        "successful_samples": len(successful_results),
        "failed_samples": len(failed_results),
        "overall_accuracy": accuracy,
        "average_inference_time": avg_inference_time,
        "total_inference_time": total_time,
        "task_statistics": task_stats,
        "sample_type_statistics": sample_type_stats,
        "tool_usage_statistics": tool_usage_stats,
        "failed_samples_details": failed_results,
        "model": model
    }

def main():
    """Main function"""
    # Configure paths
    data_path = "dataset/blink_data.jsonl"
    image_base_path = "dataset"
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    if not os.path.exists(image_base_path):
        print(f"Error: Image base path not found at {image_base_path}")
        return
    
    # Evaluation parameters
    model = "gpt-4o"
    max_samples = 50  # Set to None for full evaluation
    max_workers = 4
    
    # Run evaluation for each tool configuration
    all_results = {}
    for config_name, tools in TOOL_CONFIGS.items():
        results = evaluate_tool_config(
            config_name=config_name,
            tools=tools,
            data_path=data_path,
            image_base_path=image_base_path,
            model=model,
            max_samples=max_samples,
            max_workers=max_workers
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