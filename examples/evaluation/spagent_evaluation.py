import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm
import cv2
import numpy as np
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent import SPAgent
from spagent.models import GPTModel, QwenModel, QwenVLLMModel
from spagent.tools import (
    DepthEstimationTool,
    SegmentationTool,
    ObjectDetectionTool,
    SupervisionTool,
    YOLOETool
)
from spagent.utils.utils import (
    load_json_data, 
    extract_question_and_answer, 
    normalize_answer, 
    print_evaluation_results, 
    validate_sample_paths,
    save_result_to_csv
)

# Define server URLs
# TOOL_SERVERS = {
#     "depth": "http://10.8.131.51:30750",
#     "segmentation": "http://10.8.131.51:30646",
#     "detection": "http://10.8.131.51:30969"
# }
TOOL_SERVERS = {
    "depth": "http://127.0.0.1:20019",
    "segmentation": "http://127.0.0.1:20020",
    "detection": "http://127.0.0.1:20022"
}


TOOL_CONFIGS = {
    "depth_detection_segmentation": [
        DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"]),
        ObjectDetectionTool(use_mock=False, server_url=TOOL_SERVERS["detection"]),
        SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["segmentation"])
    ]
}

def extract_pi3_angles(agent_result: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Extract Pi3 tool angle parameters from agent result
    
    Args:
        agent_result: Result dictionary from agent.solve_problem()
        
    Returns:
        List of (azimuth_angle, elevation_angle) tuples
    """
    angles = []
    tool_calls = agent_result.get("tool_calls", [])
    
    for call in tool_calls:
        if call.get("name") == "pi3_tool":
            arguments = call.get("arguments", {})
            azimuth = arguments.get("azimuth_angle", 0)
            elevation = arguments.get("elevation_angle", 0)
            # Convert to int for cleaner statistics
            azimuth_int = int(round(azimuth))
            elevation_int = int(round(elevation))
            angles.append((azimuth_int, elevation_int))
    
    return angles


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
    
    # 从视频路径中提取文件名（不含扩展名）
    video_filename = Path(video_path).stem
    
    # Extract frames evenly
    for i in range(num_frames):
        frame_idx = int(i * frame_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = temp_dir / f"{video_filename}_frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
    
    cap.release()
    print(f"Extracted {len(frame_paths)} frames from video (duration: {total_duration:.2f}s, original fps: {original_fps:.2f}, target fps: {target_fps})")
    return frame_paths

def evaluate_single_video(
    agent: SPAgent,
    sample: Dict[str, Any], 
    video_base_path: str,
    target_fps: float = 1.0,
    max_iterations: int = 3,
    pi3_target_fps: float = None,
    config_name: str = "default",
) -> Dict[str, Any]:
    """Evaluate a single video sample
    
    Args:
        agent: SPAgent instance
        sample: Data sample
        video_base_path: Base path for videos
        max_iterations: Maximum number of tool-call iterations
        target_fps: Target frame rate for initial model judgment, default 1 fps
        pi3_target_fps: Target frame rate for pi3 tool (if None, uses target_fps)
        config_name: Configuration name for saving results
        
    Returns:
        Evaluation result dictionary
    """
    # Validate paths
    is_valid, result = validate_sample_paths(sample, video_base_path, "video")
    if not is_valid:
        return result
    
    try:
        # Extract video frames (fewer frames for initial model judgment)
        video_path = result["path"][0]  # 暂时只支持单个视频
        frame_paths = extract_video_frames(video_path, target_fps)
        
        # Use pi3_target_fps if provided, otherwise use target_fps
        actual_pi3_fps = pi3_target_fps if pi3_target_fps is not None else target_fps
        
        # Run inference using SPAgent
        # Pass video_path so that if pi3 tool is called, more frames can be extracted
        start_time = time.time()
        agent_result = agent.solve_problem(
            frame_paths,
            f"Based on these {len(frame_paths)} frames from a video (sampled at {target_fps} fps), please answer: {result['question']}",
            max_iterations=max_iterations,
            video_path=video_path,
            pi3_target_fps=actual_pi3_fps
        )
        inference_time = time.time() - start_time
        
        prediction = agent_result["answer"]
        
        # Normalize answers
        _, normalized_prediction = normalize_answer(prediction)
        _, normalized_ground_truth = normalize_answer(result["ground_truth"])
        
        # Check correctness
        is_correct = normalized_prediction == normalized_ground_truth
        
        # Extract Pi3 angle parameters
        pi3_angles = extract_pi3_angles(agent_result)
        
        task_data = {
            'question': result["question"],
            'path': result["path"],
            'analysis': prediction,
            'normalized_prediction': normalized_prediction,
            'normalized_ground_truth': normalized_ground_truth,
            'is_correct': '1' if is_correct else '0',
            'used_tools': agent_result.get("used_tools", []),
            'follow_up_prompt': agent_result["prompts"]["follow_up_prompt"]
        }
        # use the config name as the csv file name
        save_result_to_csv(task_data, csv_file=f"{config_name}.csv")

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
            "used_tools": agent_result.get("used_tools", []),
            "pi3_angles": pi3_angles  # NEW: Pi3 angle combinations
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
    image_base_path: str,
    config_name: str = "default",
    max_iterations: int = 3
) -> Dict[str, Any]:
    """Evaluate a single image sample
    
    Args:
        agent: SPAgent instance
        sample: Data sample
        image_base_path: Base path for images
        config_name: Configuration name for saving results
        max_iterations: Maximum number of tool-call iterations
        
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
            result["question"],
            max_iterations=max_iterations
        )
        inference_time = time.time() - start_time
        
        prediction = agent_result["answer"]
        
        # Normalize answers
        _, normalized_prediction = normalize_answer(prediction)
        _, normalized_ground_truth = normalize_answer(result["ground_truth"])
        
        # Check correctness
        is_correct = normalized_prediction == normalized_ground_truth
        
        # Extract Pi3 angle parameters
        pi3_angles = extract_pi3_angles(agent_result)
        
        task_data = {
            'question': result["question"],
            'path': result["path"],
            'analysis': prediction,
            'normalized_prediction': normalized_prediction,
            'normalized_ground_truth': normalized_ground_truth,
            'is_correct': '1' if is_correct else '0',
            'used_tools': agent_result.get("used_tools", []),
            'follow_up_prompt': agent_result["prompts"]["follow_up_prompt"]
        }
        # use the config name as the csv file name
        save_result_to_csv(task_data, csv_file=f"{config_name}.csv")

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
            "used_tools": agent_result.get("used_tools", []),
            "pi3_angles": pi3_angles  # NEW: Pi3 angle combinations
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
    max_workers: int = 4,
    max_iterations: int = 3,
    data_collector=None  # NEW: Optional DataCollector
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
        max_iterations: Maximum number of tool-call iterations
        
    Returns:
        Evaluation results dictionary
    """
    print(f"\nEvaluating configuration: {config_name}")
    print(f"Loading data from {data_path}")
    data = load_json_data(data_path)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples for evaluation")
    
    # Create SPAgent instance
    agent = SPAgent(
        model=GPTModel(model_name=model),
        tools=tools,
        max_workers=max_workers,
        data_collector=data_collector  # NEW: Pass DataCollector
    )
    
    print(f"Evaluating {len(data)} samples with {model}")
    
    results = []
    correct_count = 0
    total_time = 0
    
    # NEW: Track Pi3 angle combinations
    all_pi3_angles = []
    
    # Use tqdm for progress tracking
    for sample in tqdm(data, desc="Evaluating"):
        # Determine sample type and choose evaluation function
        has_image = bool(sample.get("image", []))
        has_video = bool(sample.get("video", []))
        
        if has_image and not has_video:
            # Image sample
            result = evaluate_single_sample(agent, sample, image_base_path, config_name, max_iterations)
        elif has_video and not has_image:
            # Video sample
            if sample['data_source'] == "VSI-Bench":
                target_fps = 0.1
                pi3_target_fps = 0.3  # Use more frames for pi3 reconstruction
            elif sample['data_source'] == "VLM4D":
                target_fps = 1.00
                pi3_target_fps = 5.0  # Use more frames for pi3 reconstruction
            else:
                target_fps = 1.00
                pi3_target_fps = 3.0
                print(f"The target fps parameter has not been specified for the {sample['data_source']} dataset yet, and the default value of 1.00 will be adopted")
            result = evaluate_single_video(agent, sample, image_base_path, target_fps=target_fps, pi3_target_fps=pi3_target_fps, config_name=config_name, max_iterations=max_iterations)
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
            # Collect Pi3 angles
            pi3_angles = result.get("pi3_angles", [])
            all_pi3_angles.extend(pi3_angles)
    
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
    
    # NEW: Pi3 angle distribution statistics
    pi3_angle_distribution = {}
    if all_pi3_angles:
        angle_counter = Counter(all_pi3_angles)
        # Sort by count (descending) then by angle
        sorted_angles = sorted(angle_counter.items(), key=lambda x: (-x[1], x[0]))
        pi3_angle_distribution = {
            "total_pi3_calls": len(all_pi3_angles),
            "unique_angle_combinations": len(angle_counter),
            "distribution": {
                f"({azim}, {elev})": count 
                for (azim, elev), count in sorted_angles
            },
            "top_5_combinations": [
                {"angle": f"({azim}, {elev})", "count": count, "percentage": f"{count/len(all_pi3_angles)*100:.1f}%"}
                for (azim, elev), count in sorted_angles[:5]
            ]
        }
    
    # NEW: Export collected data if DataCollector was provided
    if data_collector:
        print(f"\n{'='*60}")
        print("Data Collection Summary")
        print(f"{'='*60}")
        
        stats = data_collector.get_statistics()
        print(f"Total sessions:      {stats['total_sessions']}")
        print(f"Successful sessions: {stats['successful_sessions']}")
        print(f"Failed sessions:     {stats['failed_sessions']}")
        print(f"Total samples:       {stats['total_samples']}")
        print(f"Success rate:        {stats['success_rate']:.1%}")
        
        # Save statistics
        data_collector.save_statistics()
        
        # Export training data
        output_dir = data_collector.output_dir
        try:
            # Export in multiple formats
            
            # 1. Simple format (most concise)
            data_collector.export_for_training(
                output_file=f"{output_dir}/train_simple.jsonl",
                format="simple"
            )
            print(f"✓ Exported to {output_dir}/train_simple.jsonl (SIMPLE format)")
            
            # 2. Full JSONL format
            data_collector.export_for_training(
                output_file=f"{output_dir}/train_full.jsonl",
                format="jsonl"
            )
            print(f"✓ Exported to {output_dir}/train_full.jsonl (FULL format)")
            
            # 3. ShareGPT format with simple prompt
            data_collector.export_for_training(
                output_file=f"{output_dir}/train_sharegpt_simple.json",
                format="sharegpt",
                simple_format=True
            )
            print(f"✓ Exported to {output_dir}/train_sharegpt_simple.json (ShareGPT SIMPLE)")
            
            # 4. ShareGPT format with full prompt
            data_collector.export_for_training(
                output_file=f"{output_dir}/train_sharegpt_full.json",
                format="sharegpt",
                simple_format=False
            )
            print(f"✓ Exported to {output_dir}/train_sharegpt_full.json (ShareGPT FULL)")
            
            print(f"\n📁 Training data saved to: {output_dir}/")
        except Exception as e:
            print(f"⚠️  Warning: Failed to export training data: {e}")
            import traceback
            traceback.print_exc()
    
    result_dict = {
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
    
    # Add Pi3 angle distribution if available
    if pi3_angle_distribution:
        result_dict["pi3_angle_distribution"] = pi3_angle_distribution
    
    return result_dict

def main():
    """Main function"""
    # Configure paths
    data_path = "dataset/Sampled_Tasks_3.jsonl"
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
    max_samples = None  # Set to None for full evaluation
    max_workers = 4
    
    # Run evaluation for each tool configuration
    all_results = {}
    for config_name, tools in TOOL_CONFIGS.items():
        RUN_CONFIG_NAME = config_name
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