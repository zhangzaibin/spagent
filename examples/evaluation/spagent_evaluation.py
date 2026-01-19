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

def clean_dict_from_images(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归清理字典中的所有图片数据
    
    Args:
        data: 原始数据字典
        
    Returns:
        清理后的字典（不包含任何图片数据）
    """
    # 定义所有可能包含图片的字段
    IMAGE_FIELDS = [
        'image', 'images', 'img', 
        'camera_views', 'camera_view',
        'output_path', 'vis_path', 'visualization_path',
        'frames', 'frame',
        'visualization', 'rendered_image', 'rendered_images',
        'depth_map', 'depth_image',
        'mask', 'masks', 'segmentation_mask',
        'image_path', 'image_paths', 'img_path'
    ]
    
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # 跳过所有可能包含图片的字段
            if key in IMAGE_FIELDS:
                # 记录该字段存在但不保存内容
                cleaned[f'has_{key}'] = True
            elif isinstance(value, dict):
                # 递归清理嵌套字典
                cleaned[key] = clean_dict_from_images(value)
            elif isinstance(value, list):
                # 递归清理列表中的字典
                cleaned[key] = [
                    clean_dict_from_images(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                # 保留其他字段
                cleaned[key] = value
        return cleaned
    else:
        return data


def clean_tool_results(tool_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理工具结果，移除所有图片数据以节省空间
    
    Args:
        tool_results: 原始工具结果
        
    Returns:
        清理后的工具结果（不包含任何图片数据）
    """
    cleaned_results = {}
    
    for tool_name, result in tool_results.items():
        cleaned_results[tool_name] = clean_dict_from_images(result) if isinstance(result, dict) else result
    
    return cleaned_results


def save_detailed_interaction_records(results: List[Dict[str, Any]], output_file: str):
    """
    Save detailed interaction records for each question to a JSON file
    
    Args:
        results: List of evaluation results
        output_file: Path to output JSON file
    """
    detailed_records = []
    
    for result in results:
        if not result.get("success"):
            # For failed samples, only record basic info
            record = {
                "id": result.get("id", "unknown"),
                "success": False,
                "error": result.get("error", "Unknown error")
            }
        else:
            # For successful samples, record all details
            interaction_records = result.get("interaction_records", {})
            
            # 清理 tool_calls 中的图片数据
            tool_calls = interaction_records.get("tool_calls", [])
            cleaned_tool_calls = [
                clean_dict_from_images(call) if isinstance(call, dict) else call
                for call in tool_calls
            ]
            
            record = {
                "id": result.get("id", "unknown"),
                "success": True,
                "question": result.get("question", ""),
                "ground_truth": result.get("ground_truth", ""),
                "prediction": result.get("prediction", ""),
                "normalized_prediction": result.get("normalized_prediction", ""),
                "normalized_ground_truth": result.get("normalized_ground_truth", ""),
                "is_correct": result.get("is_correct", False),
                "task": result.get("task", "unknown"),
                "inference_time": result.get("inference_time", 0),
                
                # Tool usage information
                "used_tools": result.get("used_tools", []),
                "pi3_parameters": result.get("pi3_parameters", []),
                
                # Detailed interaction records
                "interaction": {
                    "initial_response": interaction_records.get("initial_response", ""),
                    "final_answer": interaction_records.get("final_answer", ""),
                    "iterations": interaction_records.get("iterations", 0),
                    "tool_calls": cleaned_tool_calls,  # 使用清理后的 tool_calls
                    # tool_results 不保存
                    "num_additional_images": len(interaction_records.get("additional_images", [])),  # 只保存数量
                    "baseline_answer": interaction_records.get("baseline_answer", None)
                }
            }
        
        detailed_records.append(record)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_records, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Detailed interaction records saved to: {output_file}")

def extract_pi3_parameters(agent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract Pi3 tool parameters from agent result
    
    Args:
        agent_result: Result dictionary from agent.solve_problem()
        
    Returns:
        List of dictionaries containing Pi3 parameters:
        {
            "azimuth_angle": int,
            "elevation_angle": int,
            "rotation_reference_camera": int,
            "camera_view": bool
        }
    """
    pi3_params = []
    tool_calls = agent_result.get("tool_calls", [])
    
    for call in tool_calls:
        if call.get("name") == "pi3_tool":
            arguments = call.get("arguments", {})
            azimuth = arguments.get("azimuth_angle", 0)
            elevation = arguments.get("elevation_angle", 0)
            rotation_ref_camera = arguments.get("rotation_reference_camera", 1)
            camera_view = arguments.get("camera_view", False)
            
            # Convert to int for cleaner statistics
            azimuth_int = int(round(azimuth))
            elevation_int = int(round(elevation))
            
            # Handle rotation_reference_camera - it might be a list or int
            if isinstance(rotation_ref_camera, list):
                rotation_ref_camera_int = int(rotation_ref_camera[0]) if rotation_ref_camera else 1
            else:
                rotation_ref_camera_int = int(rotation_ref_camera)
            
            # Handle camera_view - it might be a list or bool
            if isinstance(camera_view, list):
                camera_view_bool = bool(camera_view[0]) if camera_view else False
            else:
                camera_view_bool = bool(camera_view)
            
            pi3_params.append({
                "azimuth_angle": azimuth_int,
                "elevation_angle": elevation_int,
                "rotation_reference_camera": rotation_ref_camera_int,
                "camera_view": camera_view_bool
            })
    
    return pi3_params


def extract_video_frames(video_path: str, num_frames: int = 10) -> List[str]:
    """Extract frames from video by uniformly sampling a fixed number of frames
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract uniformly from the video, default 10
        
    Returns:
        List of paths to extracted frame images
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / original_fps
    
    # Use the specified number of frames directly
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
    print(f"Extracted {len(frame_paths)} frames from video (duration: {total_duration:.2f}s, original fps: {original_fps:.2f}, uniformly sampled {num_frames} frames)")
    return frame_paths

def evaluate_single_video(
    agent: SPAgent,
    sample: Dict[str, Any], 
    video_base_path: str,
    num_frames: int = 10,
    max_iterations: int = 3,
    pi3_num_frames: int = None,
    config_name: str = "default",
) -> Dict[str, Any]:
    """Evaluate a single video sample
    
    Args:
        agent: SPAgent instance
        sample: Data sample
        video_base_path: Base path for videos
        max_iterations: Maximum number of tool-call iterations
        num_frames: Number of frames to uniformly sample for initial model judgment, default 10
        pi3_num_frames: Number of frames to uniformly sample for pi3 tool (if None, uses num_frames)
        config_name: Configuration name for saving results
        
    Returns:
        Evaluation result dictionary
    """
    # Validate paths
    is_valid, result = validate_sample_paths(sample, video_base_path, "video")
    if not is_valid:
        return result
    
    try:
        # Extract video frames (uniformly sample specified number of frames)
        video_path = result["path"][0]  # 暂时只支持单个视频
        frame_paths = extract_video_frames(video_path, num_frames)
        
        # Use pi3_num_frames if provided, otherwise use num_frames
        actual_pi3_num_frames = pi3_num_frames if pi3_num_frames is not None else num_frames
        
        # Run inference using SPAgent
        # Pass video_path so that if pi3 tool is called, more frames can be extracted
        start_time = time.time()
        agent_result = agent.solve_problem(
            frame_paths,
            f"Based on these {len(frame_paths)} uniformly sampled frames from a video, please answer: {result['question']}",
            max_iterations=max_iterations,
            video_path=video_path,
            pi3_num_frames=actual_pi3_num_frames
        )
        inference_time = time.time() - start_time
        
        prediction = agent_result["answer"]
        
        # Normalize answers
        _, normalized_prediction = normalize_answer(prediction)
        _, normalized_ground_truth = normalize_answer(result["ground_truth"])
        
        # Check correctness
        is_correct = normalized_prediction == normalized_ground_truth
        
        # Extract Pi3 parameters (angles + rotation_reference_camera + camera_view)
        pi3_params = extract_pi3_parameters(agent_result)
        
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
            "pi3_parameters": pi3_params,  # NEW: Full Pi3 parameters
            # NEW: Detailed interaction records
            "interaction_records": {
                "tool_calls": agent_result.get("tool_calls", []),
                "initial_response": agent_result.get("initial_response", ""),
                "final_answer": agent_result.get("answer", ""),
                "iterations": agent_result.get("iterations", 0),
                "additional_images": agent_result.get("additional_images", []),
                "baseline_answer": agent_result.get("baseline_answer", None)
            }
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
        
        # Extract Pi3 parameters (angles + rotation_reference_camera + camera_view)
        pi3_params = extract_pi3_parameters(agent_result)
        
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
            "pi3_parameters": pi3_params,  # NEW: Full Pi3 parameters
            # NEW: Detailed interaction records
            "interaction_records": {
                "tool_calls": agent_result.get("tool_calls", []),
                "initial_response": agent_result.get("initial_response", ""),
                "final_answer": agent_result.get("answer", ""),
                "iterations": agent_result.get("iterations", 0),
                "additional_images": agent_result.get("additional_images", []),
                "baseline_answer": agent_result.get("baseline_answer", None)
            }
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
    
    # NEW: Track Pi3 parameters (angles + rotation_reference_camera + camera_view)
    all_pi3_params = []
    
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
                num_frames = 7  # Uniformly sample 10 frames
                pi3_num_frames = 7  # Use more frames for pi3 reconstruction
            elif sample['data_source'] == "VLM4D":
                num_frames = 7  # Uniformly sample 10 frames
                pi3_num_frames = 7  # Use more frames for pi3 reconstruction
            else:
                num_frames = 7
                pi3_num_frames = 7
                print(f"The num_frames parameter has not been specified for the {sample['data_source']} dataset yet, and the default value of 10 will be adopted")
            result = evaluate_single_video(agent, sample, image_base_path, num_frames=num_frames, pi3_num_frames=pi3_num_frames, config_name=config_name, max_iterations=max_iterations)
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
            # Collect Pi3 parameters
            pi3_params = result.get("pi3_parameters", [])
            all_pi3_params.extend(pi3_params)
    
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
    
    # NEW: Pi3 parameter distribution statistics
    pi3_stats = {}
    if all_pi3_params:
        # Extract angles for backward compatibility
        all_angles = [(p["azimuth_angle"], p["elevation_angle"]) for p in all_pi3_params]
        angle_counter = Counter(all_angles)
        sorted_angles = sorted(angle_counter.items(), key=lambda x: (-x[1], x[0]))
        
        # Count rotation_reference_camera usage
        rotation_ref_camera_counter = Counter([p["rotation_reference_camera"] for p in all_pi3_params])
        
        # Count camera_view usage
        camera_view_counter = Counter([p["camera_view"] for p in all_pi3_params])
        camera_view_true = camera_view_counter.get(True, 0)
        camera_view_false = camera_view_counter.get(False, 0)
        
        # Full parameter combinations (for detailed analysis)
        full_param_combinations = []
        for p in all_pi3_params:
            combo = f"(azim={p['azimuth_angle']}, elev={p['elevation_angle']}, " \
                    f"ref_cam={p['rotation_reference_camera']}, camera_view={p['camera_view']})"
            full_param_combinations.append(combo)
        
        pi3_stats = {
            "total_pi3_calls": len(all_pi3_params),
            "unique_angle_combinations": len(angle_counter),
            "angle_distribution": {
                f"({azim}, {elev})": count 
                for (azim, elev), count in sorted_angles
            },
            "top_5_angle_combinations": [
                {"angle": f"({azim}, {elev})", "count": count, 
                 "percentage": f"{count/len(all_angles)*100:.1f}%"}
                for (azim, elev), count in sorted_angles[:5]
            ],
            "rotation_reference_camera_usage": {
                f"camera_{cam}": count 
                for cam, count in sorted(rotation_ref_camera_counter.items())
            },
            "rotation_reference_camera_percentage": {
                f"camera_{cam}": f"{count/len(all_pi3_params)*100:.1f}%"
                for cam, count in sorted(rotation_ref_camera_counter.items())
            },
            "camera_view_usage": {
                "enabled_true": camera_view_true,
                "disabled_false": camera_view_false
            },
            "camera_view_percentage": {
                "enabled_true": f"{camera_view_true/len(all_pi3_params)*100:.1f}%",
                "disabled_false": f"{camera_view_false/len(all_pi3_params)*100:.1f}%"
            },
            "full_parameter_combinations": Counter(full_param_combinations)
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
        "model": model,
        "detailed_results": results,  # NEW: Include all detailed results for each sample,
        "correct_question_ids": correct_ids,
        "incorrect_question_ids": incorrect_ids
    }
    
    # Add Pi3 parameter statistics if available
    if pi3_stats:
        result_dict["pi3_statistics"] = pi3_stats
    
    # Save detailed interaction records to a separate JSON file
    detailed_output_file = f"{config_name}_detailed_interactions.json"
    save_detailed_interaction_records(results, detailed_output_file)
    
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