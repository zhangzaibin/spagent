import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm
import cv2
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vllm_models.gpt import gpt_multiple_images_inference, gpt_single_image_inference
from utils.utils import load_blink_data, extract_question_and_answer, normalize_answer, print_evaluation_results

def extract_video_frames(video_path: str, target_fps: float = 1.0) -> List[str]:
    """从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        target_fps: 目标帧率，默认每秒1帧
        
    Returns:
        帧图像的临时文件路径列表
    """
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / original_fps  # 视频总时长（秒）
    
    # 根据目标帧率计算需要提取的帧数
    num_frames = int(total_duration * target_fps)
    frame_interval = total_frames / num_frames  # 帧间隔
    
    frame_paths = []
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    
    # 均匀提取帧
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
    sample: Dict[str, Any], 
    video_base_path: str,
    model: str = "gpt-4o-mini",
    target_fps: float = 1.0
) -> Dict[str, Any]:
    """评估单个视频样本
    
    Args:
        sample: 数据样本
        video_base_path: 视频基础路径
        model: 使用的模型
        target_fps: 目标帧率，默认每秒1帧
        
    Returns:
        评估结果字典
    """
    # 提取视频路径
    video_paths = sample.get("video", [])
    if not video_paths:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "No video found"
        }
    
    # 构建完整视频路径
    video_path = os.path.join(video_base_path, video_paths[0])
    if not os.path.exists(video_path):
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": f"Video not found: {video_path}"
        }
    
    # 提取问题和答案
    conversation = sample.get("conversations", [])
    if not conversation:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "No conversation found"
        }
    
    question, ground_truth = extract_question_and_answer(conversation)
    if not question or not ground_truth:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "Question or answer not found"
        }
    
    try:
        # 提取视频帧
        frame_paths = extract_video_frames(video_path, target_fps)
        
        # 使用GPT进行推理，一次性输入所有帧
        start_time = time.time()
        prediction = gpt_multiple_images_inference(
            image_paths=frame_paths,
            prompt=f"Based on these {len(frame_paths)} frames from a video (sampled at {target_fps} fps), please answer: {question}",
            model=model,
            temperature=0.0,
            max_tokens=50
        )
        inference_time = time.time() - start_time
        
        # 标准化答案
        _, normalized_prediction = normalize_answer(prediction)
        _, normalized_ground_truth = normalize_answer(ground_truth)
        
        # 检查是否正确
        is_correct = normalized_prediction == normalized_ground_truth
        
        return {
            "id": sample.get("id", "unknown"),
            "success": True,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "normalized_prediction": normalized_prediction,
            "normalized_ground_truth": normalized_ground_truth,
            "is_correct": is_correct,
            "inference_time": inference_time,
            "task": sample.get("task", "unknown"),
            "num_frames": len(frame_paths),
            "target_fps": target_fps
        }
        
    except Exception as e:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": str(e)
        }
    finally:
        # 清理临时文件
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
    sample: Dict[str, Any], 
    image_base_path: str,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """评估单个样本
    
    Args:
        sample: 数据样本
        image_base_path: 图像基础路径
        model: 使用的模型
        
    Returns:
        评估结果字典
    """
    # 提取图像路径
    image_paths = sample.get("image", [])
    if not image_paths:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "No image found"
        }
    
    # 构建完整图像路径
    image_path = os.path.join(image_base_path, image_paths[0])
    if not os.path.exists(image_path):
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": f"Image not found: {image_path}"
        }
    
    # 提取问题和答案
    conversation = sample.get("conversations", [])
    if not conversation:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "No conversation found"
        }
    
    question, ground_truth = extract_question_and_answer(conversation)
    if not question or not ground_truth:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "Question or answer not found"
        }
    
    try:
        # 使用GPT进行推理
        start_time = time.time()
        prediction = gpt_single_image_inference(
            image_path=image_path,
            prompt=question,
            model=model,
            temperature=0.0,  # 使用确定性输出
            max_tokens=10  # 限制输出长度
        )
        inference_time = time.time() - start_time
        
        # 标准化答案
        _, normalized_prediction = normalize_answer(prediction)
        _, normalized_ground_truth = normalize_answer(ground_truth)
        
        # 检查是否正确
        is_correct = normalized_prediction == normalized_ground_truth
        
        return {
            "id": sample.get("id", "unknown"),
            "success": True,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "normalized_prediction": normalized_prediction,
            "normalized_ground_truth": normalized_ground_truth,
            "is_correct": is_correct,
            "inference_time": inference_time,
            "task": sample.get("task", "unknown")
        }
        
    except Exception as e:
        return {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": str(e)
        }

def evaluate_blink_dataset(
    data_path: str,
    image_base_path: str,
    model: str = "gpt-4o-mini",
    max_samples: int = None
) -> Dict[str, Any]:
    """评估BLINK数据集
    
    Args:
        data_path: 数据文件路径
        image_base_path: 图像基础路径
        model: 使用的模型
        max_samples: 最大评估样本数（用于测试）
        
    Returns:
        评估结果字典
    """
    print(f"Loading data from {data_path}")
    data = load_blink_data(data_path)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples for evaluation")
    
    print(f"Evaluating {len(data)} samples with {model}")
    
    results = []
    correct_count = 0
    total_time = 0
    
    # 使用tqdm显示进度
    for sample in tqdm(data, desc="Evaluating"):
        # 判断样本类型并选择相应的评估函数
        has_image = bool(sample.get("image", []))
        has_video = bool(sample.get("video", []))
        
        if has_image and not has_video:
            # 图像样本
            result = evaluate_single_sample(sample, image_base_path, model)
        elif has_video and not has_image:
            # 视频样本
            result = evaluate_single_video(sample, image_base_path, model, target_fps=0.5)
        else:
            # 无效样本或同时包含图像和视频
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
    
    # 计算统计信息
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    accuracy = correct_count / len(successful_results) if successful_results else 0
    avg_inference_time = total_time / len(successful_results) if successful_results else 0
    
    # 按任务类型分组统计
    task_stats = {}
    for result in successful_results:
        task = result.get("task", "unknown")
        if task not in task_stats:
            task_stats[task] = {"correct": 0, "total": 0}
        task_stats[task]["total"] += 1
        if result["is_correct"]:
            task_stats[task]["correct"] += 1
    
    # 计算每个任务的准确率
    for task in task_stats:
        task_stats[task]["accuracy"] = task_stats[task]["correct"] / task_stats[task]["total"]
    
    # 添加样本类型统计
    sample_type_stats = {
        "image_samples": len([r for r in results if "image" in r.get("error", "") or (r["success"] and not "num_frames" in r)]),
        "video_samples": len([r for r in results if "video" in r.get("error", "") or (r["success"] and "num_frames" in r)]),
        "invalid_samples": len([r for r in results if "Invalid sample" in r.get("error", "")])
    }
    
    return {
        "total_samples": len(data),
        "successful_samples": len(successful_results),
        "failed_samples": len(failed_results),
        "overall_accuracy": accuracy,
        "average_inference_time": avg_inference_time,
        "total_inference_time": total_time,
        "task_statistics": task_stats,
        "sample_type_statistics": sample_type_stats,
        "failed_samples_details": failed_results,
        "model": model
    }

def main():
    """主函数"""
    # 配置路径
    # data_path = "dataset/blink_data.jsonl"
    data_path = "dataset/VSI-Bench.jsonl"

    image_base_path = "dataset"  # 图像文件的基础路径
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    if not os.path.exists(image_base_path):
        print(f"Error: Image base path not found at {image_base_path}")
        return
    
    # 评估参数
    model = "gpt-4o-mini"
    max_samples = 5  # 设置为数字可以限制评估样本数（用于测试）
    
    # 执行评估
    results = evaluate_blink_dataset(
        data_path=data_path,
        image_base_path=image_base_path,
        model=model,
        max_samples=max_samples
    )
    
    # 打印结果
    print_evaluation_results(results)
    
    # 保存结果到文件
    output_file = f"blink_evaluation_results_{model.replace('-', '_')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
