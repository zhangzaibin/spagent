import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vllm_models.gpt import gpt_single_image_inference, gpt_multiple_images_inference, gpt_text_only_inference
from utils.utils import load_blink_data, extract_question_and_answer, normalize_answer, print_evaluation_results, validate_sample_paths, save_error_to_tsv
from workflows.sam2_qa_workflow import SAM2QAWorkflow


def evaluate_single_sample(
    sample: Dict[str, Any], 
    image_base_path: str,
    model: str = "gpt-4o-mini",
    workflow: SAM2QAWorkflow = None
) -> Dict[str, Any]:
    """评估单个样本
    
    Args:
        sample: 数据样本
        image_base_path: 图像基础路径
        model: 使用的模型
        workflow: SAM2 QA工作流实例
        
    Returns:
        评估结果字典
    """
    is_valid, json_result = validate_sample_paths(sample, image_base_path, "image")
    if not is_valid:
        return json_result

    try:
        # 使用SAM2工作流进行推理
        start_time = time.time()
        result = workflow.run_workflow(json_result["path"], json_result["question"])
        prediction = result.get('answer', '')
        inference_time = time.time() - start_time
        
        # 记录分割结果信息
        segmentation_info = None
        if result.get('segmentation_used') and result.get('segmentation_result'):
            segmentation_info = {
                'mask_path': result['segmentation_result'].get('mask_path'),
                'vis_path': result['segmentation_result'].get('vis_path')
            }
        
        # 标准化答案
        analysis, normalized_prediction = normalize_answer(prediction)
        _, normalized_ground_truth = normalize_answer(json_result["ground_truth"])
        
        # 检查是否正确
        is_correct = normalized_prediction == normalized_ground_truth

        # 如果错误，保存到TSV文件
        if not is_correct:
            error_data = {
                'question': json_result["question"],
                'path': json_result["path"],
                'analysis': prediction,  # 使用prediction作为analysis
                'normalized_prediction': normalized_prediction,
                'normalized_ground_truth': normalized_ground_truth
            }
            save_error_to_tsv(error_data, tsv_file=f"{sample['data_source']}_{sample['task']}.tsv")

        return {
            "id": sample.get("id", "unknown"),
            "success": True,
            "question": json_result["question"],
            "ground_truth": json_result["ground_truth"],
            "analysis": analysis,
            "normalized_prediction": normalized_prediction,
            "normalized_ground_truth": normalized_ground_truth,
            "is_correct": is_correct,
            "inference_time": inference_time,
            "task": sample.get("task", "unknown"),
            "segmentation_used": result.get('segmentation_used', False),
            "segmentation_info": segmentation_info
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
    max_samples: int = None,
    use_mock_sam: bool = True
) -> Dict[str, Any]:
    """评估BLINK数据集
    
    Args:
        data_path: 数据文件路径
        image_base_path: 图像基础路径
        model: 使用的模型
        max_samples: 最大评估样本数（用于测试）
        use_mock_sam: 是否使用mock SAM2服务
        
    Returns:
        评估结果字典
    """
    print(f"Loading data from {data_path}")
    data = load_blink_data(data_path)

    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples for evaluation")

    print(f"Evaluating {len(data)} samples with {model}")
    print(f"Using {'mock' if use_mock_sam else 'real'} SAM2 service")

    workflow = SAM2QAWorkflow(use_mock_sam=use_mock_sam, use_dino=False)

    results = []
    correct_count = 0
    total_time = 0
    segmentation_used_count = 0

    # 使用tqdm显示进度
    for sample in tqdm(data, desc="Evaluating"):
        result = evaluate_single_sample(sample, image_base_path, model, workflow=workflow)
        results.append(result)
        
        if result["success"]:
            if result["is_correct"]:
                correct_count += 1
            total_time += result["inference_time"]
            if result.get("segmentation_used", False):
                segmentation_used_count += 1
    
    # 计算统计信息
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    accuracy = correct_count / len(successful_results) if successful_results else 0
    avg_inference_time = total_time / len(successful_results) if successful_results else 0
    segmentation_usage_rate = segmentation_used_count / len(successful_results) if successful_results else 0
    
    # 按任务类型分组统计
    task_stats = {}
    for result in successful_results:
        task = result.get("task", "unknown")
        if task not in task_stats:
            task_stats[task] = {
                "correct": 0, 
                "total": 0,
                "segmentation_used": 0
            }
        task_stats[task]["total"] += 1
        if result["is_correct"]:
            task_stats[task]["correct"] += 1
        if result.get("segmentation_used", False):
            task_stats[task]["segmentation_used"] += 1
    
    # 计算每个任务的准确率和分割使用率
    for task in task_stats:
        stats = task_stats[task]
        stats["accuracy"] = stats["correct"] / stats["total"]
        stats["segmentation_usage_rate"] = stats["segmentation_used"] / stats["total"]
    
    return {
        "total_samples": len(data),
        "successful_samples": len(successful_results),
        "failed_samples": len(failed_results),
        "overall_accuracy": accuracy,
        "average_inference_time": avg_inference_time,
        "total_inference_time": total_time,
        "segmentation_usage_rate": segmentation_usage_rate,
        "segmentation_used_count": segmentation_used_count,
        "task_statistics": task_stats,
        "failed_samples_details": failed_results,
        "model": model,
        "sam2_mode": "mock" if use_mock_sam else "real"
    }

def main():
    """主函数"""
    # 配置路径
    data_path = "dataset/blink_data.jsonl"
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
    max_samples = 20  # 设置为数字可以限制评估样本数（用于测试）
    use_mock_sam = False  # 使用mock SAM2服务进行测试
    
    # 执行评估
    results = evaluate_blink_dataset(
        data_path=data_path,
        image_base_path=image_base_path,
        model=model,
        max_samples=max_samples,
        use_mock_sam=use_mock_sam
    )
    
    # 打印结果
    print_evaluation_results(results)
    
    # 保存结果到文件
    output_file = f"blink_evaluation_results_{model.replace('-', '_')}_sam2{'_mock' if use_mock_sam else ''}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 