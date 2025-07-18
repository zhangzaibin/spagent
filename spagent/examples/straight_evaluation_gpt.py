import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vllm_models.gpt import gpt_single_image_inference

def load_blink_data(data_path: str) -> List[Dict[str, Any]]:
    """加载BLINK数据集
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_question_and_answer(conversation: List[Dict[str, str]]) -> Tuple[str, str]:
    """从对话中提取问题和答案
    
    Args:
        conversation: 对话列表
        
    Returns:
        (问题, 答案) 元组
    """
    # 找到人类的问题
    human_message = None
    for msg in conversation:
        if msg["from"] == "human":
            human_message = msg["value"]
            break
    
    # 找到GPT的答案
    gpt_answer = None
    for msg in conversation:
        if msg["from"] == "gpt":
            gpt_answer = msg["value"]
            break
    
    return human_message, gpt_answer

def normalize_answer(answer: str) -> str:
    """标准化答案格式
    
    Args:
        answer: 原始答案
        
    Returns:
        标准化后的答案
    """
    # 移除空白字符并转换为大写
    answer = answer.strip().upper()
    
    # 提取选项字母 (A, B, C, D)
    if answer in ['A', 'B', 'C', 'D']:
        return answer
    
    # 如果答案包含选项字母，提取第一个
    for char in answer:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    # 如果没有找到选项字母，返回原始答案
    return answer

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
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
        
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
        result = evaluate_single_sample(sample, image_base_path, model)
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
    
    return {
        "total_samples": len(data),
        "successful_samples": len(successful_results),
        "failed_samples": len(failed_results),
        "overall_accuracy": accuracy,
        "average_inference_time": avg_inference_time,
        "total_inference_time": total_time,
        "task_statistics": task_stats,
        "failed_samples_details": failed_results,
        "model": model
    }

def print_evaluation_results(results: Dict[str, Any]):
    """打印评估结果
    
    Args:
        results: 评估结果字典
    """
    print("\n" + "="*60)
    print("BLINK DATASET EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Successful samples: {results['successful_samples']}")
    print(f"Failed samples: {results['failed_samples']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    print(f"Average inference time: {results['average_inference_time']:.2f} seconds")
    print(f"Total inference time: {results['total_inference_time']:.2f} seconds")
    
    print("\nTask-wise Statistics:")
    print("-" * 40)
    for task, stats in results['task_statistics'].items():
        print(f"{task:20s}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    if results['failed_samples_details']:
        print(f"\nFailed samples ({len(results['failed_samples_details'])}):")
        print("-" * 40)
        for failed in results['failed_samples_details'][:5]:  # 只显示前5个
            print(f"ID: {failed['id']}, Error: {failed['error']}")
        if len(results['failed_samples_details']) > 5:
            print(f"... and {len(results['failed_samples_details']) - 5} more")

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
