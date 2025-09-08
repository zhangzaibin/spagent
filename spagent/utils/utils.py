from typing import List, Dict, Any, Tuple
import json
import os
import pandas as pd


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

def validate_sample_paths(
    sample: Dict[str, Any],
    base_path: str,
    required_field: str = "image"
) -> Tuple[bool, Dict[str, Any]]:
    """验证样本路径和对话
    
    Args:
        sample: 数据样本
        base_path: 基础路径
        required_field: 需要验证的字段名称（"image" 或 "video"）
        
    Returns:
        (是否有效, 错误信息字典)
    """
    # 提取路径
    paths = sample.get(required_field, [])
    if not paths:
        return False, {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": f"No {required_field} found"
        }
    
    # 验证所有路径是否存在
    full_paths = []
    missing_paths = []
    
    for path in paths:
        full_path = os.path.join(base_path, path)
        full_paths.append(full_path)
        
        if not os.path.exists(full_path):
            missing_paths.append(full_path)
    
    # 如果有路径不存在，返回错误
    if missing_paths:
        return False, {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": f"{required_field.capitalize()} not found: {missing_paths}"
        }
    
    # 提取问题和答案
    conversation = sample.get("conversations", [])
    if not conversation:
        return False, {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "No conversation found"
        }
    
    question, ground_truth = extract_question_and_answer(conversation)
    if not question or not ground_truth:
        return False, {
            "id": sample.get("id", "unknown"),
            "success": False,
            "error": "Question or answer not found"
        }
    
    # 返回验证成功和路径信息
    return True, {
        "path": full_paths,
        "question": question,
        "ground_truth": ground_truth
    }

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

def normalize_answer(answer: str) -> tuple[str, str]:
    """Normalize answer format
    
    Args:
        answer: Original answer string
        
    Returns:
        Tuple (analysis, final_answer): Analysis content and normalized answer
    """
    original_answer = answer.strip()
    
    # Extract answer part
    processed_answer = original_answer
    answer_start = processed_answer.find("<answer>")
    answer_end = processed_answer.find("</answer>")
    
    if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
        processed_answer = processed_answer[answer_start+8:answer_end].strip()
    
    # Extract option letter if present
    final_answer = ""
    # Try to extract from formats like "(B) ..." or "B. ..."
    import re
    match = re.search(r'\(([A-D])\)|([A-D])\.', processed_answer)
    if match:
        final_answer = match.group(1) or match.group(2)
    else:
        # If no parenthesis format, look for option letters directly
        for char in processed_answer:
            if char in ['A', 'B', 'C', 'D']:
                final_answer = char
                break
    
    # If no option letter found, return the processed answer
    if not final_answer:
        final_answer = processed_answer
    
    # Since we no longer have <analysis> tags, return empty string for analysis
    return "", final_answer


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


def save_error_to_csv(error_data: Dict[str, Any], csv_file: str = "error_analysis.csv"):
    """保存错误信息到CSV文件
    
    Args:
        error_data: 包含错误信息的字典
        csv_file: CSV文件名
    """
    # 定义列名
    columns = ['question', 'path', 'is_correct', 'analysis', 'normalized_prediction', 'normalized_ground_truth', 'used_tools', 'follow_up_prompt']
    
    # 准备数据行
    row_data = {
        'question': error_data.get('question', ''),
        'path': error_data.get('path', ''),
        'is_correct': error_data.get('is_correct', ''),
        'analysis': error_data.get('analysis', ''),
        'normalized_prediction': error_data.get('normalized_prediction', ''),
        'normalized_ground_truth': error_data.get('normalized_ground_truth', ''),
        'used_tools': error_data.get('used_tools', ''),
        'follow_up_prompt': error_data.get('follow_up_prompt', '')
    }
    
    # 检查文件是否存在
    if os.path.exists(csv_file):
        # 如果文件存在，追加数据
        df_existing = pd.read_csv(csv_file)
        df_new = pd.DataFrame([row_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        # 如果文件不存在，创建新文件
        df_new = pd.DataFrame([row_data])
        df_new.to_csv(csv_file, index=False)

def extract_objects_from_response(response: str) -> list:
    """
    从回答中提取<object></object>标签包含的物体列表
    
    Args:
        response: VLLM的回答文本
        
    Returns:
        提取的物体列表
    """
    import logging
    import re
    
    logger = logging.getLogger(__name__)
    objects = []
    try:
        # 查找所有带编号的object标签对
        pattern = r'<object_\d+>(.*?)</object_\d+>'
        matches = re.findall(pattern, response)
        
        # 清理并添加到列表
        for match in matches:
            obj = match.strip()
            if obj:  # 只添加非空物体
                objects.append(obj)
                
        logger.info(f"从回答中提取到 {len(objects)} 个物体: {objects}")
    except Exception as e:
        logger.error(f"提取物体时出错: {e}")
    
    return objects

def draw_boxes_on_image(image_path: str, prompts: Dict, output_path: str = None) -> str:
    """
    在图像上绘制边界框
    
    Args:
        image_path: 输入图像路径
        prompts: 包含边界框坐标的字典，格式如 {'box': [[x1,y1,x2,y2], ...], 'labels': ['person', 'kite', ...]}
        output_path: 输出图像路径，如果为None则自动生成
        
    Returns:
        输出图像的路径
    """
    import cv2
    import numpy as np
    import os
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"outputs/boxes_{base_name}.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 绘制边界框
    if 'box' in prompts and prompts['box']:
        boxes = prompts['box']
        labels = prompts.get('labels', [])  # 获取标签列表，如果没有则为空列表
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签（优先使用类别标签，否则使用默认标签）
            if i < len(labels) and labels[i]:
                label = labels[i]
            else:
                label = f"Box {i+1}"
            
            # 计算标签文本大小
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(image, (x1, y1-label_size[1]-10), (x1+label_size[0]+10, y1), color, -1)
            
            # 绘制标签文本
            cv2.putText(image, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存图像
    cv2.imwrite(output_path, image)
    
    return output_path

def parse_json(json_output: str) -> str:
    """
    Parse JSON output by removing markdown fencing
    Based on qwen's official implementation
    
    Args:
        json_output: Raw response that may contain ```json fencing
        
    Returns:
        Clean JSON string
    """
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output