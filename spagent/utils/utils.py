from typing import List, Dict, Any, Tuple
import json

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

def normalize_answer(answer: str) -> tuple[str, str]:
    """标准化答案格式
    
    Args:
        answer: 原始答案
        
    Returns:
        元组 (analysis, final_answer): 分析内容和标准化后的答案
    """
    original_answer = answer.strip()
    
    # 提取分析部分
    analysis = ""
    analysis_start = original_answer.find("<analysis>")
    analysis_end = original_answer.find("</analysis>")
    if analysis_start != -1 and analysis_end != -1 and analysis_end > analysis_start:
        analysis = original_answer[analysis_start+10:analysis_end].strip()
        original_answer = original_answer[:analysis_start] + original_answer[analysis_end+11:]
    
    # 提取答案部分
    processed_answer = original_answer.strip()
    answer_start = processed_answer.find("<answer>")
    answer_end = processed_answer.find("</answer>")
    if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
        processed_answer = processed_answer[answer_start+8:answer_end].strip()
    
    # 提取选项字母
    final_answer = ""
    # 首先尝试从类似"(B) ..."的格式中提取
    import re
    match = re.search(r'\(([A-D])\)|([A-D])\.', processed_answer)  # 匹配(A)和A.两种形式的答案
    if match:
        final_answer = match.group(1) or match.group(2)
    else:
        # 如果没有括号格式，直接查找选项字母
        for char in processed_answer:
            if char in ['A', 'B', 'C', 'D']:
                final_answer = char
                break
    
    # 如果没有找到选项字母，返回处理后的答案或空字符串
    if not final_answer:
        final_answer = processed_answer
    
    return analysis, final_answer


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