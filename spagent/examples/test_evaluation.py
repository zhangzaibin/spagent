#!/usr/bin/env python3
"""
测试评估函数的简单脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from straight_evaluation_gpt import load_blink_data, extract_question_and_answer, normalize_answer

def test_data_loading():
    """测试数据加载功能"""
    print("Testing data loading...")
    
    try:
        data = load_blink_data("dataset/blink_data.jsonl")
        print(f"✓ Successfully loaded {len(data)} samples")
        
        if len(data) > 0:
            sample = data[0]
            print(f"✓ First sample ID: {sample.get('id', 'unknown')}")
            print(f"✓ First sample task: {sample.get('task', 'unknown')}")
            
            # 测试问题提取
            conversation = sample.get("conversations", [])
            question, answer = extract_question_and_answer(conversation)
            print(f"✓ Question: {question[:100]}...")
            print(f"✓ Answer: {answer}")
            
            # 测试答案标准化
            normalized = normalize_answer(answer)
            print(f"✓ Normalized answer: {normalized}")
            
        return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def test_answer_normalization():
    """测试答案标准化功能"""
    print("\nTesting answer normalization...")
    
    test_cases = [
        ("A", "A"),
        ("B", "B"),
        ("C", "C"),
        ("D", "D"),
        (" a ", "A"),
        ("b", "B"),
        ("The answer is C", "C"),
        ("I think it's D.", "D"),
        ("A and B", "A"),
        ("No answer", "NO ANSWER"),
    ]
    
    for input_answer, expected in test_cases:
        result = normalize_answer(input_answer)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_answer}' -> '{result}' (expected: '{expected}')")

def main():
    """主测试函数"""
    print("=" * 50)
    print("BLINK EVALUATION FUNCTION TEST")
    print("=" * 50)
    
    # 测试数据加载
    if not test_data_loading():
        print("Data loading test failed. Please check your data file.")
        return
    
    # 测试答案标准化
    test_answer_normalization()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 