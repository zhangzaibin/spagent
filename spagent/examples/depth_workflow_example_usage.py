#!/usr/bin/env python3
"""
深度估计QA工作流使用示例

这个脚本演示了如何使用深度估计QA工作流来处理不同类型的图像问答任务。
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from workflows.depth_qa_workflow import DepthQAWorkflow
# import pdb; pdb.set_trace()

def example_depth_questions():
    """深度相关问题的示例"""
    return [
        "Which objects are closest to the camera? We can use depth estimation tool.",
    ]

def example_general_questions():
    """一般视觉问题的示例"""
    return [
        "What objects are in this image?"
    ]

def run_example(workflow, image_path: str, questions: list, question_type: str):
    """运行示例问题"""
    print(f"\n=== {question_type} 示例 ===")
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 50)
        
        try:
            result = workflow.run_workflow(image_path, question)
            
            # 打印关键结果
            print(f"答案: {result['answer'][:200]}...")
            print(f"深度工具使用: {result['depth_used']}")
            
            if result['depth_result']:
                print(f"深度图路径: {result['depth_result'].get('output_path', 'N/A')}")
            
        except Exception as e:
            print(f"处理失败: {e}")

def main():
    """主函数"""
    print("深度估计QA工作流示例")
    print("=" * 50)
    
    # 创建工作流实例（使用mock深度估计）
    workflow = DepthQAWorkflow(use_mock_depth=False)
    
    # 检查工作流健康状态
    # health = workflow.health_check()
    # print(f"工作流健康状态: {health['overall_status']}")
    
    # 示例图像路径（请替换为实际路径）
    image_path = "assets/example.png"
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"Warning: Example image does not exist: {image_path}")
        print("Please provide a valid image path for testing")
        print("Or create a test image file")
        return
    
    # 运行深度相关问题示例
    depth_questions = example_depth_questions()
    run_example(workflow, image_path, depth_questions, "Depth-related")
    
    # 运行一般问题示例
    general_questions = example_general_questions()
    run_example(workflow, image_path, general_questions, "General visual")
    
    print("\n=== Example completed ===")
    print("Note: Currently using mock depth estimation service, depth maps are simulated")


if __name__ == "__main__":
    main() 