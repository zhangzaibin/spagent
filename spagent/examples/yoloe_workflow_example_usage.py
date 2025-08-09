#!/usr/bin/env python3
"""
YOLOEQA工作流使用示例

这个脚本演示了如何使用YOLOEQA工作流来处理不同类型的图像问答任务。
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from workflows.sv_yoloe_qa_workflow import YoloeQAWorkflow
# import pdb; pdb.set_trace()

def example_sv_questions():
    """supervision相关问题的示例"""
    return [
        "What objects can you detect in this image? Please use object detection to identify them.",
    ]

def example_general_questions():
    """一般视觉问题的示例"""
    return [
        "What is the color of the sky in this image?",
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
            print(f"YOLO-E工具使用: {result['yoloe_used']}")
            
            if result['yoloe_result']:
                print(f"任务类型: {result.get('task_type', 'unknown')}")
                print(f"处理后图像路径: {result['yoloe_result'].get('output_path', 'N/A')}")
            
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("yoloe QA工作流示例")
    print("=" * 50)
    
    # 创建工作流实例（使用mock supervision）
    workflow = YoloeQAWorkflow(use_mock=False)
    
    # 示例图像路径（请替换为实际路径）
    image_path = "assets/example.png"
    
    # 检查图像是否存在，如果不存在则使用测试图像
    if not os.path.exists(image_path):
        print(f"Warning: Example image does not exist: {image_path}")
        # 尝试使用数据集中的图像
        test_image_path = "/home/ubuntu/projects/spagent/dataset/BLINK/0a45758ac7376725109cba53848f2387c6f9280686ac8957868ec7f8ce71ba21.jpg"
        if os.path.exists(test_image_path):
            image_path = test_image_path
            print(f"Using test image: {image_path}")
        else:
            print("No test image available")
            return
    
    # 运行supervision相关问题示例
    sv_questions = example_sv_questions()
    run_example(workflow, image_path, sv_questions, "Supervision-related")
    
    # 运行一般问题示例
    general_questions = example_general_questions()
    run_example(workflow, image_path, general_questions, "General visual")
    
    print("\n=== Example completed ===")
    print("Note: Currently using mock supervision service, annotations are simulated")


if __name__ == "__main__":
    main() 