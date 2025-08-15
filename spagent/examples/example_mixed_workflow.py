#!/usr/bin/env python3
"""
Mixed Expert Workflow Example

This script demonstrates how to use the mixed expert workflow that combines
Depth Anything V2, SAM2, and GroundingDINO experts.

Usage:
    python example_mixed_workflow.py [image_path] [question]
"""

import sys
import os
from pathlib import Path


# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from workflows.mix_workflow import MixedExpertWorkflow, infer


def main():
    """主函数"""
    
    # 默认参数
    default_image = "path/to/your/test_image.jpg"
    default_question = "分析这张图片中物体的深度关系，并检测和分割主要对象"
    
    # 从命令行参数获取输入
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image
        print(f"使用默认图片路径: {image_path}")
    
    if len(sys.argv) > 2:
        question = sys.argv[2]
    else:
        question = default_question
        print(f"使用默认问题: {question}")
    
    print("\n" + "="*60)
    print("Mixed Expert Workflow Demo")
    print("="*60)
    print(f"图片路径: {image_path}")
    print(f"问题: {question}")
    print("="*60)
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        print("请提供有效的图片路径作为第一个参数")
        return
    
    try:
        # 使用简化接口
        print("\n[简化接口示例]")
        result = infer(image_path, question)
        
        print(f"\n最终答案: {result['answer']}")
        print(f"\n初始回答: {result['initial_response']}")
        print(f"\n使用的工具: {result['used_tools']}")
        print(f"\n生成的额外图像: {result['additional_images']}")
        
        # 详细的专家结果
        if result['expert_results']:
            print("\n" + "-"*40)
            print("专家结果详情:")
            print("-"*40)
            
            # 深度估计结果
            if 'depth' in result['expert_results']:
                depth_result = result['expert_results']['depth']
                print(f"深度估计: ✓")
                print(f"  - 输出路径: {depth_result['output_path']}")
                print(f"  - 图像尺寸: {depth_result['shape']}")
            
            # SAM2分割结果
            if 'sam2' in result['expert_results']:
                sam2_result = result['expert_results']['sam2']
                print(f"SAM2分割: ✓")
                print(f"  - 可视化路径: {sam2_result['vis_path']}")
                print(f"  - 掩码尺寸: {sam2_result['shape']}")
            
            # GroundingDINO检测结果
            if 'gdino' in result['expert_results']:
                gdino_results = result['expert_results']['gdino']
                print(f"GroundingDINO检测: ✓ ({len(gdino_results)} 个检测任务)")
                for i, gdino_result in enumerate(gdino_results):
                    print(f"  - 任务 {i+1}: {gdino_result}")
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)


def demo_advanced_usage():
    """高级用法演示"""
    print("\n[高级接口示例]")
    
    # 创建工作流实例
    workflow = MixedExpertWorkflow(ip="10.8.131.51", use_mock=True)
    
    # 运行工作流
    image_path = "path/to/your/test_image.jpg"
    question = "这张图片中有什么物体？它们的深度关系如何？请分割出主要对象。"
    
    result = workflow.run_workflow(image_path, question)
    print(f"工作流结果: {result}")


if __name__ == "__main__":
    main()
    
    # 可选：演示高级用法
    if len(sys.argv) > 3 and sys.argv[3] == "--advanced":
        demo_advanced_usage() 