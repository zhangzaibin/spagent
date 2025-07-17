#!/usr/bin/env python3
"""
简单的测试脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_workflow():
    """测试工作流"""
    try:
        # 导入工作流
        from spagent.workflows.depth_qa_workflow import infer
        
        # 创建测试图像路径
        test_image_path = "assets/test_image.png"
        
        # 如果测试图像不存在，创建一个简单的测试图像
        if not os.path.exists(test_image_path):
            print("创建测试图像...")
            create_test_image(test_image_path)
        
        # 测试深度相关问题
        print("=== 测试深度相关问题 ===")
        result1 = infer(
            image_path=test_image_path,
            question="How is the depth distribution of objects in this image?"
        )
        print(f"答案: {result1['answer'][:100]}...")
        print(f"使用深度工具: {result1['depth_used']}")
        
        # 测试一般问题
        print("\n=== 测试一般问题 ===")
        result2 = infer(
            image_path=test_image_path,
            question="What objects are in this image?"
        )
        print(f"答案: {result2['answer'][:100]}...")
        print(f"使用深度工具: {result2['depth_used']}")
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def create_test_image(image_path):
    """创建测试图像"""
    try:
        import cv2
        import numpy as np
        
        # 创建assets目录
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # 创建简单的测试图像
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # 添加一些几何形状
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.circle(image, (300, 100), 50, (0, 255, 0), -1)
        cv2.putText(image, "Test Image", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 保存图像
        cv2.imwrite(image_path, image)
        print(f"测试图像已创建: {image_path}")
        
    except ImportError:
        print("警告: 无法创建测试图像，请手动提供图像文件")
        # 创建一个空的图像文件作为占位符
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, 'w') as f:
            f.write("placeholder")

if __name__ == "__main__":
    print("开始测试深度估计QA工作流...")
    test_workflow() 