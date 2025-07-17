import json
import logging
import os
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from vllm_models.gpt import gpt_single_image_inference, gpt_multiple_images_inference, gpt_text_only_inference

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMockClient:
    """简单的Mock客户端，用于测试"""
    
    def infer(self, image_path: str):
        """模拟深度估计"""
        import cv2
        import numpy as np
        
        # 创建一个简单的深度图
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # 生成模拟深度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
        
        # 保存深度图
        output_path = f"mock_depth_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, depth_map)
        
        return {
            'success': True,
            'output_path': output_path,
            'shape': depth_map.shape
        }

class DepthQAWorkflow:
    """深度估计问答工作流"""
    
    def __init__(self, use_mock_depth: bool = True):
        """
        初始化工作流
        
        Args:
            use_mock_depth: 是否使用mock深度估计服务
        """
        self.use_mock_depth = use_mock_depth
        
        # 初始化深度估计客户端
        if use_mock_depth:
            # 动态导入mock服务
            try:
                from tmp.mock_depth_service import MockOpenPIClient
                self.depth_client = MockOpenPIClient()
                logger.info("使用Mock深度估计服务")
            except ImportError:
                # 如果导入失败，创建一个简单的mock客户端
                logger.warning("无法导入MockOpenPIClient，使用简单mock")
                self.depth_client = SimpleMockClient()
        else:
            # 这里可以替换为真实的深度估计客户端
            try:
                from external_experts.Depth_AnythingV2.depth_client import OpenPIClient
                self.depth_client = OpenPIClient("http://localhost:5000")
                logger.info("使用真实深度估计服务")
            except ImportError:
                logger.error("无法导入真实深度估计客户端")
                raise

    def call_depth_estimation(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        调用深度估计专家
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            深度估计结果
        """
        logger.info(f"调用深度估计专家处理图片: {image_path}")
        
        try:
            # 执行深度估计
            result = self.depth_client.infer(image_path)
            
            if result and result.get('success'):
                logger.info("深度估计完成")
                return result
            else:
                logger.error("深度估计失败")
                return None
                
        except Exception as e:
            logger.error(f"深度估计调用异常: {e}")
            return None

    def needs_depth_tool(self, response: str) -> bool:
        """
        检查VLLM回答是否需要深度工具
        
        Args:
            response: VLLM的回答
            
        Returns:
            是否需要深度工具
        """
        # 简单的关键词检测
        depth_keywords = [
            "depth", "distance", "3d", "spatial", "foreground", "background",
            "closer", "farther", "depth map", "depth estimation"
        ]
        
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in depth_keywords)

    def run_workflow(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        运行完整的深度估计QA工作流
        
        Args:
            image_path: 输入图像路径
            question: 用户问题
            
        Returns:
            完整的工作流结果
        """
        logger.info("开始执行深度估计QA工作流")
        
        # 1. VLLM先回答
        initial_response = gpt_single_image_inference(
            image_path=image_path,
            prompt=question,
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # 2. 检查是否需要深度工具
        if self.needs_depth_tool(initial_response):
            logger.info("VLLM需要深度工具，调用深度估计")
            depth_result = self.call_depth_estimation(image_path)
            
            if depth_result and depth_result.get('output_path'):
                # 3. 重新给VLLM回答，同时传入原图和深度图
                final_response = gpt_multiple_images_inference(
                    image_paths=[image_path, depth_result['output_path']],
                    prompt=f"{question}\n\nThe first image is the original image, and the second image is the depth map. Please provide a detailed answer using both the original image and depth information.",
                    model="gpt-4o-mini",
                    temperature=0.7
                )
            else:
                logger.warning("深度估计失败，使用原始回答")
                depth_result = None
                final_response = initial_response
        else:
            logger.info("VLLM不需要深度工具")
            depth_result = None
            final_response = initial_response
        
        # 4. 生成输出
        return {
            "answer": final_response,
            "depth_used": depth_result is not None,
            "depth_result": depth_result
        }

def infer(image_path: str, question: str, use_mock_depth: bool = True) -> Dict[str, Any]:
    """
    简单的推理接口
    
    Args:
        image_path: 输入图像路径
        question: 用户问题
        use_mock_depth: 是否使用mock深度估计服务
        
    Returns:
        推理结果
    """
    workflow = DepthQAWorkflow(use_mock_depth=use_mock_depth)
    return workflow.run_workflow(image_path, question)

def main():
    """主程序示例"""
    # 创建工作流实例
    workflow = DepthQAWorkflow(use_mock_depth=True)
    
    # 示例图像路径和问题
    image_path = "assets/image.png"  # Please replace with actual image path
    question = "How is the depth distribution of objects in this image? Is there a significant depth difference between foreground and background?"
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        logger.warning(f"Example image does not exist: {image_path}")
        logger.info("Please provide a valid image path for testing")
        return
    
    # 运行工作流
    result = workflow.run_workflow(image_path, question)
    
    # 输出结果
    print("\n=== Depth Estimation QA Workflow Results ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 