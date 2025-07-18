import json
import logging
import os
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
from external_experts.SAM2.sam2_client import SAM2Client

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from vllm_models.gpt import gpt_single_image_inference, gpt_multiple_images_inference, gpt_text_only_inference

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMockClient:
    """简单的Mock客户端，用于测试"""
    
    def infer(self, image_path: str, prompts: Optional[Dict] = None):
        """模拟图像分割"""
        import cv2
        import numpy as np
        
        # 创建一个简单的分割掩码
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # 生成模拟掩码（简单的圆形区域）
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        radius = min(image.shape[0], image.shape[1]) // 4
        cv2.circle(mask, center, radius, (255,), -1)
        
        # 创建可视化结果
        vis_image = image.copy()
        # 创建与掩码区域相同大小的红色overlay
        overlay = np.zeros_like(vis_image)
        overlay[mask > 0] = [0, 0, 255]  # BGR格式，红色
        # 在掩码区域进行图像混合
        vis_image = cv2.addWeighted(vis_image, 0.5, overlay, 0.5, 0)
        
        # 保存结果
        mask_path = f"assets/mock_mask_{os.path.basename(image_path)}"
        vis_path = f"assets/mock_vis_{os.path.basename(image_path)}"
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(vis_path, vis_image)
        
        return {
            'success': True,
            'mask_path': mask_path,
            'vis_path': vis_path,
            'shape': mask.shape
        }

class SAM2QAWorkflow:
    """SAM2图像分割问答工作流"""
    
    def __init__(self, use_mock_sam: bool = True):
        """
        初始化工作流
        
        Args:
            use_mock_sam: 是否使用mock SAM2服务
        """
        self.use_mock_sam = use_mock_sam
        
        # 初始化SAM2客户端
        if use_mock_sam:
            # 使用简单的mock客户端
            self.sam_client = SimpleMockClient()
            logger.info("使用Mock SAM2服务")
        else:
            # 使用真实的SAM2客户端
            try:
                self.sam_client = SAM2Client("http://127.0.0.1:5000")
                logger.info("使用真实SAM2服务")
            except ImportError:
                logger.error("无法导入真实SAM2客户端")
                raise

    def call_sam_segmentation(self, image_path: str, prompts: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        调用SAM2分割专家
        
        Args:
            image_path: 输入图片路径
            prompts: 提示信息（点击坐标、框选区域等）
            
        Returns:
            分割结果
        """
        logger.info(f"调用SAM2专家处理图片: {image_path}")
        
        try:
            # 执行图像分割
            result = self.sam_client.infer(image_path, prompts)
            
            if result and result.get('success'):
                logger.info("图像分割完成")
                return result
            else:
                logger.error("图像分割失败")
                return None
                
        except Exception as e:
            logger.error(f"SAM2调用异常: {e}")
            return None

    def needs_segmentation_tool(self, response: str) -> bool:
        """
        检查VLLM回答是否需要分割工具
        
        Args:
            response: VLLM的回答
            
        Returns:
            是否需要分割工具
        """
        # 简单的关键词检测
        segmentation_keywords = [
            "segment", "mask", "region", "object", "foreground", "background",
            "area", "part", "portion", "section", "separate", "isolate",
            "extract", "select", "highlight"
        ]
        
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in segmentation_keywords)

    def extract_coordinates_from_response(self, response: str) -> Optional[Dict]:
        """
        从VLLM回答中提取坐标信息
        
        Args:
            response: VLLM的回答
            
        Returns:
            提取的坐标信息
        """
        # TODO: 使用更复杂的方法从回答中提取坐标
        # 目前使用默认坐标
        return {
            'point_coords': [[900, 540]],  # 默认点击坐标
            'point_labels': [1]  # 1表示前景点
        }

    def run_workflow(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        运行完整的SAM2 QA工作流
        
        Args:
            image_path: 输入图像路径
            question: 用户问题
            
        Returns:
            完整的工作流结果
        """
        logger.info("开始执行SAM2 QA工作流")
        
        # 1. VLLM先回答
        initial_response = gpt_single_image_inference(
            image_path=image_path,
            prompt=question,
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # 2. 检查是否需要分割工具
        if self.needs_segmentation_tool(initial_response):
            logger.info("VLLM需要分割工具，调用SAM2")
            
            # 从回答中提取坐标信息
            prompts = self.extract_coordinates_from_response(initial_response)
            
            # 执行分割
            sam_result = self.call_sam_segmentation(image_path, prompts)
            
            if sam_result and sam_result.get('vis_path'):
                # 3. 重新给VLLM回答，同时传入原图和分割结果
                final_response = gpt_multiple_images_inference(
                    image_paths=[image_path, sam_result['vis_path']],
                    prompt=f"{question}\n\nThe first image is the original image, and the second image shows the segmentation result. Please provide a detailed answer using both the original image and segmentation information.",
                    model="gpt-4o-mini",
                    temperature=0.7
                )
            else:
                logger.warning("图像分割失败，使用原始回答")
                sam_result = None
                final_response = initial_response
        else:
            logger.info("VLLM不需要分割工具")
            sam_result = None
            final_response = initial_response
        
        # 4. 生成输出
        return {
            "answer": final_response,
            "segmentation_used": sam_result is not None,
            "segmentation_result": sam_result
        }

def infer(image_path: str, question: str, use_mock_sam: bool = True) -> Dict[str, Any]:
    """
    简单的推理接口
    
    Args:
        image_path: 输入图像路径
        question: 用户问题
        use_mock_sam: 是否使用mock SAM2服务
        
    Returns:
        推理结果
    """
    workflow = SAM2QAWorkflow(use_mock_sam=use_mock_sam)
    return workflow.run_workflow(image_path, question)

def main():
    """主程序示例"""
    # 创建工作流实例
    workflow = SAM2QAWorkflow(use_mock_sam=False)
    
    # 示例图像路径和问题
    image_path = "assets/example.png"  # 替换为实际的测试图片路径
    question = "Can you segment the main object in this image? If you can't segment it, please output your most interest object in the image and give out bounding box of the object. The bounding box format please following this [x1,y1,x2,y2]"
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        logger.warning(f"Example image does not exist: {image_path}")
        logger.info("Please provide a valid image path for testing")
        return
    
    # 运行工作流
    result = workflow.run_workflow(image_path, question)
    
    # 输出结果
    print("\n=== SAM2 QA Workflow Results ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 