import cv2
import numpy as np
import os
import logging
import base64
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockDepthService:
    """Mock depth estimation service that simulates the behavior of the real API"""
    
    def __init__(self):
        """初始化mock服务"""
        self.is_healthy = True
    
    def health_check(self) -> Dict[str, Any]:
        """检查服务健康状态"""
        logger.info("Mock服务健康检查")
        return {
            "status": "healthy",
            "service": "mock_depth_estimation",
            "message": "Mock service is running"
        }
    
    def test_infer(self) -> Dict[str, Any]:
        """使用内置测试数据进行推理"""
        logger.info("Mock测试推理")
        # 创建一个简单的测试深度图
        test_depth = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        test_depth_colored = cv2.applyColorMap(test_depth, cv2.COLORMAP_MAGMA)
        
        # 保存测试结果
        test_output_path = "mock_test_depth.png"
        cv2.imwrite(test_output_path, test_depth_colored)
        
        return {
            "success": True,
            "output_path": test_output_path,
            "shape": test_depth_colored.shape,
            "message": "Mock test inference completed"
        }
    
    def infer(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        模拟深度估计推理
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            推理结果字典，包含深度图和相关信息
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"图片文件不存在: {image_path}")
                return None
            
            # 读取图片
            logger.info(f"读取图片: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                return None
            
            logger.info(f"图片尺寸: {image.shape}")
            
            # 模拟深度估计过程
            # 这里我们创建一个基于图像亮度的简单深度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 使用高斯模糊和边缘检测来模拟深度信息
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # 创建深度图（基于亮度和边缘信息）
            depth_map = cv2.addWeighted(blurred, 0.7, edges, 0.3, 0)
            
            # 应用颜色映射
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
            
            # 生成输出文件名
            input_filename = os.path.basename(image_path)
            output_filename = f"mock_depth_{input_filename}"
            if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_filename += '.png'
            
            # 保存结果
            cv2.imwrite(output_filename, depth_colored)
            logger.info(f"Mock深度图已保存至: {output_filename}")
            
            return {
                'depth_array': depth_colored,
                'depth_raw': depth_map,
                'shape': depth_colored.shape,
                'output_path': output_filename,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Mock推理失败: {e}")
            return None

# 为了保持与原始client接口一致，创建一个兼容的类
class MockOpenPIClient:
    """Mock客户端，保持与OpenPIClient相同的接口"""
    
    def __init__(self, server_url: str = "mock://localhost:5000"):
        """
        初始化mock客户端
        
        Args:
            server_url: 服务器地址（mock模式下不使用）
        """
        self.server_url = server_url
        self.service = MockDepthService()
    
    def health_check(self):
        """检查服务器健康状态"""
        return self.service.health_check()
    
    def test_infer(self):
        """使用服务器内置的测试数据进行推理"""
        return self.service.test_infer()
    
    def infer(self, image_path):
        """
        发送图片进行深度估计
        
        Args:
            image_path: 图片路径
            
        Returns:
            推理结果，如果失败则返回None
        """
        return self.service.infer(image_path)

if __name__ == "__main__":
    # 测试mock服务
    client = MockOpenPIClient()
    
    # 健康检查
    health = client.health_check()
    print("健康检查结果:", health)
    
    # 测试推理
    test_result = client.test_infer()
    print("测试推理结果:", test_result)
    
    # 如果有测试图片，可以进行实际推理
    # result = client.infer("path/to/test/image.jpg")
    # print("实际推理结果:", result) 