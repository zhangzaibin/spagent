# 集成了service类和client类

import cv2
import numpy as np
import os
import logging
import base64
import torch
import requests
import time
from typing import Optional, Dict, Any
from PIL import Image
import traceback
from depth_anything_v2.dpt import DepthAnythingV2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class DepthService:
    """Depth estimation service that provides both real and mock implementations"""
    
    def __init__(self, use_mock: bool = False, encoder: str = 'vitb', checkpoint_path: Optional[str] = None):
        """
        初始化深度估计服务
        
        Args:
            use_mock: 是否使用mock服务
            encoder: 编码器类型 ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: 模型权重文件路径
        """
        self.use_mock = use_mock
        self.model = None
        self.server_url = "http://localhost:5000"
        
        if not use_mock:
            self.load_model(encoder, checkpoint_path)
    
    def load_model(self, encoder: str = 'vitb', checkpoint_path: Optional[str] = None) -> bool:
        """加载深度估计模型"""
        try:
            logger.info(f"正在加载模型（编码器类型：{encoder}）...")
            
            # 创建模型实例
            logger.info(f"正在创建模型实例，配置：{model_configs[encoder]}")
            self.model = DepthAnythingV2(**model_configs[encoder])
            
            # 加载检查点
            if checkpoint_path:
                logger.info(f"正在从 {checkpoint_path} 加载模型权重...")
                try:
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    self.model.load_state_dict(state_dict)
                    logger.info("模型权重加载成功")
                except Exception as e:
                    logger.error(f"加载模型权重失败：{e}")
                    return False
                
            # 设置设备和评估模式
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            logger.info(f"使用设备：{device}")
            
            self.model = self.model.to(device).eval()
            
            # 使用测试输入验证模型
            try:
                logger.info("正在使用测试数据验证模型...")
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                logger.info("模型验证成功！")
            except Exception as e:
                logger.error(f"模型验证失败：{e}")
                return False
            
            logger.info("模型加载完成！")
            return True
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """检查服务健康状态"""
        if self.use_mock:
            logger.info("Mock服务健康检查")
            return {
                "status": "healthy",
                "service": "mock_depth_estimation",
                "message": "Mock service is running"
            }
        
        try:
            model_type = None
            if self.model is not None:
                model_type = self.model.encoder if hasattr(self.model, 'encoder') else '未知'
                
            status = {
                "status": "健康",
                "model_loaded": self.model is not None,
                "model_type": model_type,
                "device": str(next(self.model.parameters()).device) if self.model is not None else None
            }
            logger.info(f"健康检查结果：{status}")
            return status
        except Exception as e:
            logger.error(f"健康检查失败：{e}")
            return {
                "status": "不健康",
                "error": str(e)
            }
    
    def test_infer(self) -> Dict[str, Any]:
        """使用内置测试数据进行推理"""
        if self.use_mock:
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
        
        try:
            # 检查模型是否已加载
            if self.model is None:
                logger.error("模型未加载")
                return {
                    "success": False,
                    "error": "模型未加载"
                }
                
            # 创建测试图像（黑白渐变）
            logger.info("正在创建测试图像...")
            test_image = np.zeros((256, 256, 3), dtype=np.uint8)
            for i in range(256):
                test_image[:, i] = i
            
            # 运行推理
            logger.info("正在进行测试推理...")
            with torch.no_grad():
                depth = self.model.infer_image(test_image)
                logger.info(f"推理成功，输出尺寸：{depth.shape}")
            
            logger.info("测试推理完成")
            return {
                "success": True,
                "message": "测试推理成功",
                "shape": list(depth.shape)
            }
            
        except Exception as e:
            logger.error(f"测试推理失败：{e}")
            logger.error(f"错误追踪：{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def infer(self, image_path: str, input_size: int = 518, return_colored: bool = True) -> Optional[Dict[str, Any]]:
        """
        执行深度估计推理
        
        Args:
            image_path: 输入图片路径
            input_size: 输入图片大小
            return_colored: 是否返回彩色深度图
            
        Returns:
            推理结果字典，包含深度图和相关信息
        """
        if self.use_mock:
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
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                edges = cv2.Canny(blurred, 50, 150)
                depth_map = cv2.addWeighted(blurred, 0.7, edges, 0.3, 0)
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
        
        try:
            # 检查模型是否已加载
            if self.model is None:
                logger.error("模型未加载")
                return None
                
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
            
            # 运行推理
            logger.info("正在进行深度估计...")
            with torch.no_grad():
                depth = self.model.infer_image(image, input_size)
            
            # 归一化深度图
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            # 如果需要，应用颜色映射
            if return_colored:
                import matplotlib
                cmap = matplotlib.colormaps.get_cmap('Spectral_r')
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            # 生成输出文件名
            input_filename = os.path.basename(image_path)
            output_filename = f"outputs/depth_{input_filename}"
            if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_filename += '.png'
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            # 保存结果
            success = cv2.imwrite(output_filename, depth)
            if not success:
                logger.error(f"无法保存深度图到: {output_filename}")
                return None
                
            logger.info(f"深度图已保存至: {output_filename}")
            
            return {
                'depth_array': depth,
                'shape': depth.shape,
                'output_path': output_filename,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            logger.error(f"错误追踪：{traceback.format_exc()}")
            return None

class DepthClient:
    """Depth estimation client that provides a unified interface for both real and mock services"""
    
    def __init__(self, server_url: str = "http://localhost:5000", use_mock: bool = False, encoder: str = 'vitb', checkpoint_path: Optional[str] = None):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址，如 'http://localhost:5000'
            use_mock: 是否使用mock服务
            encoder: 编码器类型
            checkpoint_path: 模型权重文件路径
        """
        self.server_url = server_url.rstrip('/')
        self.service = DepthService(use_mock=use_mock, encoder=encoder, checkpoint_path=checkpoint_path)
    
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
    # 测试服务
    # 设置模型路径
    checkpoint_path = 'spagent/external_experts/Depth_AnythingV2/checkpoints/depth_anything_v2_vitb.pth'
    
    # 创建服务器客户端
    client = DepthClient(use_mock=False, encoder='vitb', checkpoint_path=checkpoint_path)  # 使用真实模型进行测试
    
    # 1. 健康检查
    logger.info("\n=== 执行健康检查 ===")
    health = client.health_check()
    if health:
        logger.info(f"服务器状态: {health}")
    else:
        logger.error("服务器健康检查失败，退出程序")
        exit(1)
    
    # 2. 测试推理
    logger.info("\n=== 执行测试推理 ===")
    test_result = client.test_infer()
    if test_result:
        logger.info(f"测试推理成功: {test_result}")
    else:
        logger.error("测试推理失败，退出程序")
        exit(1)
    
    # 3. 处理实际图片
    logger.info("\n=== 处理图片 ===")
    image_path = "assets/example.png"
    result = client.infer(image_path)
    
    if result:
        logger.info("图片处理成功！")
        logger.info(f"- 输入图片: {image_path}")
        logger.info(f"- 输出图片: {result['output_path']}")
        logger.info(f"- 图片尺寸: {result['shape']}")
    else:
        logger.error("图片处理失败")
