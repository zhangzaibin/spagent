import pickle
import base64
import requests
import json
import logging
import cv2
import numpy as np
import os
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenPIClient:
    def __init__(self, server_url):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址，如 'http://localhost:5000'
        """
        self.server_url = server_url.rstrip('/')
    
    def health_check(self):
        """检查服务器健康状态"""
        try:
            logger.info("正在检查服务器状态...")
            response = requests.get(f'{self.server_url}/health', timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return None
    
    def test_infer(self):
        """使用服务器内置的测试数据进行推理"""
        try:
            logger.info("发送测试推理请求...")
            response = requests.get(f'{self.server_url}/test', timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info("测试推理完成")
            return result
        except Exception as e:
            logger.error(f"测试推理失败: {e}")
            return None
    
    def infer(self, image_path):
        """
        发送图片进行深度估计
        
        Args:
            image_path: 图片路径
            
        Returns:
            推理结果，如果失败则返回None
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
            
            # 将图片编码为base64
            _, buffer = cv2.imencode('.jpg', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 准备请求数据
            data = {
                'image': image_b64,
                'input_size': 518,
                'return_colored': True
            }
            
            logger.info("发送推理请求...")
            # 发送POST请求
            response = requests.post(
                f'{self.server_url}/infer',
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            if result.get('success'):
                # 解码深度图
                depth_bytes = base64.b64decode(result['depth_map'])
                depth_array = cv2.imdecode(
                    np.frombuffer(depth_bytes, np.uint8),
                    cv2.IMREAD_COLOR if result.get('return_colored', True) else cv2.IMREAD_GRAYSCALE
                )
                
                # 生成输出文件名（基于输入文件名）
                input_filename = os.path.basename(image_path)
                output_filename = f"depth_{input_filename}"
                if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    output_filename += '.png'
                
                # 保存结果
                cv2.imwrite(output_filename, depth_array)
                logger.info(f"深度图已保存至: {output_filename}")
                
                return {
                    'depth_array': depth_array,
                    'shape': result['shape'],
                    'output_path': output_filename
                }
            else:
                logger.error(f"服务器返回错误: {result.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"推理请求失败: {e}")
            return None

def main():
    """主程序"""
    # 服务器地址
    SERVER_URL = "http://localhost:5000"
    
    # 创建客户端
    client = OpenPIClient(SERVER_URL)
    
    # 1. 健康检查
    logger.info("\n=== 执行健康检查 ===")
    health = client.health_check()
    if health:
        logger.info(f"服务器状态: {health}")
    else:
        logger.error("服务器健康检查失败，退出程序")
        return
    
    # 等待一下确保服务器准备好
    time.sleep(2)
    
    # 2. 测试推理
    logger.info("\n=== 执行测试推理 ===")
    test_result = client.test_infer()
    if test_result:
        logger.info(f"测试推理成功: {test_result}")
    else:
        logger.error("测试推理失败，退出程序")
        return
    
    # 等待一下确保服务器准备好
    time.sleep(2)
    
    # 3. 处理实际图片
    logger.info("\n=== 处理图片 ===")
    image_path = "/home/ubuntu/projects/VLMEvalKit/data/BLINK/0a1e41153f247f60533dbea180e06cfbba7538415209f91f93fdd648065757a4.jpg"
    result = client.infer(image_path)
    
    if result:
        logger.info("图片处理成功！")
        logger.info(f"- 输入图片: {image_path}")
        logger.info(f"- 输出图片: {result['output_path']}")
        logger.info(f"- 图片尺寸: {result['shape']}")
    else:
        logger.error("图片处理失败")

if __name__ == '__main__':
    main()