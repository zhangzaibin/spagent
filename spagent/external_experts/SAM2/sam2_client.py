import base64
import requests
import json
import logging
import cv2
import numpy as np
import os
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 添加文件处理器
file_handler = logging.FileHandler('sam_client.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

class SAM2Client:
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
    
    def test(self):
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
    
    def infer(self, image_path, prompts=None):
        """
        发送图片进行分割
        
        Args:
            image_path: 图片路径
            prompts: 提示信息，可包含以下键：
                - point_coords: 点击坐标列表 [[x1, y1], [x2, y2], ...]
                - point_labels: 点标签列表 [1, 1, 0, ...] (1表示前景，0表示背景)
                - box: 框选坐标 [x1, y1, x2, y2]
                - text: 文本描述
            
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
            image_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            # 准备请求数据
            data = {
                'image': image_b64,
                'conf': 0.5  # 置信度阈值
            }
            
            # 添加提示信息
            if prompts:
                data.update(prompts)
            
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
                # 解码掩码
                mask_bytes = base64.b64decode(result['mask'])
                mask_array = cv2.imdecode(
                    np.frombuffer(mask_bytes, np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )
                
                # 生成输出文件名（基于输入文件名）
                input_filename = os.path.basename(image_path)
                output_filename = f"mask_{input_filename}"
                if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    output_filename += '.png'
                
                # 保存结果
                cv2.imwrite(output_filename, mask_array)
                logger.info(f"掩码已保存至: {output_filename}")
                
                # 可视化结果（将掩码叠加到原图上）
                overlay = image.copy()
                colored_mask = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
                colored_mask[mask_array > 0] = [0, 0, 255]  # 红色 ，bgr
                cv2.addWeighted(colored_mask, 0.5, overlay, 0.5, 0, overlay)
                
                # 保存可视化结果
                vis_filename = f"vis_{input_filename}"
                if not vis_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    vis_filename += '.png'
                cv2.imwrite(vis_filename, overlay)
                logger.info(f"可视化结果已保存至: {vis_filename}")
                
                return {
                    'mask_array': mask_array,
                    'shape': result['shape'],
                    'mask_path': output_filename,
                    'vis_path': vis_filename,
                    'success': True
                }
            else:
                logger.error(f"服务器返回错误: {result.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"推理请求失败: {e}")
            return None
    
    def infer_video(self, video_path, prompts=None):
        """
        发送视频进行分割
        
        Args:
            video_path: 视频文件路径
            prompts: 提示信息（用于第一帧），格式同infer方法
            
        Returns:
            处理后的视频路径，如果失败则返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在: {video_path}")
                return None
            
            # 读取视频文件
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            video_b64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # 准备请求数据
            data = {
                'video': video_b64,
                'conf': 0.5  # 置信度阈值
            }
            
            # 添加提示信息
            if prompts:
                data.update(prompts)
            
            logger.info("发送视频处理请求...")
            # 发送POST请求
            response = requests.post(
                f'{self.server_url}/infer_video',
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=300  # 视频处理可能需要更长时间
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            if result.get('success'):
                # 解码视频
                video_bytes = base64.b64decode(result['video'])
                
                # 生成输出文件名
                input_filename = os.path.basename(video_path)
                output_filename = f"mask_{input_filename}"
                if not output_filename.lower().endswith('.mp4'):
                    output_filename += '.mp4'
                
                # 保存视频
                with open(output_filename, 'wb') as f:
                    f.write(video_bytes)
                logger.info(f"处理后的视频已保存至: {output_filename}")
                
                return {
                    'output_path': output_filename,
                    'frames': result['frames'],
                    'fps': result['fps'],
                    'size': result['size']
                }
            else:
                logger.error(f"服务器返回错误: {result.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"视频处理请求失败: {e}")
            return None

def main():
    """主程序"""
    # 服务器地址
    SERVER_URL = "http://127.0.0.1:5000"
    
    # 创建客户端
    client = SAM2Client(SERVER_URL)
    
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
    test_result = client.test()
    if test_result:
        logger.info(f"测试推理成功: {test_result}")
    else:
        logger.error("测试推理失败，退出程序")
        return
    
    # 等待一下确保服务器准备好
    time.sleep(2)
    
    # 3. 处理图片示例
    logger.info("\n=== 处理图片示例 ===")
    # 使用点提示
    image_path = "/media/rbh/2TB/pycharmprojects_rbh/spagent/assets/example.png"  # 替换为实际的测试图片路径
    prompts = {
        'point_coords': [[900, 540]],  # 点击坐标
        'point_labels': [1]  # 1表示前景点
    }
    result = client.infer(image_path, prompts)
    
    if result:
        logger.info("图片处理成功！")
        logger.info(f"- 输入图片: {image_path}")
        logger.info(f"- 掩码文件: {result['mask_path']}")
        logger.info(f"- 可视化文件: {result['vis_path']}")
        logger.info(f"- 掩码尺寸: {result['shape']}")
    else:
        logger.error("图片处理失败")
    
    # 4. 处理视频示例
    # logger.info("\n=== 处理视频示例 ===")
    # video_path = "assets/test.mp4"  # 替换为实际的测试视频路径
    # prompts = {
    #     'point_coords': [[100, 100]],  # 第一帧的点击坐标
    #     'point_labels': [1]  # 1表示前景点
    # }
    # result = client.infer_video(video_path, prompts)
    
    # if result:
    #     logger.info("视频处理成功！")
    #     logger.info(f"- 输入视频: {video_path}")
    #     logger.info(f"- 输出视频: {result['output_path']}")
    #     logger.info(f"- 总帧数: {result['frames']}")
    #     logger.info(f"- FPS: {result['fps']}")
    #     logger.info(f"- 分辨率: {result['size']}")
    # else:
    #     logger.error("视频处理失败")

if __name__ == '__main__':
    main() 