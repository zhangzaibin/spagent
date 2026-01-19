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
file_handler = logging.FileHandler('grounding_dino_client.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

class GroundingDINOClient:
    """
    Grounding DINO 目标检测客户端
    
    支持多种文本提示格式：
    - 单个对象: "person" -> "<object_1>person</object_1>"
    - 点分隔: "car.tree" -> "<object_1>car</object_1> <object_2>tree</object_2>"
    - 逗号分隔: "person, banana" -> "<object_1>person</object_1> <object_2>banana</object_2>"
    """
    def __init__(self, server_url):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址，如 'http://localhost:5001'
        """
        self.server_url = server_url.rstrip('/')
    
    def _format_text_prompt(self, text_prompt):
        """
        将文本提示转换为XML-like格式
        
        Args:
            text_prompt: 原始文本提示，支持逗号或点分隔，如 "person, banana" 或 "person.banana"
            
        Returns:
            格式化后的文本提示，如 "<object_1>person</object_1> <object_2>banana</object_2>"
        """
        if not text_prompt:
            return "<object_1>object</object_1>"
        
        # 支持逗号或点分隔
        if ',' in text_prompt:
            objects = [obj.strip() for obj in text_prompt.split(',') if obj.strip()]
        elif '.' in text_prompt:
            objects = [obj.strip() for obj in text_prompt.split('.') if obj.strip()]
        else:
            # 如果没有分隔符，当作单个对象处理
            objects = [text_prompt.strip()]
        
        # 转换为XML-like格式
        formatted_objects = []
        for i, obj in enumerate(objects, 1):
            formatted_objects.append(f"{obj}")
        
        return ".".join(formatted_objects)
    
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
    
    def test_text_formatting(self):
        """测试文本提示格式化功能"""
        test_cases = [
            "person",
            "car.tree", 
            "person, banana",
            "dog, cat, bird",
            "apple.orange.banana",
            ""
        ]
        
        logger.info("=== 测试文本提示格式化 ===")
        for test_case in test_cases:
            formatted = self._format_text_prompt(test_case)
            logger.info(f"输入: '{test_case}' -> 输出: '{formatted}'")
        
        return True
    
    def infer(self, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        发送图片进行目标检测
        
        Args:
            image_path: 图片路径
            text_prompt: 文本提示，如 "person", "car", "dog" 等，支持逗号或点分隔
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配阈值
            
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
            
            # 转换text_prompt格式：从 "person, banana" 或 "person.banana" 转换为 "<object_1>person</object_1> <object_2>banana</object_2>"
            formatted_prompt = self._format_text_prompt(text_prompt)
            
            # 准备请求数据
            data = {
                'image': image_b64,
                'text_prompt': formatted_prompt,
                'box_threshold': box_threshold,
                'text_threshold': text_threshold
            }
            
            logger.info(f"发送目标检测请求，文本提示：{formatted_prompt}")
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
                # 解码标注图像
                annotated_bytes = base64.b64decode(result['annotated_image'])
                annotated_array = cv2.imdecode(
                    np.frombuffer(annotated_bytes, np.uint8),
                    cv2.IMREAD_COLOR
                )
                
                # 生成输出文件名（基于输入文件名）
                input_filename = os.path.basename(image_path)
                name, ext = os.path.splitext(input_filename)
                output_filename = f"outputs/detected_{name}{ext}"
                
                # 保存结果
                cv2.imwrite(output_filename, annotated_array)
                logger.info(f"检测结果已保存至: {output_filename}")
                
                # 处理检测结果
                detections = result.get('detections', [])
                logger.info(f"检测到 {len(detections)} 个目标")
                
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    label = detection['label']
                    # logger.info(f"目标 {i+1}: {label} (置信度: {confidence:.3f}) 位置: {bbox}")
                
                return {
                    'detections': detections,
                    'annotated_image': annotated_array,
                    'output_path': output_filename,
                    'shape': result['shape'],
                    'success': True
                }
            else:
                logger.error(f"服务器返回错误: {result.get('message')}")
                return None
                
        except Exception as e:
            logger.error(f"推理请求失败: {e}")
            return None
    
    def infer_video(self, video_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        发送视频进行目标检测
        
        Args:
            video_path: 视频文件路径
            text_prompt: 文本提示，支持逗号或点分隔
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配阈值
            
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
            
            # 转换text_prompt格式
            formatted_prompt = self._format_text_prompt(text_prompt)
            
            # 准备请求数据
            data = {
                'video': video_b64,
                'text_prompt': formatted_prompt,
                'box_threshold': box_threshold,
                'text_threshold': text_threshold
            }
            
            logger.info(f"发送视频处理请求，文本提示：{formatted_prompt}")
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
                name, ext = os.path.splitext(input_filename)
                output_filename = f"outputs/detected_{name}{ext}"
                
                # 保存视频
                with open(output_filename, 'wb') as f:
                    f.write(video_bytes)
                logger.info(f"处理后的视频已保存至: {output_filename}")
                
                return {
                    'output_path': output_filename,
                    'frames': result['frames'],
                    'fps': result['fps'],
                    'size': result['size'],
                    'total_detections': result.get('total_detections', 0)
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
    SERVER_URL = "http://localhost:5001"
    
    # 创建客户端
    client = GroundingDINOClient(SERVER_URL)
    
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
    
    # 2.5. 测试文本格式化功能
    logger.info("\n=== 测试文本格式化功能 ===")
    client.test_text_formatting()
    
    # 等待一下确保服务器准备好
    time.sleep(2)
    
    # 3. 处理图片示例
    logger.info("\n=== 处理图片示例 ===")
    image_path = "assets/example.png"  # 替换为实际的测试图片路径
    # 支持多种输入格式，会自动转换为XML-like格式：
    # "car.tree" -> "<object_1>car</object_1> <object_2>tree</object_2>"
    # "person, banana" -> "<object_1>person</object_1> <object_2>banana</object_2>"
    text_prompt = "car.tree" 
    
    result = client.infer(image_path, text_prompt)
    
    if result:
        logger.info("图片处理成功！")
        logger.info(f"- 输入图片: {image_path}")
        logger.info(f"- 输出图片: {result['output_path']}")
        logger.info(f"- 检测到 {len(result['detections'])} 个目标")
        logger.info(f"- 图片尺寸: {result['shape']}")
    else:
        logger.error("图片处理失败")
    
    # 4. 处理视频示例
    logger.info("\n=== 处理视频示例 ===")
    video_path = "assets/test.mp4"  # 替换为实际的测试视频路径
    # 支持多种输入格式，会自动转换为XML-like格式：
    # "car" -> "<object_1>car</object_1>"
    # "person, dog" -> "<object_1>person</object_1> <object_2>dog</object_2>"
    text_prompt = "car"  # 检测汽车
    
    result = client.infer_video(video_path, text_prompt)
    
    if result:
        logger.info("视频处理成功！")
        logger.info(f"- 输入视频: {video_path}")
        logger.info(f"- 输出视频: {result['output_path']}")
        logger.info(f"- 总帧数: {result['frames']}")
        logger.info(f"- FPS: {result['fps']}")
        logger.info(f"- 分辨率: {result['size']}")
        logger.info(f"- 总检测数: {result['total_detections']}")
    else:
        logger.error("视频处理失败")

if __name__ == '__main__':
    main() 