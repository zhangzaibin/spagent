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


class AnnotationClient:
    def __init__(self, server_url):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址，如 'http://localhost:8000'
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

    def infer(self, image_path, task, model_name):
        """
        发送图片进行标注
        
        Args:
            image_path: 图片路径
            task: 标注任务（如 'image_det', 'image_seg'）
            model_name: 使用的模型名称（如 'yolov8n.pt'）
            
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
                'task': task,
                'model_name': model_name
            }

            logger.info(f"发送推理请求，任务：{task}，模型：{model_name}...")
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
                # 解码返回的图像（base64）
                img_bytes = base64.b64decode(result['annotated_image'])
                annotated_img = cv2.imdecode(
                    np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                )

                # 生成输出文件名（基于输入文件名）
                input_filename = os.path.basename(image_path)
                output_filename = f"outputs/annotated_{task}_{input_filename}"
                if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    output_filename += '.jpg'

                os.makedirs(os.path.dirname(output_filename), exist_ok=True)

                # 保存结果
                cv2.imwrite(output_filename, annotated_img)
                logger.info(f"标注图已保存至: {output_filename}")

                return {
                    'annotated_image': annotated_img,
                    'shape': annotated_img.shape,
                    'output_path': output_filename,
                    'success': True
                }
            else:
                logger.error(f"服务器返回错误: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"推理请求失败: {e}")
            return None


def main():
    """主程序"""
    SERVER_URL = "http://localhost:8000"  # 修改为你的服务地址

    client = AnnotationClient(SERVER_URL)

    logger.info("\n=== 执行健康检查 ===")
    health = client.health_check()
    if not health:
        logger.error("健康检查失败")
        return

    logger.info("\n=== 测试推理 ===")
    test = client.test_infer()
    if not test:
        logger.error("测试推理失败")
        return

    logger.info("\n=== 实际图片推理 ===")
    image_path = "/home/ubuntun/projects/spagent/street.jpg"
    task = "image_seg"  # 或 "image_det"
    model_name = "yolov8n-seg.pt"

    result = client.infer(image_path, task, model_name)
    if result:
        logger.info(f"处理完成，输出图像: {result['output_path']}")
    else:
        logger.error("图片推理失败")


if __name__ == '__main__':
    main()
