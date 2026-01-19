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

    def infer_video(self, video_path, task, model_name, names, output_path=None):
        """
        发送完整视频文件到服务器进行标注
        
        Args:
            video_path: 视频文件路径
            task: 标注任务（'image' 或 'video'）
            model_name: 使用的模型名称
            names: 类别名称列表
            output_path: 输出视频路径，如果为None则自动生成
            
        Returns:
            处理结果，如果失败则返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在: {video_path}")
                return None
            
            # 生成输出路径
            if output_path is None:
                input_filename = os.path.splitext(os.path.basename(video_path))[0]
                output_path = f"outputs/annotated_{task}_{input_filename}.mp4"
            
            logger.info(f"开始上传视频文件: {video_path}")
            
            # 读取整个视频文件
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            
            # 将视频文件编码为base64
            video_b64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # 准备请求数据
            data = {
                'video': video_b64,  # 改为video字段
                'task': task,
                'model_name': model_name,
                'class_names': names,
                'output_filename': os.path.basename(output_path)
            }
            
            logger.info(f"发送视频处理请求，文件大小: {len(video_bytes)/1024/1024:.1f}MB")
            
            # 发送请求到视频处理接口
            response = requests.post(
                f'{self.server_url}/infer_video',  # 新的视频处理接口
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=300  # 视频处理需要更长时间
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            if result.get('success'):
                # 下载处理后的视频
                video_bytes = base64.b64decode(result['annotated_video'])
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 保存结果视频
                with open(output_path, 'wb') as f:
                    f.write(video_bytes)
                
                logger.info(f"视频处理完成: {output_path}")
                return {
                    'success': True,
                    'output_path': output_path,
                    'total_frames': result.get('total_frames', 0)
                }
            else:
                logger.error(f"服务器返回错误: {result.get('error')}")
                return None
            
        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            return None

    def infer(self, image_path, task, model_name, names):
        """
        发送图片进行标注
        
        Args:
            image_path: 图片路径
            task: 标注任务（'image' 或 'video'）
            model_name: 使用的模型名称
            names: 类别名称列表
            
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
                'model_name': model_name,
                'class_names': names
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
    SERVER_URL = "http://0.0.0.0:8000"  # 修改为你的服务地址

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

    # 图片处理示例
    logger.info("\n=== 图片推理 ===")
    image_path = "image.jpg"
    task = "image"  # 图片任务
    model_name = "yoloe-v8l-seg.pt"
    names = ["cat", "car"]

    result = client.infer(image_path, task, model_name, names)
    if result:
        logger.info(f"图片处理完成，输出图像: {result['output_path']}")
    else:
        logger.error("图片推理失败")

    # 视频处理示例
    logger.info("\n=== 视频推理 ===")
    video_path = "assets/suitcases-1280x720.mp4"
    video_task = "video"  # 视频任务
    video_names = ["suitcase"]

    if os.path.exists(video_path):
        video_result = client.infer_video(video_path, video_task, model_name, video_names)
        if video_result:
            logger.info(f"视频处理完成，输出视频: {video_result['output_path']}")
            logger.info(f"处理了 {video_result['total_frames']} 帧")
        else:
            logger.error("视频推理失败")
    else:
        logger.warning(f"视频文件不存在，跳过视频测试: {video_path}")

    logger.info("\n=== 处理完成 ===")


if __name__ == '__main__':
    main()
