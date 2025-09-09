import base64
import requests
import cv2
import numpy as np
from PIL import Image
import io
import logging
import os
import time

logger = logging.getLogger(__name__)

class MoondreamClient:
    def __init__(self, server_url="http://localhost:20024"):
        """
        初始化Moondream客户端
        
        Args:
            server_url: Moondream服务器地址
        """
        self.server_url = server_url
        
    def _encode_image(self, image):
        """将图像编码为base64字符串"""
        if isinstance(image, str):
            # 如果是文件路径
            with open(image, 'rb') as f:
                image_bytes = f.read()
        elif isinstance(image, np.ndarray):
            # 如果是numpy数组
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()
        elif isinstance(image, Image.Image):
            # 如果是PIL图像
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
        else:
            raise ValueError("不支持的图像格式")
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _save_annotated_image(self, annotated_b64, input_path, task_type):
        """保存标注后的图像到outputs文件夹"""
        try:
            if not annotated_b64:
                return None
                
            # 创建outputs文件夹
            os.makedirs("outputs", exist_ok=True)
            
            # 解码base64图像
            annotated_bytes = base64.b64decode(annotated_b64)
            annotated_array = cv2.imdecode(
                np.frombuffer(annotated_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            # 生成输出文件名
            if isinstance(input_path, str):
                input_filename = os.path.basename(input_path)
                name, ext = os.path.splitext(input_filename)
            else:
                name, ext = "image", ".jpg"
            
            output_filename = f"outputs/{task_type}_{name}{ext}"
            
            # 保存图像
            cv2.imwrite(output_filename, annotated_array)
            logger.info(f"标注图像已保存到: {output_filename}")
            
            return output_filename
            
        except Exception as e:
            logger.error(f"保存标注图像失败：{e}")
            return None

    def health_check(self):
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.server_url}/health")
            return response.json()
        except Exception as e:
            logger.error(f"健康检查失败：{e}")
            return {"status": "不健康", "error": str(e)}
    
    def test(self):
        """测试服务器"""
        try:
            response = requests.get(f"{self.server_url}/test")
            return response.json()
        except Exception as e:
            logger.error(f"测试失败：{e}")
            return {"success": False, "error": str(e)}
    
    def caption(self, image):
        """
        生成图像描述
        
        Args:
            image: 图像（文件路径、numpy数组或PIL图像）
            
        Returns:
            dict: 包含描述文本的结果
        """
        try:
            image_b64 = self._encode_image(image)
            
            data = {
                "image": image_b64,
                "task": "caption"
            }
            
            response = requests.post(f"{self.server_url}/infer", json=data)
            return response.json()
            
        except Exception as e:
            logger.error(f"图像描述失败：{e}")
            return {"success": False, "error": str(e)}
    
    def query(self, image, question):
        """
        对图像进行问答
        
        Args:
            image: 图像（文件路径、numpy数组或PIL图像）
            question: 问题文本
            
        Returns:
            dict: 包含答案的结果
        """
        try:
            image_b64 = self._encode_image(image)
            
            data = {
                "image": image_b64,
                "task": "query",
                "question": question
            }
            
            response = requests.post(f"{self.server_url}/infer", json=data)
            return response.json()
            
        except Exception as e:
            logger.error(f"图像问答失败：{e}")
            return {"success": False, "error": str(e)}
    
    def detect(self, image, object_name):
        """
        检测图像中的对象
        
        Args:
            image: 图像（文件路径、numpy数组或PIL图像）
            object_name: 要检测的对象名称
            
        Returns:
            dict: 包含检测结果的字典
        """
        try:
            image_b64 = self._encode_image(image)
            
            data = {
                "image": image_b64,
                "task": "detect",
                "object": object_name
            }
            
            response = requests.post(f"{self.server_url}/infer", json=data)
            result = response.json()
            
            # 如果成功并且有标注图像，保存到本地
            if result.get('success') and result.get('annotated_image'):
                output_path = self._save_annotated_image(
                    result['annotated_image'], 
                    image, 
                    "detected"
                )
                # 将输出路径添加到结果中
                result['output_path'] = output_path
                
            return result
            
        except Exception as e:
            logger.error(f"对象检测失败：{e}")
            return {"success": False, "error": str(e)}
    
    def point(self, image, object_name):
        """
        定位图像中的对象点
        
        Args:
            image: 图像（文件路径、numpy数组或PIL图像）
            object_name: 要定位的对象名称
            
        Returns:
            dict: 包含点坐标的结果
        """
        try:
            image_b64 = self._encode_image(image)
            
            data = {
                "image": image_b64,
                "task": "point",
                "object": object_name
            }
            
            response = requests.post(f"{self.server_url}/infer", json=data)
            result = response.json()
            
            # 如果成功并且有标注图像，保存到本地
            if result.get('success') and result.get('annotated_image'):
                output_path = self._save_annotated_image(
                    result['annotated_image'], 
                    image, 
                    "pointing"
                )
                # 将输出路径添加到结果中
                result['output_path'] = output_path
                
            return result
            
        except Exception as e:
            logger.error(f"对象定位失败：{e}")
            return {"success": False, "error": str(e)}


def main():
    """测试Moondream客户端的所有功能"""
    
    # 服务器地址
    SERVER_URL = "http://localhost:20024"
    
    # 创建客户端
    client = MoondreamClient(SERVER_URL)
    
    # 1. 健康检查
    print("\n1️⃣ === 健康检查 ===")
    health = client.health_check()
    print(f"健康状态: {health}")
    
    if not health.get('model_loaded', False):
        print("❌ 模型未加载，请先启动服务器并设置API密钥")
        return False
    
    print("✅ 服务器健康状态良好")
    
    # 2. 测试接口
    print("\n2️⃣ === 测试接口 ===")
    test_result = client.test()
    print(f"测试结果: {test_result}")
    
    if not test_result.get('success', False):
        print("❌ 测试接口失败")
        return False
    
    print("✅ 测试接口成功")
    
    # 准备测试图像
    test_images = ["assets/example.png"]
    
    # 找到存在的测试图像
    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("❌ 未找到测试图像，请确保assets文件夹中有图像文件")
        return False
    
    print(f"📷 使用测试图像: {image_path}")
    
    # 等待服务器准备
    time.sleep(1)
    
    # 3. 图像描述测试
    print("\n3️⃣ === 图像描述测试 ===")
    try:
        caption_result = client.caption(image_path)
        if caption_result.get('success'):
            print(f"✅ 图像描述: {caption_result['caption']}")
        else:
            print(f"❌ 图像描述失败: {caption_result.get('error')}")
    except Exception as e:
        print(f"❌ 图像描述异常: {e}")
    
    # 等待服务器准备
    time.sleep(2)
    
    # 4. 视觉问答测试
    print("\n4️⃣ === 视觉问答测试 ===")
    questions = [
        "What objects can you see in this image?"
    ]
    
    for i, question in enumerate(questions, 1):
        try:
            print(f"\n问题 {i}: {question}")
            query_result = client.query(image_path, question)
            
            if query_result.get('success'):
                print(f"✅ 答案: {query_result['answer']}")
            else:
                print(f"❌ 问答失败: {query_result.get('error')}")
                
        except Exception as e:
            print(f"❌ 问答异常: {e}")
        
        # 避免请求过于频繁
        time.sleep(1)
    
    # 5. 对象检测测试
    print("\n5️⃣ === 对象检测测试 ===")
    objects_to_detect = ["person", "car"]
    
    for i, obj in enumerate(objects_to_detect, 1):
        try:
            print(f"\n检测 {i}: {obj}")
            detect_result = client.detect(image_path, obj)
            
            if detect_result.get('success'):
                detections = detect_result['detections']
                print(f"✅ 检测到 {len(detections)} 个 '{obj}'")
                
                if detect_result.get('output_path'):
                    print(f"   📁 标注图像已保存: {detect_result['output_path']}")
            else:
                print(f"❌ 检测失败: {detect_result.get('error')}")
                
        except Exception as e:
            print(f"❌ 检测异常: {e}")
        
        # 避免请求过于频繁
        time.sleep(1.5)
    
    # 6. 对象定位测试
    print("\n6️⃣ === 对象定位测试 ===")
    objects_to_point = ["person", "car"]
    
    for i, obj in enumerate(objects_to_point, 1):
        try:
            print(f"\n定位 {i}: {obj}")
            point_result = client.point(image_path, obj)
            
            if point_result.get('success'):
                points = point_result['points']
                print(f"✅ 定位到 {len(points)} 个 '{obj}' 的关键点")
                if point_result.get('output_path'):
                    print(f"标注图像已保存: {point_result['output_path']}")
            else:
                print(f"定位失败: {point_result.get('error')}")
                
        except Exception as e:
            print(f"定位异常: {e}")
        
        # 避免请求过于频繁
        time.sleep(1.5)
    return True


if __name__ == "__main__":   
    try:
        success = main()
        if success:
            print("\n✅ 所有测试完成！")
        else:
            print("\n❌ 测试过程中遇到问题，请检查服务器状态")
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
