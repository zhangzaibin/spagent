import json
import logging
import os
import sys
import base64
import cv2
import numpy as np
import re
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.utils import extract_objects_from_response
from vllm_models.gpt import gpt_single_image_inference, gpt_multiple_images_inference, gpt_text_only_inference

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GDINO_SYSTEM_PROMPT = """You are a helpful assistant that can analyze images using Grounding DINO object detection capabilities.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function",
 "function":{
 "name":"gdino_detection_tool",
 "description":"Perform advanced object detection using Grounding DINO model. This tool can detect objects with custom text prompts specified by the user. It supports both image and video processing with high accuracy object localization and bounding box detection using natural language descriptions. Note: This tool only performs detection (bounding boxes), not segmentation.",
 "parameters":{
     "type":"object",
     "properties":{
         "image_path":{"type":"string","description":"The path to the input image for Grounding DINO processing."},
         "task":{"type":"string","description":"The processing task type: 'image' for single image object detection, or 'video' for video frame processing.","enum":["image","video"]},
         "text_prompt":{"type":"string","description":"Natural language text prompt describing objects to detect (e.g., 'person walking', 'red car', 'small dog'). Grounding DINO can understand complex text descriptions."}
     },
     "required":["image_path","task","text_prompt"]
 }
}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "gdino_detection_tool", "arguments": {"image_path": "input_image.jpg", "task": "image", "text_prompt": "person.car.bicycle"}}  
</tool_call>"""


# 用户prompt模板
def get_user_prompt(question: str) -> str:
    """
    生成用户prompt
    
    Args:
        question: 用户问题
        
    Returns:
        格式化的用户prompt
    """
    return f"""
Analyze the image carefully and answer the user's question. If you need to identify, detect, locate, count, or analyze specific objects in the image, use the **gdino_detection_tool** to get precise object detection results with bounding boxes.

When calling the tool, provide a text prompt describing what objects you want to detect (e.g., for "How many cars are there?", use text_prompt: "car").

Note: The Grounding DINO tool provides object detection with bounding boxes only, using natural language text prompts.

Think step by step and format your response as: 
<think>...</think> 
<tool_call>...</tool_call> (if Grounding DINO detection is needed)
<analysis>...</analysis> (analyze the detection results if tool was used)
<object_1>...</object_1> <object_2>...</object_2>...<object_n>...</object_n> (list objects involved in this question)
<answer>...</answer>

Question: {question}"""

# 后续分析prompt模板
def get_follow_up_prompt(question: str, initial_response: str) -> str:
    """
    生成后续分析prompt（当需要Grounding dino处理时）
    
    Args:
        question: 原始问题
        initial_response: VLLM的初始回答
        
    Returns:
        格式化的后续分析prompt
    """
    return f"""Based on the original image and the Grounding DINO processed image with object detection annotations (bounding boxes), please provide a comprehensive answer to the question: {question}

Your previous response was: {initial_response}

Please analyze both the original image and the annotated image with Grounding DINO detection results to provide a detailed answer about the objects, their locations, counts, relationships, and any other relevant details in the scene. The Grounding DINO tool has provided bounding box annotations for the detected objects.

Format your response as: <analysis>...</analysis> <answer>...</answer>"""

# 完整的prompt组合
def get_complete_prompt(question: str) -> str:
    """
    获取完整的prompt（系统指令 + 用户指令）
    
    Args:
        question: 用户问题
        
    Returns:
        完整的prompt字符串
    """
    return GDINO_SYSTEM_PROMPT + get_user_prompt(question)


class SimpleMockGdinoClient:
    """简单的Mock Grounding DINO客户端，用于测试"""
    
    def infer(self, image_path: str, text_prompt: str = None, box_threshold=0.35, text_threshold=0.25):
        """
        模拟Grounding DINO图片检测
        
        Args:
            image_path: 输入图像路径
            text_prompt: 文本提示，用点分隔多个对象 (e.g., "person. car. dog")
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配阈值
        """
        import cv2
        import numpy as np
        
        logger.info(f"Mock Grounding DINO处理图片: {image_path}, 文本提示: {text_prompt}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Cannot read image"}
            
        # 模拟目标检测：画几个边界框
        annotated = image.copy()
        h, w = image.shape[:2]
        
        # 解析文本提示
        detections = []
        if text_prompt:
            # 按点号分割文本提示
            objects = [obj.strip() for obj in text_prompt.split('.') if obj.strip()]
            
            for i, obj_name in enumerate(objects[:3]):  # 最多显示3个对象
                # 计算边界框位置
                x1 = (w // 4) + i * (w // 6)
                y1 = (h // 4) + i * (h // 8)
                x2 = x1 + w // 5
                y2 = y1 + h // 4
                
                # 确保边界框在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 画边界框
                color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][i % 3]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签
                confidence = 0.85 + i * 0.05
                label = f"{obj_name} {confidence:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 记录检测结果
                detections.append({
                    "id": i,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "label": obj_name
                })
        else:
            # 如果没有指定文本提示，默认检测一些通用对象
            cv2.rectangle(annotated, (w//4, h//4), (w//2, h//2), (0, 255, 0), 2)
            cv2.putText(annotated, "Object 0.90", (w//4, h//4-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            detections.append({
                "id": 0,
                "bbox": [w//4, h//4, w//2, h//2],
                "confidence": 0.90,
                "label": "object"
            })
        
        # 保存处理后的图像
        output_path = f"outputs/mock_gdino_det_{os.path.basename(image_path)}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        
        return {
            'success': True,
            'output_path': output_path,
            'detections': detections,
            'shape': annotated.shape,
            'text_prompt': text_prompt or "object"
        }
    
    def infer_video(self, video_path: str, text_prompt: str = None, box_threshold=0.35, text_threshold=0.25):
        """
        模拟Grounding DINO视频检测
        
        Args:
            video_path: 输入视频路径
            text_prompt: 文本提示
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配阈值
        """
        logger.info(f"Mock Grounding DINO处理视频: {video_path}, 文本提示: {text_prompt}")
        
        # 简单起见，对于视频我们只返回第一帧的处理结果
        # 实际应用中可以提取关键帧进行处理
        return {
            'success': True,
            'message': 'Video processing simulated',
            'output_path': None,
            'text_prompt': text_prompt or "object"
        }


class GdinoQAWorkflow:
    """
    Grounding DINO Vision QA Workflow - 使用Grounding DINO进行视觉问答的工作流
    """
    
    def __init__(self, use_mock: bool = True):
        """
        初始化工作流
        
        Args:
            use_mock: 是否使用Mock客户端进行测试
            gdino_client_url: Grounding DINO服务器URL
        """
        self.use_mock = use_mock

        # 初始化Grounding DINO客户端
        if use_mock:
            # 使用mock服务
            self.gdino_client = SimpleMockGdinoClient()
            logger.info("使用Mock Grounding DINO服务")
        else:
            # 使用真实的Grounding DINO服务
            try:
                try:
                    from spagent.external_experts.GroundingDINO.grounding_dino_client import GroundingDINOClient
                except ImportError:
                    try:
                        from ..external_experts.GroundingDINO.grounding_dino_client import GroundingDINOClient
                    except ImportError:
                        from external_experts.GroundingDINO.grounding_dino_client import GroundingDINOClient

                self.gdino_client = GroundingDINOClient(server_url="http://0.0.0.0:5001")
                logger.info("使用真实Grounding DINO服务")
            except ImportError:
                logger.error("无法导入真实Grounding DINO客户端")
                raise

    def _need_gdino_detection(self, response: str) -> bool:
        """
        判断是否需要调用Grounding DINO检测工具
        
        Args:
            response: VLLM的初始响应
            
        Returns:
            bool: 是否需要调用Grounding DINO
        """
        # 查找工具调用标记
        return "<tool_call>" in response and "gdino_detection_tool" in response

    def _extract_tool_params(self, response: str) -> dict:
        """
        从VLLM响应中提取工具调用参数
        
        Args:
            response: VLLM的初始响应
            
        Returns:
            dict: 工具参数字典，包含task
        """
        import re
        
        # 提取<tool_call>标签内容
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        if not tool_call_match:
            return {"task": "image"}
            
        tool_content = tool_call_match.group(1).strip()
        
        params = {
            "task": "image"  # 默认值
        }
        
        # 提取task参数
        task_match = re.search(r'task["\']?\s*:\s*["\']?(\w+)["\']?', tool_content)
        if task_match:
            params["task"] = task_match.group(1)
                
        return params
        
    def _call_gdino_detection(self, image_path: str, task: str = "image", text_prompt: str = None) -> str:
        """
        调用Grounding DINO检测服务（仅检测，不分割）
        
        Args:
            image_path: 图片路径
            task: 任务类型 ("image" 或 "video")
            text_prompt: 文本提示描述要检测的对象
            
        Returns:
            str: 标注后的图片路径（带边界框）
        """
        try:
            # 使用已初始化的客户端
            if self.use_mock:
                # 使用Mock客户端
                if task == "video":
                    # 调用视频检测
                    result = self.gdino_client.infer_video(
                        video_path=image_path,
                        text_prompt=text_prompt
                    )
                else:
                    # 调用图片检测
                    result = self.gdino_client.infer(
                        image_path=image_path,
                        text_prompt=text_prompt
                    )
            else:
                # 使用真实的Grounding DINO客户端
                if task == "video":
                    # 调用视频检测
                    result = self.gdino_client.infer_video(
                        video_path=image_path,
                        text_prompt=text_prompt or "object"
                    )
                else:
                    # 调用图片检测
                    result = self.gdino_client.infer(
                        image_path=image_path,
                        text_prompt=text_prompt or "object"
                    )
                
            # 返回标注后的图片路径
            if result and result.get('success'):
                # Mock客户端和真实客户端都返回output_path
                output_path = result.get('output_path')
                if output_path:
                    return output_path
                    
            logger.warning(f"Grounding DINO detection failed: {result}")
            return image_path  # 返回原图片路径
                
        except Exception as e:
            logger.error(f"Error calling Grounding DINO detection: {e}")
            return image_path  # 返回原图片路径
            
    def run_workflow(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        运行完整的视觉问答工作流
        
        Args:
            image_path: 图片路径
            question: 用户问题
            
        Returns:
            完整的工作流结果
        """
        logger.info("开始执行Grounding DINO QA工作流")

        # 使用新的prompt模板
        complete_prompt = get_complete_prompt(question)
        
        # 1. VLLM先回答
        initial_response = gpt_single_image_inference(
            image_path=image_path,
            prompt=complete_prompt,
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        logger.info(f"初始回答: {initial_response}")
        
        # 2. 检查是否需要Grounding DINO工具
        needs_tool = self._need_gdino_detection(initial_response)
        
        if needs_tool:
            # 从初始回答中提取对象列表，参考sam2的做法
            objects = extract_objects_from_response(initial_response)
            
            # 生成text_prompt，参考sam2的做法
            if len(objects) == 0:
                text_prompt = "object"  # 默认检测通用对象
            else:
                # 用点号连接多个对象，符合Grounding DINO的格式
                text_prompt = ".".join(objects)
            
            # 提取工具参数（只提取task参数）
            tool_params = self._extract_tool_params(initial_response)
            task_type = tool_params.get("task", "image")
            
            logger.info(f"VLLM需要Grounding DINO工具，任务类型: {task_type}, 文本提示: {text_prompt}")
            
            # 调用Grounding DINO检测
            annotated_image_path = self._call_gdino_detection(
                image_path=image_path,
                task=task_type,
                text_prompt=text_prompt
            )
            
            if annotated_image_path and annotated_image_path != image_path:
                # 3. 重新给VLLM回答，同时传入原图和处理后的图
                follow_up_prompt = get_follow_up_prompt(question, initial_response)
                
                final_response = gpt_multiple_images_inference(
                    image_paths=[image_path, annotated_image_path],
                    prompt=follow_up_prompt,
                    model="gpt-4o-mini",
                    temperature=0.7
                )
                logger.info(f"最终回答: {final_response}")

                gdino_result = {
                    'success': True,
                    'output_path': annotated_image_path,
                    'task': task_type,
                    'text_prompt': text_prompt
                }
            else:
                logger.warning("Grounding DINO处理失败，使用原始回答")
                gdino_result = None
                final_response = initial_response
        else:
            logger.info("VLLM不需要Grounding DINO工具，直接回答")
            gdino_result = None
            final_response = initial_response
            task_type = None
        
        # 4. 生成输出
        return {
            "answer": final_response,
            "gdino_used": gdino_result is not None,
            "gdino_result": gdino_result,
            "task_type": task_type if gdino_result else None
        }

def main():
    """主程序示例"""
    # 创建工作流实例
    workflow = GdinoQAWorkflow(use_mock=False)
    
    # 示例图像路径
    image_path = "assets/example.png"
    video_path = "assets/suitcases-1280x720.mp4"
    
    # 测试问题
    questions = [
        "What objects can you see in this image?",  # 应该触发object detection
        "How many cars are in this image?",  # 应该触发car detection
        "What is the color of the sky in this image?",  # 不应该触发Grounding DINO工具
    ]

    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        logger.warning(f"Example image does not exist: {image_path}")
        logger.info("Using test image from dataset instead")
        # 尝试使用数据集中的图像
        test_image_path = "/home/ubuntu/projects/spagent/dataset/BLINK/0a45758ac7376725109cba53848f2387c6f9280686ac8957868ec7f8ce71ba21.jpg"
        if os.path.exists(test_image_path):
            image_path = test_image_path
        else:
            logger.error("No test image available")
            return

    print("\n=== Grounding DINO QA Workflow Demo ===")

    # 运行不同类型的问题测试
    for i, question in enumerate(questions, 1):
        print(f"\n--- 测试 {i}: {question} ---")
        
        # 运行工作流
        result = workflow.run_workflow(image_path, question)
        
        # 输出结果
        print(f"问题: {question}")
        print(f"是否使用Grounding DINO: {result.get('gdino_used', False)}")
        if result.get('gdino_used'):
            print(f"任务类型: {result.get('task_type', 'unknown')}")
            print(f"处理后图像: {result.get('gdino_result', {}).get('output_path', 'None')}")
        print(f"最终答案: {result['answer']}")
        print("-" * 80)

    # 视频处理示例（可选）
    # if os.path.exists(video_path):
    #     print("\n=== 视频处理示例 ===")
    #     video_result = workflow.run_workflow(video_path, "What objects can you see in this video?")
        
    #     print(f"视频处理结果: {video_result.get('answer', 'No answer')}")
    #     if video_result.get('gdino_used'):
    #         print(f"处理后视频路径: {video_result.get('gdino_result', {}).get('output_path', 'None')}")

if __name__ == "__main__":
    main() 