import json
import logging
import os
import sys
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from vllm_models.gpt import gpt_single_image_inference, gpt_multiple_images_inference, gpt_text_only_inference

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPERVISION_SYSTEM_PROMPT = """You are a helpful assistant that can analyze images using object detection and segmentation tools.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function",
 "function":{
 "name":"supervision_tool",
 "description":"Perform object detection or segmentation on the input image to identify and analyze objects in the scene. Use 'image_det' for object detection with bounding boxes, or 'image_seg' for instance segmentation with masks.",
 "parameters":{
     "type":"object",
     "properties":{
         "image_path":{"type":"string","description":"The path to the input image for processing."},
         "task":{"type":"string","description":"The task type: 'image_det' for object detection or 'image_seg' for segmentation.","enum":["image_det","image_seg"]}
     },
 "required":["image_path","task"]}
}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "supervision_tool", "arguments": {"image_path": "input_image.jpg", "task": "image_det"}}  
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
    return f"\nThink first, call **supervision_tool** if needed for object detection or segmentation, then answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>\n\nQuestion: {question}"

# 后续分析prompt模板
def get_follow_up_prompt(question: str, initial_response: str) -> str:
    """
    生成后续分析prompt（当需要supervision处理时）
    
    Args:
        question: 原始问题
        initial_response: VLLM的初始回答
        
    Returns:
        格式化的后续分析prompt
    """
    return f"""Based on the original image and the processed image with object detection/segmentation annotations, please provide a comprehensive answer to the question: {question}

Your previous response was: {initial_response}

Please analyze both the original image and the annotated image to provide a detailed answer about the objects, their locations, relationships, and any other relevant details in the scene.

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
    return SUPERVISION_SYSTEM_PROMPT + get_user_prompt(question)



class SimpleMockSupervisionClient:
    """简单的Mock Supervision客户端，用于测试"""
    
    def infer(self, image_path: str, task: str = "image_det"):
        """
        模拟supervision处理
        
        Args:
            image_path: 输入图像路径
            task: 任务类型 ('image_det' 或 'image_seg')
        """
        import cv2
        import numpy as np
        
        logger.info(f"Mock Supervision处理: {image_path}, 任务: {task}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # 模拟不同任务的处理
        if task == "image_det":
            # 模拟目标检测：画几个边界框
            annotated = image.copy()
            h, w = image.shape[:2]
            
            # 画几个模拟的检测框
            cv2.rectangle(annotated, (w//4, h//4), (w//2, h//2), (0, 255, 0), 2)
            cv2.putText(annotated, "Object 1", (w//4, h//4-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.rectangle(annotated, (w//2, h//3), (3*w//4, 2*h//3), (255, 0, 0), 2)
            cv2.putText(annotated, "Object 2", (w//2, h//3-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        elif task == "image_seg":
            # 模拟实例分割：创建一些彩色mask区域
            annotated = image.copy()
            h, w = image.shape[:2]
            
            # 创建模拟的分割mask
            mask1 = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask1, (w//4, h//4), (w//2, h//2), 255, -1)
            annotated[mask1 > 0] = annotated[mask1 > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            mask2 = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask2, (w//2, h//3), (3*w//4, 2*h//3), 255, -1)
            annotated[mask2 > 0] = annotated[mask2 > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        
        # 保存处理后的图像
        output_path = f"outputs/mock_supervision_{task}_{os.path.basename(image_path)}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        
        return {
            'success': True,
            'output_path': output_path,
            'shape': annotated.shape,
            'task': task
        }


class SVQAWorkflow:
    """Supervision问答工作流"""
    
    def __init__(self, use_mock_sv: bool = True):
        """
        初始化工作流
        
        Args:
            use_mock_sv: 是否使用mock supervision服务
        """
        self.use_mock_sv = use_mock_sv
        
        # 初始化supervision客户端
        if use_mock_sv:
            # 使用mock服务
            self.supervision_client = SimpleMockSupervisionClient()
            logger.info("使用Mock Supervision服务")
        else:
            # 使用真实的supervision服务
            try:
                from external_experts.supervision.supervision_client import AnnotationClient
                self.supervision_client = AnnotationClient("http://localhost:8000")
                logger.info("使用真实Supervision服务")
            except ImportError:
                logger.error("无法导入真实supervision客户端")
                raise
        

    def call_supervision(self, image_path: str, task: str = "image_det") -> Optional[Dict[str, Any]]:
        """
        调用supervision专家
        
        Args:
            image_path: 输入图片路径
            task: 任务类型 ('image_det' 或 'image_seg')
            
        Returns:
            supervision处理结果
        """
        logger.info(f"调用Supervision专家处理图片: {image_path}, 任务: {task}")
        
        try:
            if self.use_mock_sv:
                # 使用mock客户端
                result = self.supervision_client.infer(image_path, task)
            else:
                # 使用真实客户端
                result = self.supervision_client.infer(image_path, task, "yolov8n-seg.pt")
            
            if result and result.get('success'):
                logger.info("Supervision处理完成")
                return result
            else:
                logger.error("Supervision处理失败")
                return None
                
        except Exception as e:
            logger.error(f"Supervision调用异常: {e}")
            return None

    def needs_supervision_tool(self, response: str) -> tuple[bool, str]:
        """
        检查VLLM回答是否需要supervision工具，并确定任务类型
        
        Args:
            response: VLLM的回答
            
        Returns:
            (是否需要supervision工具, 任务类型)
        """
        response_lower = response.lower()
        
        # 检查是否包含tool_call标签
        if "<tool_call>" in response_lower:
            # 尝试解析任务类型
            if "image_seg" in response_lower or "segmentation" in response_lower:
                return True, "image_seg"
            else:
                return True, "image_det"
            
        # 检查是否明确提到supervision相关关键词
        detection_keywords = [
            "supervision_tool", "object detection", "detect objects", "find objects", 
            "identify objects", "locate objects", "bounding box"
        ]
        
        segmentation_keywords = [
            "segmentation", "segment", "mask", "outline", "precise boundaries"
        ]
        
        # 检查分割关键词
        if any(keyword in response_lower for keyword in segmentation_keywords):
            return True, "image_seg"
            
        # 检查检测关键词
        if any(keyword in response_lower for keyword in detection_keywords):
            return True, "image_det"
            
        return False, "image_det"
    def run_workflow(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        运行完整的Supervision QA工作流
        
        Args:
            image_path: 输入图像路径
            question: 用户问题
            
        Returns:
            完整的工作流结果
        """
        logger.info("开始执行Supervision QA工作流")

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
        
        # 2. 检查是否需要supervision工具
        needs_tool, task_type = self.needs_supervision_tool(initial_response)
        
        if needs_tool:
            logger.info(f"VLLM需要supervision工具，任务类型: {task_type}")
            supervision_result = self.call_supervision(image_path, task_type)
            
            if supervision_result and supervision_result.get('output_path'):
                # 3. 重新给VLLM回答，同时传入原图和处理后的图
                follow_up_prompt = get_follow_up_prompt(question, initial_response)
                
                final_response = gpt_multiple_images_inference(
                    image_paths=[image_path, supervision_result['output_path']],
                    prompt=follow_up_prompt,
                    model="gpt-4o-mini",
                    temperature=0.7
                )
                logger.info(f"最终回答: {final_response}")
            else:
                logger.warning("Supervision处理失败，使用原始回答")
                supervision_result = None
                final_response = initial_response
        else:
            logger.info("VLLM不需要supervision工具，直接回答")
            supervision_result = None
            final_response = initial_response
        
        # 4. 生成输出
        return {
            "answer": final_response,
            "supervision_used": supervision_result is not None,
            "supervision_result": supervision_result,
            "task_type": task_type if supervision_result else None
        }

def infer(image_path: str, question: str, use_mock_sv: bool = True) -> Dict[str, Any]:
    """
    简单的推理接口
    
    Args:
        image_path: 输入图像路径
        question: 用户问题
        use_mock_sv: 是否使用mock supervision服务
        
    Returns:
        推理结果
    """
    workflow = SVQAWorkflow(use_mock_sv=use_mock_sv)
    return workflow.run_workflow(image_path, question)

def main():
    """主程序示例"""
    # 创建工作流实例
    workflow = SVQAWorkflow(use_mock_sv=False)
    
    # 示例图像路径和问题
    image_path = "assets/example.png"  # Please replace with actual image path
    
    # 测试两种不同的问题类型
    questions = [
        "What objects can you see in this image?",  # 应该触发object detection
        "Can you segment the main objects in this image and tell me about their precise boundaries?",  # 应该触发segmentation
        "What is the color of the sky in this image?",  # 不应该触发supervision工具
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
    
    print("\n=== Supervision QA Workflow Demo ===")
    
    # 运行不同类型的问题测试
    for i, question in enumerate(questions, 1):
        print(f"\n--- 测试 {i}: {question} ---")
        
        # 运行工作流
        result = workflow.run_workflow(image_path, question)
        
        # 输出结果
        print(f"问题: {question}")
        print(f"是否使用Supervision: {result.get('supervision_used', False)}")
        if result.get('supervision_used'):
            print(f"任务类型: {result.get('task_type', 'unknown')}")
            print(f"处理后图像: {result.get('supervision_result', {}).get('output_path', 'None')}")
        print(f"最终答案: {result['answer']}")
        print("-" * 80)

if __name__ == "__main__":
    main() 