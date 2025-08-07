import json
import logging
import os
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
from external_experts.SAM2.sam2_client import SAM2Client
import ast
from external_experts.GroundingDINO.grounding_dino_client import GroundingDINOClient
# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.utils import draw_boxes_on_image, parse_json, extract_objects_from_response
from vllm_models.gpt import gpt_single_image_inference, gpt_multiple_images_inference, gpt_text_only_inference
from vllm_models.qwen import qwen_single_image_inference, qwen_multiple_images_inference, qwen_text_only_inference

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QWEN2_5_VL_PROMPT = """You are given an image and a text description.  
Your task is to find the object described in the text and output its bounding boxes in image coordinates.  
Each box should be given as [x1, y1, x2, y2], where (x1, y1) represents the top-left corner of the box, and (x2, y2) represents the bottom-right corner.  
Outline the position of each object described in the text andd output all the coordinates in JSON format.
\n\nText: {text}"""
# Output only strictly in the following format:  
# <box1>[x1, y1, x2, y2]</box1><box2>[x1, y1, x2, y2]</box2>...<boxn>[x1, y1, x2, y2]</boxn>
SAM2_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function",
 "function":{
 "name":"segment_image_tool",
 "description":"Segment objects in the image based on user's request. Can use points, boxes to guide segmentation.",
 "parameters":{
     "type":"object",
     "properties":{
         "image_path":{"type":"string","description":"The path to the input image for segmentation."},
         "point_coords":{"type":"array","items":{"type":"array","items":{"type":"number"}},"description":"Optional list of point coordinates [[x1,y1], [x2,y2], ...]"},
         "point_labels":{"type":"array","items":{"type":"number"},"description":"Optional list of point labels (1 for foreground, 0 for background)"},
         "box":{"type":"array","items":{"type":"number"},"description":"Optional bounding box coordinates [x1,y1,x2,y2]"},
     },
     "required":["image_path"]
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
{"name": "segment_image_tool", "arguments": {"image_path": "input.jpg", "point_coords": [[100, 100]], "point_labels": [1]}}  
</tool_call>"""

def get_user_prompt(question: str) -> str:
    """
    生成用户prompt
    
    Args:
        question: 用户问题
        
    Returns:
        格式化的用户prompt
    """
    return f"\nThink first, and second give the confidence score of your answer, if the confidence score is lower than 0.5, you should call the tools, Remember to strictly follow the strategy of using a score threshold to decide whether to call a tool. And last, you should answer What objects are involved in this question? Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <object_1>...</object_1> <object_2>...</object_2>...<object_n>...</object_n> <answer>...</answer>\n\nQuestion: {question}"



def get_follow_up_prompt(question: str, initial_response: str) -> str:
    """
    生成后续分析prompt（当需要分割图时）
    
    Args:
        question: 原始问题
        initial_response: VLLM的初始回答
        
    Returns:
        格式化的后续分析prompt
    """
    return f"""Based on the original image and the segmentation map, please provide a comprehensive answer to the question: {question}

Your previous response was: {initial_response}

Remember you need to pay special attention to the objects that are colored in the segmentation map, because these are the objects mentioned in the question. Please analyze both the original image and the segmentation map to provide a detailed answer about the spatial relationships, segmentation distribution, and 3D structure of objects in the scene.

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
    return SAM2_SYSTEM_PROMPT + get_user_prompt(question)

def get_qwen2_5_prompt(text):
    return QWEN2_5_VL_PROMPT.format(text=text)


class SimpleMockClient:
    """简单的Mock客户端，用于测试"""
    
    def infer(self, image_path: str, prompts: Optional[Dict] = None):
        """模拟图像分割"""
        import cv2
        import numpy as np
        
        # 创建一个简单的分割掩码
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # 生成模拟掩码（简单的圆形区域）
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        radius = min(image.shape[0], image.shape[1]) // 4
        cv2.circle(mask, center, radius, (255,), -1)
        
        # 创建可视化结果
        vis_image = image.copy()
        # 创建与掩码区域相同大小的红色overlay
        overlay = np.zeros_like(vis_image)
        overlay[mask > 0] = [0, 0, 255]  # BGR格式，红色
        # 在掩码区域进行图像混合
        vis_image = cv2.addWeighted(vis_image, 0.5, overlay, 0.5, 0)
        
        # 保存结果
        mask_path = f"assets/mock_mask_{os.path.basename(image_path)}"
        vis_path = f"assets/mock_vis_{os.path.basename(image_path)}"
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(vis_path, vis_image)
        
        return {
            'success': True,
            'mask_path': mask_path,
            'vis_path': vis_path,
            'shape': mask.shape
        }

class SAM2QAWorkflow:
    """SAM2图像分割问答工作流"""
    
    def __init__(self, use_mock_sam: bool = True, use_dino: bool = False):
        """
        初始化工作流
        
        Args:
            use_mock_sam: 是否使用mock SAM2服务
        """
        self.use_mock_sam = use_mock_sam
        self.use_dino = use_dino
        
        # 初始化SAM2客户端
        if use_mock_sam:
            # 使用简单的mock客户端
            self.sam_client = SimpleMockClient()
            logger.info("使用Mock SAM2服务")
        else:
            # 使用真实的SAM2客户端
            try:
                self.sam_client = SAM2Client("http://0.0.0.0:5000")
                if self.use_dino:
                    self.groundingdino_client = GroundingDINOClient("http://0.0.0.0:5001")
                logger.info("使用真实SAM2服务")
            except ImportError:
                logger.error("无法导入真实SAM2客户端")
                raise

    def call_sam_segmentation(self, image_path: str, prompts: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        调用SAM2分割专家
        
        Args:
            image_path: 输入图片路径
            prompts: 提示信息（点击坐标、框选区域等）
            
        Returns:
            分割结果
        """
        logger.info(f"调用SAM2专家处理图片: {image_path}")
        
        try:
            # 执行图像分割
            result = self.sam_client.infer(image_path, prompts)
            
            if result and result.get('success'):
                logger.info("图像分割完成")
                return result
            else:
                logger.error("图像分割失败")
                return None
                
        except Exception as e:
            logger.error(f"SAM2调用异常: {e}")
            return None
    
    def call_groundingdino(self, image_path: str, prompts: Optional[Dict] = None) -> Optional[List[Dict]]:
        """
        调用groundingdino模型为sam2生成prompt
        
        Args:
            image_path: 输入图片路径
            prompts: 要grounding的目标
            
        Returns:
            grounding结果，格式为 [{'box': [x1, y1, x2, y2], 'label': 'label'}, ...]
        """
        logger.info(f"调用groundingdino专家处理图片: {image_path}")
        
        try:
            # 执行图像检测
            result = self.groundingdino_client.infer(image_path, prompts)
            visual_prompt_response = []
            
            if result and result.get('success'):
                # 获取检测结果
                detections = result.get('detections', [])
                image_shape = result.get('shape', [])
                image_height, image_width = image_shape[0], image_shape[1]
                
                if detections:
                    for detection in detections:
                        # 获取检测结果的边界框和标签
                        bbox = detection['bbox']  # [cx, cy, w, h]
                        label = detection.get('label', 'unknown')  # 获取标签
                        
                        # 将box从cxcywh转成xyxy格式
                        cx, cy, w, h = bbox
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        bbox_xyxy = [
                            x1 * image_width,   
                            y1 * image_height,  
                            x2 * image_width,   
                            y2 * image_height  
                        ]
                        
                        # 添加到结果列表，格式与extract_coordinates_from_response一致
                        visual_prompt_response.append({
                            'box': bbox_xyxy,
                            'label': label
                        })
                        
                else:
                    logger.warning("未检测到目标")
                    return None
                
                logger.info("图像grounding完成")
                return visual_prompt_response
            else:
                logger.error("图像grounding失败")
                return None
                
        except Exception as e:
            logger.error(f"groundingdino调用异常: {e}")
            return None

    def needs_segmentation_tool(self, response: str) -> bool:
        """
        检查VLLM回答是否需要分割工具
        
        Args:
            response: VLLM的回答
            
        Returns:
            是否需要分割工具
        """

        if "<tool_call>" in response.lower():
            return True
        else:
            return False
        # # 简单的关键词检测
        # segmentation_keywords = [
        #     "segment", "mask", "region", "object", "foreground", "background",
        #     "area", "part", "portion", "section", "separate", "isolate",
        #     "extract", "select", "highlight"
        # ]
        
        # response_lower = response.lower()
        # return any(keyword in response_lower for keyword in segmentation_keywords)

    def extract_coordinates_from_response(self, response: str) -> Optional[List[Dict]]:
        """
        Extract coordinates from JSON format in the response text
        Based on qwen's official implementation with ast.literal_eval
        
        Args:
            response: Response text containing box coordinates in JSON format
            
        Returns:
            List of dictionaries containing box coordinates and labels, or None if extraction fails
        """
        try:
            # Parse JSON using qwen's method
            clean_json = parse_json(response)
            
            try:
                # First try with ast.literal_eval (qwen's preferred method)
                json_output = ast.literal_eval(clean_json)
            except Exception as e:
                logger.warning(f"ast.literal_eval failed: {e}")
                try:
                    # Fallback: try to fix incomplete JSON as qwen does
                    end_idx = clean_json.rfind('"}') + len('"}')
                    truncated_text = clean_json[:end_idx] + "]"
                    json_output = ast.literal_eval(truncated_text)
                except Exception as e2:
                    logger.warning(f"Truncated JSON also failed: {e2}")
                    # Final fallback: try standard json.loads
                    json_output = json.loads(clean_json)
            
            # Extract boxes and labels
            prompts = []
            for item in json_output:
                if 'bbox_2d' in item and len(item['bbox_2d']) == 4:
                    box = [float(coord) for coord in item['bbox_2d']]
                    label = item.get('label', 'unknown')
                    prompts.append({
                        'box': box,
                        'label': label
                    })
            
            if prompts:
                logger.info(f"Successfully extracted {len(prompts)} bounding boxes")
                logger.info(f"Extracted prompts: {prompts}")
                return prompts
            else:
                logger.error("No valid boxes found in response")
                return None

        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
            logger.error(f"Problematic response: {response}")
            return None

    def run_workflow(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        运行完整的SAM2 QA工作流
        
        Args:
            image_path: 输入图像路径
            question: 用户问题
            
        Returns:
            完整的工作流结果
        """
        logger.info("开始执行SAM2 QA工作流")

        complete_prompt = get_complete_prompt(question)
        
        # 1. VLLM先回答
        initial_response = gpt_single_image_inference(
            image_path=image_path,
            prompt=complete_prompt,
            model="gpt-4o-mini",
            temperature=0.7
        )

        if not self.use_dino:# 2.调用qwen2.5-vl-32B去给出点或框的visual prompt，交给sam2去分割
            objects = extract_objects_from_response(initial_response)
            # 将物体列表用and连接，处理单个物体和多个物体的情况
            if len(objects) == 0:
                visual_text = "no specific object"
            elif len(objects) == 1:
                visual_text = objects[0]
            else:
                # 最后两个物体用and连接，之前的用逗号分隔
                visual_text = ", ".join(objects[:-1]) + " and " + objects[-1]
            prompt_for_visual = get_qwen2_5_prompt(visual_text)
            visual_prompt_response = qwen_single_image_inference(
                image_path=image_path,
                prompt=prompt_for_visual,
                model="qwen2.5-vl-32b-instruct",
                temperature=0.7
            )
        else:# 2.调用grounding dino去给出点或框的visual prompt，交给sam2去分割
            objects = extract_objects_from_response(initial_response)
            # 将物体列表用and连接，处理单个物体和多个物体的情况
            if len(objects) == 0:
                visual_text = "no specific object"
            elif len(objects) == 1:
                visual_text = objects[0]
            else:
                # 最后两个物体用and连接，之前的用逗号分隔
                visual_text = ".".join(objects[:])

        # 3. 检查是否需要分割工具
        if self.needs_segmentation_tool(initial_response):
            logger.info("VLLM需要分割工具，调用SAM2")
            
            if not self.use_dino:
                # 用qwen2.5-vl-32B生成visual prompt
                prompts = self.extract_coordinates_from_response(visual_prompt_response)
            else:
                # 用groundingdino生成visual prompt
                prompts = self.call_groundingdino(image_path, visual_text)
            
            # 如果成功提取到边界框，先绘制到图像上用于可视化
            if prompts and len(prompts) > 0:
                # 为draw_boxes_on_image函数准备参数，转换为原有格式
                boxes_for_draw = [item['box'] for item in prompts]
                labels_for_draw = [item['label'] for item in prompts]
                draw_prompts = {'box': boxes_for_draw, 'labels': labels_for_draw}
                
                box_vis_path = draw_boxes_on_image(image_path, draw_prompts)
                logger.info(f"边界框已绘制到图像: {box_vis_path}")
            
            # 执行分割
            # 为SAM2准备传统格式的prompts
            sam_prompts = {'box': [item['box'] for item in prompts]} if prompts else None
            sam_result = self.call_sam_segmentation(image_path, sam_prompts)
            
            if sam_result and sam_result.get('vis_path'):
                follow_up_prompt = get_follow_up_prompt(question, initial_response)
                # 3. 重新给VLLM回答，同时传入原图和分割结果
                final_response = gpt_multiple_images_inference(
                    image_paths=[image_path, sam_result['vis_path']],
                    prompt=follow_up_prompt,
                    model="gpt-4o-mini",
                    temperature=0.7
                )
                print(f"follow_up_response: {final_response}")
            else:
                logger.warning("图像分割失败，使用原始回答")
                sam_result = None
                final_response = initial_response
        else:
            logger.info("VLLM不需要分割工具")
            sam_result = None
            final_response = initial_response
            print(f"final_response: {final_response}")
        
        # 4. 生成输出
        return {
            "answer": final_response,
            "segmentation_used": sam_result is not None,
            "segmentation_result": sam_result
        }

def infer(image_path: str, question: str, use_mock_sam: bool = True) -> Dict[str, Any]:
    """
    简单的推理接口
    
    Args:
        image_path: 输入图像路径
        question: 用户问题
        use_mock_sam: 是否使用mock SAM2服务
        
    Returns:
        推理结果
    """
    workflow = SAM2QAWorkflow(use_mock_sam=use_mock_sam, use_dino=False)
    return workflow.run_workflow(image_path, question)

def main():
    """主程序示例"""
    # 创建工作流实例
    workflow = SAM2QAWorkflow(use_mock_sam=False, use_dino=False)
    
    # 示例图像路径和问题
    image_path = "assets/example.png"  # 替换为实际的测试图片路径
    question = "Can you segment the main object in this image? If you can't segment it, please output your most interest object in the image and give out bounding box of the object. The bounding box format please following this [x1,y1,x2,y2]"
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        logger.warning(f"Example image does not exist: {image_path}")
        logger.info("Please provide a valid image path for testing")
        return
    
    # 运行工作流
    result = workflow.run_workflow(image_path, question)
    
    # 输出结果
    print("\n=== SAM2 QA Workflow Results ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 