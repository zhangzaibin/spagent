import json
from typing import List

MIXED_EXPERT_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function",
 "function":{
 "name":"depth_estimation_tool",
 "description":"Generate a depth map for the input image to analyze the 3D spatial relationships and depth distribution of objects in the scene.",
 "parameters":{
     "type":"object",
     "properties":{"image_path":{"type":"string","description":"The path to the input image for depth estimation."}
   },
 "required":["image_path"]}
}},
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
}},
{"type":"function",
 "function":{
 "name":"detect_objects_tool",
 "description":"Detect and locate objects in the image using GroundingDINO for open-vocabulary object detection.",
 "parameters":{
     "type":"object",
     "properties":{
         "image_path":{"type":"string","description":"The path to the input image for object detection."},
         "text_prompt":{"type":"string","description":"Text description of objects to detect."},
         "box_threshold":{"type":"number","description":"Confidence threshold for box detection."},
         "text_threshold":{"type":"number","description":"Confidence threshold for text matching."}
     },
     "required":["image_path", "text_prompt"]
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
{"name": "depth_estimation_tool", "arguments": {"image_path": "input.jpg"}}  
</tool_call>"""

def get_user_prompt(question: str) -> str:
    """
    生成用户prompt
    
    Args:
        question: 用户问题
        
    Returns:
        格式化的用户prompt
    """
    return f"\nThink first, analyze what tools might be needed (depth estimation, segmentation, object detection), then call appropriate tools if needed. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>\n\nQuestion: {question}"

def get_follow_up_prompt(question: str, initial_response: str, used_tools: List[str]) -> str:
    """
    生成后续分析的prompt
    
    Args:
        question: 用户问题
        initial_response: VLLM的初始回答
        used_tools: 使用的工具列表
        
    Returns:
        后续分析的prompt字符串
    """
    tool_descriptions = {
        "depth_estimation_tool": "depth map",
        "segment_image_tool": "segmentation mask",
        "detect_objects_tool": "object detection results"
    }
    
    tool_outputs = ", ".join([tool_descriptions[tool] for tool in used_tools])
    
    return f"""Initial question: {question}
Initial response: {initial_response}

Now you have access to additional information from the tools: {tool_outputs}

Please analyze all available information (original image and {tool_outputs}) to provide a comprehensive answer about the scene.

Format your response as: <analysis>...</analysis> <answer>...</answer>"""

def get_complete_prompt(question: str) -> str:
    """
    获取完整的prompt（系统指令 + 用户指令）
    
    Args:
        question: 用户问题
        
    Returns:
        完整的prompt字符串
    """
    return MIXED_EXPERT_SYSTEM_PROMPT + get_user_prompt(question) 