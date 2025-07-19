"""
深度估计问答工作流的Prompt模板
基于论文中的工具调用格式设计
"""

# 系统指令prompt
DEPTH_ESTIMATION_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"depth_estimation_tool","description":"Generate a depth map for the input image to analyze the 3D spatial relationships and depth distribution of objects in the scene.","parameters":{"type":"object","properties":{"image_path":{"type":"string","description":"The path to the input image for depth estimation."}},"required":["image_path"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "depth_estimation_tool", "arguments": {"image_path": "input_image.jpg"}}  
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
    return f"\nThink first, call **depth_estimation_tool** if needed, then answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>\n\nQuestion: {question}"

# 后续分析prompt模板
def get_follow_up_prompt(question: str, initial_response: str) -> str:
    """
    生成后续分析prompt（当需要深度图时）
    
    Args:
        question: 原始问题
        initial_response: VLLM的初始回答
        
    Returns:
        格式化的后续分析prompt
    """
    return f"""Based on the original image and the depth map, please provide a comprehensive answer to the question: {question}

Your previous response was: {initial_response}

Please analyze both the original image and the depth map to provide a detailed answer about the spatial relationships, depth distribution, and 3D structure of objects in the scene.

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
    return DEPTH_ESTIMATION_SYSTEM_PROMPT + get_user_prompt(question)

# 中文版本的prompt模板
DEPTH_ESTIMATION_SYSTEM_PROMPT_ZH = """你是一个有用的助手。

# 工具
你可以调用一个或多个函数来协助用户查询。
在<tools></tools> XML标签内提供了函数签名：
<tools>
{"type":"function","function":{"name":"depth_estimation_tool","description":"为输入图像生成深度图，分析场景中物体的3D空间关系和深度分布。","parameters":{"type":"object","properties":{"image_path":{"type":"string","description":"用于深度估计的输入图像路径。"}},"required":["image_path"]}}}
</tools>

# 如何调用工具
在<tool_call></tool_call> XML标签内返回包含函数名称和参数的json对象：
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**示例**:  
<tool_call>  
{"name": "depth_estimation_tool", "arguments": {"image_path": "input_image.jpg"}}  
</tool_call>"""

def get_user_prompt_zh(question: str) -> str:
    """
    生成中文用户prompt
    
    Args:
        question: 用户问题
        
    Returns:
        格式化的中文用户prompt
    """
    return f"\n先思考，如果需要的话调用**depth_estimation_tool**，然后回答。严格按照格式：<think>...</think> <tool_call>...</tool_call>（如果需要工具） <answer>...</answer>\n\n问题：{question}"

def get_complete_prompt_zh(question: str) -> str:
    """
    获取完整的中文prompt
    
    Args:
        question: 用户问题
        
    Returns:
        完整的中文prompt字符串
    """
    return DEPTH_ESTIMATION_SYSTEM_PROMPT_ZH + get_user_prompt_zh(question) 