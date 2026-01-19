import os
import base64
from openai import OpenAI
from pathlib import Path
from typing import List, Optional, Union

# Initialize Qwen client with DashScope configuration
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_image(image_path: str) -> str:
    """将图像文件编码为base64字符串
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        base64编码的图像字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_message_with_image(text: str, image_path: Optional[str] = None) -> dict:
    """创建包含文本和图像的消息
    
    Args:
        text: 文本内容
        image_path: 图像文件路径（可选）
        
    Returns:
        包含文本和图像的消息字典
    """
    message = {
        "role": "user",
        "content": []
    }
    
    # 添加文本内容
    if text:
        message["content"].append({
            "type": "text",
            "text": text
        })
    
    # 添加图像内容
    if image_path:
        base64_image = encode_image(image_path)
        message["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    return message

def qwen_single_image_inference(
    image_path: str, 
    prompt: str, 
    model: str = "qwen3-vl-8b-thinking",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """Qwen单图像推理函数
    
    Args:
        image_path: 图像文件路径
        prompt: 提示文本
        model: 使用的模型名称，默认为qwen2.5-vl-7b-instruct
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        Qwen的回复文本
    """
    message = create_message_with_image(prompt, image_path)
    
    response = client.chat.completions.create(
        model=model,
        messages=[message],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

def qwen_multiple_images_inference(
    image_paths: List[str], 
    prompt: str, 
    model: str = "qwen3-vl-8b-thinking",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """Qwen多图像推理函数
    
    Args:
        image_paths: 图像文件路径列表
        prompt: 提示文本
        model: 使用的模型名称，默认为qwen2.5-vl-7b-instruct
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        Qwen的回复文本
    """
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
    
    # 添加所有图像
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        message["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    response = client.chat.completions.create(
        model=model,
        messages=[message],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

def qwen_text_only_inference(
    prompt: str, 
    model: str = "qwen2.5-7b-instruct",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """Qwen纯文本推理函数
    
    Args:
        prompt: 提示文本
        model: 使用的模型名称，默认为qwen2.5-7b-instruct（纯文本模型）
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        Qwen的回复文本
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

def qwen_raw_response(
    image_path: str, 
    prompt: str, 
    model: str = "qwen2.5-vl-7b-instruct",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> dict:
    """Qwen原始响应函数，返回完整的响应对象
    
    Args:
        image_path: 图像文件路径
        prompt: 提示文本
        model: 使用的模型名称
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        完整的响应对象
    """
    message = create_message_with_image(prompt, image_path)
    
    response = client.chat.completions.create(
        model=model,
        messages=[message],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.model_dump()

# 使用示例
if __name__ == "__main__":
    # 单图像推理示例
    result = qwen_single_image_inference(
        image_path="assets/example.png",
        prompt="What is in the image?"
    )
    print("单图像推理结果:", result)
    
    # 多图像推理示例
    result = qwen_multiple_images_inference(
        image_paths=["assets/example.png", "assets/image.png"],
        prompt="请比较这两张图片的差异"
    )
    print("多图像推理结果:", result)
    
    # 纯文本推理示例
    result = qwen_text_only_inference(
        prompt="Write a one-sentence bedtime story about a unicorn."
    )
    print("纯文本推理结果:", result)
    
    # 原始响应示例
    raw_result = qwen_raw_response(
        image_path="assets/example.png",
        prompt="What is in the image?"
    )
    print("原始响应:", raw_result) 