from openai import OpenAI
import base64
from pathlib import Path
from typing import List, Optional, Union
import os

# client = OpenAI()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
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

def gpt_single_image_inference(
    image_path: str, 
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """GPT单图像推理函数
    
    Args:
        image_path: 图像文件路径
        prompt: 提示文本
        model: 使用的模型名称
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        GPT的回复文本
    """
    message = create_message_with_image(prompt, image_path)
    
    response = client.chat.completions.create(
        model=model,
        messages=[message],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

def gpt_multiple_images_inference(
    image_paths: List[str], 
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """GPT多图像推理函数
    
    Args:
        image_paths: 图像文件路径列表
        prompt: 提示文本
        model: 使用的模型名称
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        GPT的回复文本
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

def gpt_text_only_inference(
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """GPT纯文本推理函数
    
    Args:
        prompt: 提示文本
        model: 使用的模型名称
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        GPT的回复文本
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

# 使用示例
if __name__ == "__main__":
    # 单图像推理示例
    result = gpt_single_image_inference(
        image_path="assets/image.png",
        prompt="请描述这张图片中的内容"
    )
    print("单图像推理结果:", result)
    
    # 多图像推理示例
    # result = gpt_multiple_images_inference(
    #     image_paths=["assets/image1.png", "assets/image2.png"],
    #     prompt="请比较这两张图片的差异"
    # )
    # print("多图像推理结果:", result)
    
    # 纯文本推理示例
    # result = gpt_text_only_inference(
    #     prompt="Write a one-sentence bedtime story about a unicorn."
    # )
    # print("纯文本推理结果:", result)