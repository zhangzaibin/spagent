import os
import base64
import requests
import mimetypes
from openai import OpenAI
from pathlib import Path
from typing import List, Optional, Union

from qwen import encode_image, create_message_with_image

# Initialize Qwen client with vLLM configuration
client = OpenAI(
    api_key="dummy",  # vLLM doesn't require a real API key
    base_url="http://10.8.131.51:30058/v1",
)

def qwen_single_image_inference(
    image_path: str, 
    prompt: str, 
    model: str = "qwen-vl-2B",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """Qwen单图像推理函数（使用vLLM）"""
    try:
        message = create_message_with_image(prompt, image_path)
        
        print(f"发送请求到模型: {model}")
        print(f"提示词: {prompt}")
        print(f"图像路径: {image_path}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[message],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"推理过程中出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        return f"推理失败: {str(e)}"

def qwen_multiple_images_inference(
    image_paths: List[str], 
    prompt: str, 
    model: str = "qwen-vl-2B",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """Qwen多图像推理函数（使用vLLM）"""
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
    
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
    model: str = "qwen-vl-2B",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """Qwen纯文本推理函数（使用vLLM）"""
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

    result = qwen_single_image_inference(
        image_path="assets/example.png",
        prompt="What is in the image?"
    )
    print(result)
