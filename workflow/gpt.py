from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI(api_key="sk-STKyxNjkvs5qdc2Yg0FepKdQ4ZcvoeJTxlOzzzfXLYASVV2P", base_url="http://35.220.164.252:3888/v1/")

def encode_image(image_path):
    """将图像文件编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_message_with_image(text, image_path=None):
    """创建包含文本和图像的消息"""
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

# 示例1: 纯文本输入
def text_only_example():
    print("=== 纯文本示例 ===")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Write a one-sentence bedtime story about a unicorn."
            }
        ]
    )
    print(response.choices[0].message.content)

# 示例2: 图像+文本输入
def image_and_text_example(image_path):
    print("=== 图像+文本示例 ===")
    message = create_message_with_image(
        text="请描述这张图片中的内容",
        image_path=image_path
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[message]
    )
    print(response.choices[0].message.content)

# 示例3: 多图像输入
def multiple_images_example(image_paths):
    print("=== 多图像示例 ===")
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请比较这两张图片的差异"
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
        model="gpt-4o-mini",
        messages=[message]
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    # 运行纯文本示例
    text_only_example()
    
    # 运行图像+文本示例（需要提供图像路径）
    image_path = "assets/image.png"  # 替换为你的图像路径
    image_and_text_example(image_path)
    
    # 运行多图像示例（需要提供多个图像路径）
    # image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]  # 替换为你的图像路径
    # multiple_images_example(image_paths)