#!/usr/bin/env python3
"""
使用PyTorch环境读取TFRecord文件并转换为对话格式的JSON
不依赖TensorFlow，使用tfrecord库
"""

import torch
import tfrecord  # 纯Python库用于读取TFRecord文件
import json
import os
from PIL import Image
import io

output_jsonl_path = 'dataset/ERQA_data.jsonl'
images_dir = 'dataset/ERQA_images'

def convert_erqa_to_conversations():
    """将ERQA数据集转换为对话格式的JSONL文件，并保存图片"""
    
    # 路径配置
    tfrecord_path = 'dataset/ERQA/erqa.tfrecord'

    # 创建图片目录
    os.makedirs(images_dir, exist_ok=True)
    
    print("🚀 ERQA数据集转换工具 (PyTorch版本)")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    print(f"输入文件: {tfrecord_path}")
    print(f"输出JSONL: {output_jsonl_path}")
    print(f"图片目录: {images_dir}")
    print("=" * 60)
    
    # 使用tfrecord库加载数据集
    dataset = tfrecord.tfrecord_loader(tfrecord_path, None)
    
    total_examples = 0
    total_images_saved = 0
    
    print("正在处理数据...")
    
    # 打开JSONL文件用于写入
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for i, example in enumerate(dataset):
            total_examples += 1
            
            try:
                # 提取和解码数据
                question = example['question'].decode('utf-8') if isinstance(example['question'], bytes) else str(example['question'])
                answer = example['answer'].decode('utf-8') if isinstance(example['answer'], bytes) else str(example['answer'])
                
                # 处理question_type
                question_type_raw = example['question_type']
                if isinstance(question_type_raw, bytes):
                    question_type = question_type_raw.decode('utf-8')
                elif hasattr(question_type_raw, '__len__') and len(question_type_raw) > 0:
                    if isinstance(question_type_raw[0], bytes):
                        question_type = question_type_raw[0].decode('utf-8')
                    else:
                        question_type = str(question_type_raw[0])
                else:
                    question_type = "Unknown"
                
                # 处理visual_indices
                visual_indices = example['visual_indices']
                if hasattr(visual_indices, 'tolist'):
                    visual_indices = visual_indices.tolist()
                elif not isinstance(visual_indices, list):
                    visual_indices = [visual_indices] if visual_indices is not None else []
                
                # 处理图像数据
                image_data = example['image/encoded']
                image_paths = []
                
                # 处理单张图片的情况 (bytes类型)
                if isinstance(image_data, bytes):
                    try:
                        # 保存图片
                        img = Image.open(io.BytesIO(image_data))
                        
                        # 生成文件名
                        filename = f"ERQA_{i:04d}.jpg"
                        filepath = os.path.join(images_dir, filename)
                        relative_path = f"ERQA_images/{filename}"

                        # 保存图片
                        img.save(filepath, 'JPEG')
                        img.close()
                        
                        image_paths.append(relative_path)
                        total_images_saved += 1
                        
                    except Exception as e:
                        print(f"警告: 保存图片失败 (示例 {i+1}): {e}")
                
                # 处理多张图片的情况 (numpy数组或列表)
                elif hasattr(image_data, '__len__') and len(image_data) > 0:
                    for j in range(len(image_data)):
                        img_bytes = image_data[j]
                        
                        # 确保是bytes格式
                        if isinstance(img_bytes, bytes):
                            try:
                                img = Image.open(io.BytesIO(img_bytes))
                                
                                filename = f"ERQA_{i:04d}_image_{j}.jpg"
                                filepath = os.path.join(images_dir, filename)
                                relative_path = f"ERQA_images/{filename}"
                                
                                img.save(filepath, 'JPEG')
                                img.close()
                                
                                image_paths.append(relative_path)
                                total_images_saved += 1
                                
                            except Exception as e:
                                print(f"警告: 保存图片 {j} 失败 (示例 {i+1}): {e}")
                        else:
                            print(f"警告: 示例 {i+1} 图片 {j} 不是bytes格式: {type(img_bytes)}")
                
                else:
                    print(f"警告: 示例 {i+1} 没有有效的图像数据")
                
                # 构建对话格式的数据
                conversation_entry = {
                    "id": f"erqa_{i:04d}",
                    "image": image_paths, 
                    "video": [],
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt", 
                            "value": answer
                        }
                    ],
                    "task": question_type,
                    "input_type": "Image", 
                    "output_type": "MCQ", 
                    "data_source": "ERQA", 
                    "others": {"visual_indices": visual_indices},
                }
                
                # 写入JSONL文件（每行一个JSON对象）
                jsonl_file.write(json.dumps(conversation_entry, ensure_ascii=False) + '\n')
                
                # 显示进度
                if (i + 1) % 50 == 0:
                    print(f"已处理 {i + 1} 个示例，保存了 {total_images_saved} 张图片...")
                    
            except Exception as e:
                print(f"错误: 处理示例 {i+1} 时出错: {e}")
                continue
    
    print(f"\n✅ 转换完成！")
    print(f"总共处理了 {total_examples} 个示例")
    print(f"成功保存了 {total_images_saved} 张图片")
    print(f"输出JSONL文件: {output_jsonl_path}")
    print(f"图片目录: {images_dir}")

def verify_conversion():
    """验证转换结果"""    
    print(f"\n🔍 验证转换结果...")
    print("=" * 60)
    data = []
    try:
        # 检查JSON文件
        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))

        
        print(f"JSON文件条目数: {len(data)}")
        
        # 显示前3个示例
        for i in range(min(3, len(data))):
            entry = data[i]
            print(f"\n示例 {i+1}:")
            print(f"  ID: {entry['id']}")
            print(f"  问题类型: {entry['task']}")
            print(f"  对话条数: {len(entry['conversations'])}")
            print(f"  Human: {entry['conversations'][0]['value'][:100]}...")
            print(f"  GPT: {entry['conversations'][1]['value']}")
            print(f"  图片数量: {len(entry['image'])}")
            print(f"  图片路径: {entry['image']}")
            print(f"  视觉索引: {entry['others'].get('visual_indices', [])}")
            
            # 验证图片文件
            for img_path in entry['image']:
                full_path = f"./data/{img_path}"
                exists = os.path.exists(full_path)
                print(f"    图片文件 {img_path}: {'存在' if exists else '不存在'}")
        
        # 检查图片目录
        if os.path.exists(images_dir):
            image_files = os.listdir(images_dir)
            print(f"\n图片目录中的文件数量: {len(image_files)}")
            print(f"示例图片文件: {image_files[:5]}")
        
        print(f"\n✅ 验证完成")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")

def show_statistics():
    """显示数据集统计信息"""
    print(f"\n📊 数据集统计信息:")
    print("=" * 60)
    
    try:
        data = []
        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))
        
        # 基本统计
        print(f"总示例数: {len(data)}")
        
        # 问题类型统计
        question_types = {}
        image_counts = {}
        
        for item in data:
            q_type = item['task']
            question_types[q_type] = question_types.get(q_type, 0) + 1
            
            img_count = len(item['image'])
            image_counts[img_count] = image_counts.get(img_count, 0) + 1
        
        print(f"\n问题类型分布:")
        for q_type, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {q_type}: {count}")
        
        print(f"\n图片数量分布:")
        for img_count, count in sorted(image_counts.items()):
            print(f"  {img_count} 张图片: {count} 个示例")
        
    except Exception as e:
        print(f"❌ 统计信息生成失败: {e}")

if __name__ == "__main__":
    print("🔥 使用PyTorch环境转换ERQA数据集")
    print("转换为对话格式并保存图片")
    print("=" * 60)
    
    # 执行转换
    convert_erqa_to_conversations()
    
    # 验证结果
    verify_conversion()
    
    # 显示统计信息
    show_statistics()
    