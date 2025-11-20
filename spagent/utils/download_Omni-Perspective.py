"""
下载并处理 Omni-Perspective 数据集
数据集来源: https://huggingface.co/datasets/Icey444/Omni-perspective

数据集字段说明:
- question/prompt: 包含 <image> 占位符的文本问题
- choices: 选项字典，键为 A, B, C, D
- label: 正确答案 (A/B/C/D)
- images: 图片序列

格式说明：
- 仿照 BLINK 数据集格式
- 图片保存为独立文件
- JSON 中存储图片路径
"""

import os
import json
from datasets import load_dataset, Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np


def convert_to_blink_format(parquet_dir="/home/ubuntu/Downloads", 
                            save_dir="dataset", 
                            pattern="val-*.parquet"):
    """
    将 Omni-Perspective 数据集转换为 BLINK 格式
    
    Args:
        parquet_dir: parquet文件所在目录
        save_dir: 保存目录
        pattern: 文件匹配模式
    
    Returns:
        转换的数据条目数量
    """
    print("=" * 60)
    print("开始处理 Omni-Perspective 数据集")
    print("=" * 60)
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 创建图片保存文件夹
    image_folder = save_path / "Omni_Perspective_images"
    print(f"\n准备创建图片文件夹: {image_folder.absolute()}")
    image_folder.mkdir(parents=True, exist_ok=True)
    print(f"图片文件夹创建成功!")
    
    # 加载数据集
    print(f"\n从本地加载 parquet 文件...")
    parquet_path = Path(parquet_dir)
    parquet_files = sorted(parquet_path.glob(pattern))
    
    if not parquet_files:
        raise FileNotFoundError(f"未找到匹配的parquet文件: {parquet_dir}/{pattern}")
    
    print(f"找到 {len(parquet_files)} 个parquet文件:")
    for f in parquet_files:
        print(f"  - {f.name}")
    
    dataset = load_dataset('parquet', data_files=[str(f) for f in parquet_files], split='train')
    print(f"\n数据集加载成功: {len(dataset)} 条样本")
    
    # 处理所有样本
    all_converted_data = []
    total_processed = 0
    
    print(f"\n开始转换数据...")
    
    for idx, sample in enumerate(dataset):
        try:
            # 处理图片 - 保存PIL对象为文件
            image_paths = []
            images = sample['images']
            
            for img_idx, pil_image in enumerate(images):
                # 图片文件命名: val_000000_img1.png
                image_filename = f"val_{idx:06d}_img{img_idx+1}.png"
                image_path = image_folder / image_filename
                
                # 检查并转换为PIL Image对象
                if hasattr(pil_image, 'save') and hasattr(pil_image, 'mode'):
                    # 是有效的PIL Image对象，直接保存
                    pil_image.save(str(image_path))
                elif hasattr(pil_image, '__array__'):
                    # 如果是numpy数组或类似对象，转换为PIL Image
                    from PIL import Image
                    import numpy as np
                    img_array = np.array(pil_image)
                    Image.fromarray(img_array).save(str(image_path))
                else:
                    # 尝试转换为PIL Image
                    from PIL import Image
                    Image.fromarray(pil_image).save(str(image_path))
                
                # 在JSON中保存相对于dataset目录的路径
                image_paths.append(f"Omni_Perspective_images/{image_filename}")
            
            # 解析 choices 字段（从JSON字符串到字典）
            choices_str = sample['choices']
            choices_dict = json.loads(choices_str.replace("'", '"'))
            
            # 构建对话内容
            conversations = []
            
            # 添加人类问题 - 使用 question 字段（完整问题）并添加选项
            # 注意：prompt 字段包含省略号，question 字段才是完整内容
            question_text = sample.get('question', sample.get('prompt', ''))
            
            # 添加选项信息
            if choices_dict:
                question_text += "\nSelect from the following choices.\n"
                for key in sorted(choices_dict.keys()):
                    question_text += f"({key}) {choices_dict[key]}\n"
            
            conversations.append({
                "from": "human",
                "value": question_text
            })
            
            # 添加GPT回答 - 只返回字母
            answer = sample['label']  # 已经是 'A', 'B', 'C', 或 'D'
            conversations.append({
                "from": "gpt",
                "value": answer
            })
            
            # 构建JSON条目（仿照BLINK格式）
            json_entry = {
                "id": f"Omni_Perspective_{idx}",
                "image": image_paths,  # 图片路径列表
                "video": [],
                "conversations": conversations,
                "task": sample.get('topic', 'Perspective'),  # 使用topic字段作为task
                "input_type": "image",
                "output_type": "MCQ",
                "data_source": "Omni_Perspective",
                "sub_task": "",  # 使用qtype作为sub_task
                "others": {
                    "image_id": sample.get('image_id', ''),
                    "qtype": sample.get('qtype', ''),
                    "difficulty_level": sample.get('difficulty_level', ''),
                    "topic": sample.get('topic', ''),
                    "prompt_pov": sample.get('prompt_pov', ''),
                    "id": sample.get('id', '')
                }
            }
            
            all_converted_data.append(json_entry)
            total_processed += 1
            
            if total_processed % 200 == 0:
                print(f"  已处理 {total_processed} / {len(dataset)} 条数据...")
                
        except Exception as e:
            print(f"  处理样本 {idx} 时出错: {e}")
            continue
    
    print(f"\n所有数据处理完成！总共处理了 {total_processed} 条数据")
    
    # 保存JSONL文件（每个条目一行）
    json_path = save_path / 'Omni_Perspective_All.jsonl'
    print(f"\n准备保存JSONL文件到: {json_path.absolute()}")
    
    if all_converted_data:
        with open(json_path, 'w', encoding='utf-8') as f:
            for item in all_converted_data:
                json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
        print(f"JSONL文件保存成功!")
    else:
        print("警告: 没有数据需要保存!")
    
    # 统计信息
    print(f"\n" + "=" * 60)
    print("数据转换完成！")
    print("=" * 60)
    print(f"总样本数: {len(all_converted_data)}")
    print(f"总图片数: {len(all_converted_data) * 5}")
    print(f"JSONL文件: {json_path}")
    print(f"图片文件夹: {image_folder}")
    
    # 统计各任务/主题的数据量
    task_counts = {}
    qtype_counts = {}
    difficulty_counts = {}
    
    for item in all_converted_data:
        task = item["task"]
        sub_task = item["sub_task"]
        difficulty = item["others"].get("difficulty_level", "unknown")
        
        task_counts[task] = task_counts.get(task, 0) + 1
        qtype_counts[sub_task] = qtype_counts.get(sub_task, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    print("\n按主题(topic)统计:")
    for task, count in sorted(task_counts.items(), key=lambda x: (x[0] is None, x[0] or "")):
        task_str = task if task is not None else "Unknown"
        print(f"  {task_str}: {count} 条")
    
    print("\n按问题类型(qtype)统计:")
    for qtype, count in sorted(qtype_counts.items(), key=lambda x: (x[0] is None, x[0] or "")):
        qtype_str = qtype if qtype is not None else "Unknown"
        print(f"  {qtype_str}: {count} 条")
    
    print("\n按难度级别统计:")
    for diff, count in sorted(difficulty_counts.items(), key=lambda x: (x[0] is None, x[0] or "")):
        diff_str = diff if diff is not None else "Unknown"
        print(f"  {diff_str}: {count} 条")
    
    # 打印第一条数据作为示例
    if all_converted_data:
        print("\n第一条数据示例:")
        print(json.dumps(all_converted_data[0], ensure_ascii=False, indent=2))
    
    return len(all_converted_data)



# ============================================================
# 以下是辅助函数，用于加载和探索数据集（可选使用）
# ============================================================

def load_from_parquet_files(parquet_dir="/home/ubuntu/Downloads", pattern="val-*.parquet"):
    """
    从本地parquet文件加载数据集（辅助函数）
    
    Args:
        parquet_dir: parquet文件所在目录
        pattern: 文件匹配模式
    
    Returns:
        加载的数据集
    """
    from datasets import load_dataset
    
    parquet_path = Path(parquet_dir)
    parquet_files = sorted(parquet_path.glob(pattern))
    
    if not parquet_files:
        raise FileNotFoundError(f"未找到匹配的parquet文件: {parquet_dir}/{pattern}")
    
    print(f"找到 {len(parquet_files)} 个parquet文件:")
    for f in parquet_files:
        print(f"  - {f.name}")
    
    # 加载所有parquet文件
    dataset = load_dataset('parquet', data_files=[str(f) for f in parquet_files], split='train')
    
    print(f"\n已加载本地数据集: {len(dataset)} 条样本")
    return dataset


def load_local_dataset(data_dir="./data/omni_perspective"):
    """
    从本地加载已下载的数据集（使用save_to_disk保存的格式）
    
    Args:
        data_dir: 数据集目录
    
    Returns:
        加载的数据集
    """
    from datasets import load_from_disk
    
    dataset_path = Path(data_dir) / "dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集未找到: {dataset_path}")
    
    dataset = load_from_disk(str(dataset_path))
    print(f"已加载本地数据集: {len(dataset)} 条样本")
    return dataset


def explore_dataset_structure(dataset=None, data_dir=None):
    """
    探索数据集结构
    
    Args:
        dataset: 数据集对象（如果已加载）
        data_dir: 数据集目录（如果需要加载）
    """
    if dataset is None:
        if data_dir is None:
            dataset = load_dataset("Icey444/Omni-perspective", split="val")
        else:
            dataset = load_local_dataset(data_dir)
    
    print("=== 数据集结构分析 ===")
    print(f"\n总样本数: {len(dataset)}")
    print(f"\n字段列表: {list(dataset.features.keys())}")
    
    # 分析每个字段
    sample = dataset[0]
    print("\n字段详情:")
    for key in sample.keys():
        print(f"\n【{key}】")
        value = sample[key]
        print(f"  - 类型: {type(value).__name__}")
        
        if key == 'images':
            print(f"  - 图片数量: {len(value) if hasattr(value, '__len__') else 'N/A'}")
        elif key == 'choices':
            if isinstance(value, dict):
                print(f"  - 键: {list(value.keys())}")
                print(f"  - 示例: {value}")
            elif isinstance(value, str):
                # 如果是JSON字符串，尝试解析
                try:
                    import json
                    parsed = json.loads(value)
                    print(f"  - JSON解析后的键: {list(parsed.keys())}")
                    print(f"  - 示例: {parsed}")
                except:
                    print(f"  - 示例: {value[:200]}...")
            else:
                print(f"  - 示例: {value}")
        elif key == 'label':
            print(f"  - 示例值: {value}")
        else:
            if isinstance(value, str):
                print(f"  - 示例: {value[:150]}...")
            else:
                print(f"  - 示例: {value}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转换 Omni-Perspective 数据集为 BLINK 格式")
    parser.add_argument("--parquet_dir", type=str, default="/home/ubuntu/Downloads",
                        help="parquet文件所在目录")
    parser.add_argument("--save_dir", type=str, default="dataset",
                        help="保存目录（将创建 Omni_Perspective_images 文件夹和 Omni_Perspective_All.jsonl）")
    parser.add_argument("--pattern", type=str, default="val-*.parquet",
                        help="parquet文件匹配模式")
    
    args = parser.parse_args()
    
    # 转换数据集
    try:
        total_converted = convert_to_blink_format(
            parquet_dir=args.parquet_dir,
            save_dir=args.save_dir,
            pattern=args.pattern
        )
        print(f"\n✓ 转换完成！共处理 {total_converted} 条数据")
    except Exception as e:
        print(f"\n✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
