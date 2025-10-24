#!/usr/bin/env python3
"""
将 MindCube 数据集转换为统一的 JSONL 格式（类似 BLINK）
"""

import os
import json
import argparse
import subprocess
import sys


def parse_answer_from_question(question, gt_answer):
    """
    从问题中解析选项，并将 gt_answer (如 "C") 转换为实际答案文本
    如果无法解析，则直接返回原始 gt_answer
    """
    # 尝试从问题中提取选项
    lines = question.split('\n')
    options = {}
    
    for line in lines:
        line = line.strip()
        # 匹配格式如 "A. xxx" 或 "(A) xxx"
        if line and len(line) > 3:
            if line[0] in ['A', 'B', 'C', 'D'] and line[1] in ['.', ')']:
                option_letter = line[0]
                option_text = line[2:].strip()
                if option_text.startswith(')'):
                    option_text = option_text[1:].strip()
                options[option_letter] = option_text
    
    # 如果找到了选项且 gt_answer 是字母，返回对应的文本
    if options and gt_answer in options:
        return options[gt_answer]
    
    # 否则返回原始答案
    return gt_answer


def convert_mindcube_to_blink_format(input_file, output_file, image_prefix="mindcube/data/"):
    """
    将 MindCube 的 JSONL 文件转换为 BLINK 格式
    
    Args:
        input_file: 输入的 MindCube JSONL 文件路径
        output_file: 输出的 BLINK 格式 JSONL 文件路径
        image_prefix: 图片路径前缀
    """
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    print(f"📂 读取文件: {input_file}")
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                
                # 提取原始字段
                original_id = item.get("id", f"MindCube_{idx}")
                question = item.get("question", "")
                gt_answer = item.get("gt_answer", "")
                images = item.get("images", [])
                category = item.get("category", [])
                item_type = item.get("type", "")
                meta_info = item.get("meta_info", [])
                
                # 转换图片路径：添加前缀
                converted_images = [os.path.join(image_prefix, img) for img in images]
                
                # 解析答案（尝试将字母转换为完整答案）
                answer_text = parse_answer_from_question(question, gt_answer)
                
                # 构建 conversations 格式
                conversations = [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer_text}
                ]
                
                # 确定任务类型：从 ID 中提取 _ 前面的部分
                # 例如 "among_group693_gen_6_2" -> "among"
                task_name = original_id.split('_')[0] if '_' in original_id else "spatial_reasoning"
                
                # 构建新格式的数据项
                converted_item = {
                    "id": original_id,
                    "image": converted_images,
                    "video": [],
                    "conversations": conversations,
                    "task": task_name,
                    "input_type": "image",
                    "output_type": "MCQ",
                    "data_source": "MindCube",
                    "sub_task": "",
                    "others": {
                        "category": category,
                        "meta_info": meta_info
                    }
                }
                
                converted_data.append(converted_item)
                
            except Exception as e:
                print(f"  ⚠️ 处理第 {idx} 行时出错: {e}")
                continue
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\n✅ 转换完成!")
    print(f"📊 总共转换 {len(converted_data)} 条数据")
    print(f"📁 输出文件: {os.path.abspath(output_file)}")
    
    # 显示示例
    if converted_data:
        print("\n📄 示例数据:")
        print(json.dumps(converted_data[0], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description='将 MindCube 数据集转换为 BLINK 格式')
    parser.add_argument('--input', '-i', 
                       default='dataset/mindcube/data/raw/MindCube_tinybench.jsonl',
                       help='输入的 MindCube JSONL 文件路径')
    parser.add_argument('--output', '-o',
                       default='dataset/MindCube_data.jsonl',
                       help='输出的 JSONL 文件路径')
    parser.add_argument('--image-prefix', '-p',
                       default='mindcube/data/',
                       help='图片路径前缀')
    parser.add_argument('--skip-download', 
                       action='store_true',
                       help='跳过下载步骤，直接转换数据')
    
    args = parser.parse_args()
    
    # 如果没有跳过下载，先运行下载脚本
    if not args.skip_download:
        print("=" * 60)
        print("📥 步骤 1/2: 下载 MindCube 数据集")
        print("=" * 60)
        
        download_script = "spagent/utils/download_MindCube.sh"
        
        if not os.path.exists(download_script):
            print(f"⚠️  下载脚本不存在: {download_script}")
            print("   继续执行转换步骤...")
        else:
            try:
                # 运行下载脚本
                result = subprocess.run(
                    ["bash", download_script],
                    capture_output=False,
                    text=True,
                    check=True
                )
                print("\n✅ 数据集下载完成\n")
            except subprocess.CalledProcessError as e:
                print(f"\n⚠️  下载脚本执行失败: {e}")
                print("   继续执行转换步骤...\n")
            except Exception as e:
                print(f"\n⚠️  执行下载时出错: {e}")
                print("   继续执行转换步骤...\n")
    
    # 转换数据
    print("=" * 60)
    print("🔄 步骤 2/2: 转换数据格式")
    print("=" * 60)
    convert_mindcube_to_blink_format(args.input, args.output, args.image_prefix)


if __name__ == "__main__":
    main()
