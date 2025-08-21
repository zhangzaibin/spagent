from datasets import load_dataset, load_from_disk
import json
import sys
import ast
import os
from PIL import Image
import requests
from io import BytesIO

# BLINK数据集的所有子任务
subtasks = [
    'Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 
    'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 
    'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 
    'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 
    'Relative_Depth', 'Spatial_Relation'
]

# 创建图片保存文件夹
image_folder = "dataset/BLINK_images"
print(f"准备创建图片文件夹: {os.path.abspath(image_folder)}")
os.makedirs(image_folder, exist_ok=True)
print(f"图片文件夹创建成功: {os.path.exists(image_folder)}")

# 检查dataset文件夹是否存在
dataset_folder = "dataset"
print(f"检查dataset文件夹: {os.path.abspath(dataset_folder)}")
os.makedirs(dataset_folder, exist_ok=True)
print(f"dataset文件夹存在: {os.path.exists(dataset_folder)}")

# 处理所有子任务的数据
all_converted_data = []
total_processed = 0

for subtask in subtasks:
    print(f"\n开始处理子任务: {subtask}")
    
    try:
        # 加载每个子任务的数据集
        dataset = load_dataset("BLINK-Benchmark/BLINK", subtask)
        
        # 只处理验证集
        if "val" in dataset:
            samples = dataset["val"]
            print(f"  处理验证集: {len(samples)} 条数据")
            
            for idx, sample in enumerate(samples):
                # 处理图片
                image_paths = []
                
                # 遍历所有可能的图片字段
                image_keys = [key for key in sample.keys() if key.startswith('image_')]
                image_keys.sort()  # 确保顺序：image_1, image_2, image_3...
                
                for img_key in image_keys:
                    if sample[img_key] is not None and hasattr(sample[img_key], 'save'):
                        # 提取图片编号
                        img_num = img_key.split('_')[1]
                        image_filename = f"{subtask}_val_{idx:06d}_img{img_num}.jpg"
                        image_path = os.path.join(image_folder, image_filename)
                        sample[img_key].save(image_path)
                        # 在JSON中保存相对于dataset目录的路径
                        image_paths.append(f"BLINK_images/{image_filename}")
                
                # 构建对话内容
                conversations = []
                
                # 添加人类问题 - 使用prompt字段（如果存在）或question字段
                question_text = ""
                if "prompt" in sample and sample["prompt"]:
                    question_text = sample["prompt"]
                elif "question" in sample and sample["question"]:
                    question_text = sample["question"]
                    # 如果有选项，添加选项信息
                    if "choices" in sample and sample["choices"]:
                        question_text += "\nSelect from the following choices.\n"
                        for i, choice in enumerate(sample["choices"]):
                            question_text += f"({chr(65+i)}) {choice}\n"
                
                if question_text:
                    conversations.append({
                        "from": "human",
                        "value": question_text
                    })
                
                # 添加GPT回答
                if "answer" in sample and sample["answer"] != "hidden":
                    answer = sample["answer"]
                    
                    # 如果答案是 "(A)" 格式，提取括号中的字母
                    if isinstance(answer, str) and answer.startswith("(") and answer.endswith(")"):
                        answer = answer[1:-1]  # 去掉括号，只保留字母
                    
                    # 如果答案是choice中的文本，尝试转换为字母
                    elif "choices" in sample and sample["choices"]:
                        try:
                            if answer in sample["choices"]:
                                answer_idx = sample["choices"].index(answer)
                                answer = chr(65 + answer_idx)  # 转换为A, B, C...
                        except:
                            pass
                    
                    conversations.append({
                        "from": "gpt", 
                        "value": str(answer)
                    })
                
                # 构建JSON条目
                json_entry = {
                    "id": f"{subtask}_BLINK_{idx}",
                    "image": image_paths,  # 图片路径列表
                    "video": [],
                    "conversations": conversations,
                    "task": subtask,
                    "input_type": "image",
                    "output_type": "MCQ",
                    "data_source": "BLINK",
                    "sub_task": "",
                    "others": {}
                }
                
                # 添加原始数据中的其他字段到others中
                excluded_keys = {"question", "answer", "choices", "prompt", "sub_task"} | set(image_keys)
                for key, value in sample.items():
                    if key not in excluded_keys:
                        json_entry["others"][key] = value
                
                all_converted_data.append(json_entry)
                total_processed += 1
                
                if total_processed % 500 == 0:
                    print(f"    已总共处理 {total_processed} 条数据...")
                        
    except Exception as e:
        print(f"  处理子任务 {subtask} 时出错: {e}")
        continue

print(f"\n所有子任务处理完成！总共处理了 {total_processed} 条数据")

# 保存JSONL文件（每个ID一行）
json_path = 'dataset/BLINK_All_Tasks.jsonl'
print(f"准备保存JSONL文件到: {os.path.abspath(json_path)}")
print(f"数据条目数量: {len(all_converted_data)}")

if all_converted_data:
    with open(json_path, 'w', encoding='utf-8') as f:
        for item in all_converted_data:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    print(f"JSONL文件保存成功!")
else:
    print("警告: 没有数据需要保存!")

print(f"\n数据转换完成！")
print(f"总共处理了 {len(all_converted_data)} 条数据")
print(f"JSONL文件保存为: {json_path}")
print(f"图片保存在文件夹: {image_folder}")

# 统计每个任务的数据量
task_counts = {}
for item in all_converted_data:
    task = item["task"]
    task_counts[task] = task_counts.get(task, 0) + 1

print("\n各任务数据统计:")
for task, count in sorted(task_counts.items()):
    print(f"  {task}: {count} 条")

# 打印第一条数据作为示例
if all_converted_data:
    print("\n第一条数据示例:")
    print(json.dumps(all_converted_data[0], ensure_ascii=False, separators=(',', ':')))
