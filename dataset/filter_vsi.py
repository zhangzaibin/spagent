#!/usr/bin/env python3
"""
从VSI_Bench.jsonl中每个task类别选取指定数量的样本
"""

import json
import random
from collections import defaultdict
import os


def filter_vsi_by_task(
    input_jsonl="dataset/VSI_Bench.jsonl",
    output_jsonl="dataset/VSI_Bench_filtered.jsonl",
    samples_per_task=20,
    random_seed=42
):
    """
    从VSI_Bench.jsonl中每个task类别选取指定数量的样本
    
    参数:
        input_jsonl (str): 输入的JSONL文件路径
        output_jsonl (str): 输出的JSONL文件路径
        samples_per_task (int): 每个task类别选取的样本数量
        random_seed (int): 随机种子，用于可重复的随机选择
    """
    
    # 设置随机种子
    random.seed(random_seed)
    
    print(f"📖 读取数据文件: {input_jsonl}")
    
    # 读取所有数据并按task分类
    task_data = defaultdict(list)
    total_count = 0
    
    try:
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    task = data.get('task', 'unknown')
                    task_data[task].append(data)
                    total_count += 1
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在 {input_jsonl}")
        return
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return
    
    print(f"✅ 读取完成，共 {total_count} 条数据")
    print(f"\n📊 各task类别统计:")
    for task, samples in sorted(task_data.items()):
        print(f"  {task}: {len(samples)} 条")
    
    # 从每个task类别中随机选取样本
    filtered_data = []
    task_stats = {}
    
    print(f"\n🎲 从每个task类别随机选取 {samples_per_task} 个样本...")
    for task, samples in sorted(task_data.items()):
        if len(samples) <= samples_per_task:
            # 如果该类别样本数不足，全部选取
            selected = samples
            print(f"  {task}: 选取 {len(selected)} 条 (全部样本，不足{samples_per_task}条)")
        else:
            # 随机选取指定数量
            selected = random.sample(samples, samples_per_task)
            print(f"  {task}: 选取 {len(selected)} 条")
        
        filtered_data.extend(selected)
        task_stats[task] = len(selected)
    
    # 保存到新文件
    print(f"\n💾 保存筛选后的数据到: {output_jsonl}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_jsonl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
        print(f"✅ 保存成功!")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")
        return
    
    # 输出统计信息
    print(f"\n📈 筛选结果统计:")
    print(f"  原始数据总量: {total_count} 条")
    print(f"  筛选后数据量: {len(filtered_data)} 条")
    print(f"  task类别数量: {len(task_stats)} 个")
    
    print(f"\n📊 筛选后各task类别分布:")
    for task, count in sorted(task_stats.items()):
        print(f"  {task}: {count} 条")
    
    # 打印第一条数据作为示例
    if filtered_data:
        print(f"\n📄 第一条数据示例:")
        print(json.dumps(filtered_data[0], ensure_ascii=False, indent=2))
    
    print(f"\n🎉 筛选完成!")
    print(f"   输入文件: {input_jsonl}")
    print(f"   输出文件: {output_jsonl}")


if __name__ == "__main__":
    # 默认从每个task类别选取20个样本
    filter_vsi_by_task(
        input_jsonl="dataset/VSI_Bench.jsonl",
        output_jsonl="dataset/VSI_Bench_filtered.jsonl",
        samples_per_task=20,
        random_seed=42
    )






