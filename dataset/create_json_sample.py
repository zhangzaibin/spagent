#!/usr/bin/env python3
"""
从数据文件中按任务抽取样本数据
每个任务抽取指定数量的样本，生成新的sample文件
"""

import json
import random
from collections import defaultdict
import os
import argparse


def extract_samples_by_task(input_file, output_file, samples_per_task=30):
    """
    从JSONL文件中按任务抽取样本
    
    Args:
        input_file (str): 输入的JSONL文件路径
        output_file (str): 输出的JSONL文件路径
        samples_per_task (int): 每个任务抽取的样本数量
    """
    
    print(f"🚀 开始从 {input_file} 抽取样本数据")
    print(f"📊 每个任务抽取: {samples_per_task} 个样本")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    # 读取所有数据并按任务分类
    task_data = defaultdict(list)
    total_count = 0
    
    print("📖 读取数据并按任务分类...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                task = data.get('task', 'unknown')
                task_data[task].append(data)
                total_count += 1
                
                if line_num % 500 == 0:
                    print(f"  已读取 {line_num} 条数据...")
                    
            except json.JSONDecodeError as e:
                print(f"❌ 第 {line_num} 行JSON解析错误: {e}")
                continue
    
    print(f"✅ 数据读取完成，总计 {total_count} 条数据")
    
    # 显示任务分布
    print(f"\n📈 任务分布统计:")
    for task, data_list in sorted(task_data.items()):
        print(f"  {task}: {len(data_list)} 条")
    
    # 从每个任务中抽取样本
    print(f"\n🎯 开始抽取样本 (每个任务 {samples_per_task} 个):")
    selected_samples = []
    task_sample_counts = {}
    
    for task, data_list in sorted(task_data.items()):
        available_count = len(data_list)
        sample_count = min(samples_per_task, available_count)
        
        # 随机抽取样本
        if sample_count > 0:
            sampled_data = random.sample(data_list, sample_count)
            selected_samples.extend(sampled_data)
            task_sample_counts[task] = sample_count
            
            status = "✅" if sample_count == samples_per_task else "⚠️"
            print(f"  {status} {task}: 抽取 {sample_count}/{available_count} 个样本")
        else:
            task_sample_counts[task] = 0
            print(f"  ❌ {task}: 无可用数据")
    
    # 按原始顺序排序（基于ID中的索引）
    print(f"\n📝 对抽取的样本进行排序...")
    def extract_idx_from_id(data):
        try:
            return int(data['id'].split('_')[-1])
        except:
            return 0
    
    selected_samples.sort(key=extract_idx_from_id)
    
    # 保存抽取的样本
    print(f"\n💾 保存样本到: {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            json.dump(sample, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print(f"✅ 样本保存完成!")
    
    # 输出最终统计
    print(f"\n📊 抽取结果统计:")
    print(f"  总样本数: {len(selected_samples)}")
    print(f"  任务数量: {len(task_sample_counts)}")
    
    print(f"\n📈 各任务样本统计:")
    for task, count in sorted(task_sample_counts.items()):
        percentage = (count / samples_per_task * 100) if samples_per_task > 0 else 0
        print(f"  {task}: {count} 个样本 ({percentage:.1f}%)")
    
    # 验证生成的文件
    print(f"\n🔍 验证生成的文件...")
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            actual_count = sum(1 for _ in f)
        print(f"  ✅ 文件验证成功，实际包含 {actual_count} 条数据")
    else:
        print(f"  ❌ 文件生成失败")
        return False
    
    return True


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从数据文件中按任务抽取样本数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python create_json_sample.py --input_file dataset/ERQA_All_Data.jsonl --sample 30
  python create_json_sample.py --input_file dataset/cvbench_data.jsonl --sample 50
        """
    )
    
    # 添加必需的参数
    parser.add_argument("--input_file", type=str, required=True, help="输入的JSONL文件路径")
    parser.add_argument("--sample", type=int, required=True, help="每个任务抽取的样本数量")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 自动生成输出文件名
    input_path = args.input_file
    base_name = os.path.splitext(input_path)[0]  # 移除文件扩展名
    extension = os.path.splitext(input_path)[1]  # 获取文件扩展名
    output_file = f"{base_name}_sample{args.sample}{extension}"
    
    print("🚀 样本抽取工具")
    print("=" * 80)
    print(f"📁 输入文件: {args.input_file}")
    print(f"📁 输出文件: {output_file}")
    print(f"📊 每任务样本数: {args.sample}")
    print("=" * 80)
    
    # 执行抽取
    success = extract_samples_by_task(
        input_file=args.input_file,
        output_file=output_file,
        samples_per_task=args.sample
    )
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 样本抽取完成！")
        print(f"✅ 输入文件: {args.input_file}")
        print(f"✅ 输出文件: {output_file}")
        print(f"✅ 每任务样本数: {args.sample}")
    else:
        print("❌ 样本抽取失败，请检查错误信息")
    
    return success


if __name__ == "__main__":
    main()