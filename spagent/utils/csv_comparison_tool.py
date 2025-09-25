#!/usr/bin/env python3
"""
CSV对比分析工具
封装了3个主要功能：
1. extract_tool_improved: 提取使用工具正确不使用错误的数据
2. extract_tool_degraded: 提取使用工具错误不使用正确的数据  
3. extract_both_wrong: 提取两种方法都错误的数据
"""

import pandas as pd
import os

def load_and_merge_data(tool_results_path, no_tool_results_path):
    """
    加载并合并两个CSV文件
    
    Args:
        tool_results_path: 使用工具的结果CSV文件路径
        no_tool_results_path: 不使用    print("🚀 CSV对比分析工具")
    print(f"📁 使用工具结果: {args.tool_csv}")
    print(f"📁 基线结果: {args.baseline_csv}")
    print(f"📂 输出目录: {args.output_dir}")
    print("-" * 60)SV文件路径
    
    Returns:
        comparison_df: 合并后的DataFrame
        success: 是否成功加载
    """
    # 检查文件是否存在
    if not os.path.exists(tool_results_path):
        print(f"❌ 使用工具结果文件不存在: {tool_results_path}")
        return None, False
    
    if not os.path.exists(no_tool_results_path):
        print(f"❌ 不使用工具结果文件不存在: {no_tool_results_path}")
        return None, False
    
    # 检查并创建输出目录
    if not os.path.exists(args.output_dir):
        print(f"📁 输出目录不存在，正在创建: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"✅ 输出目录创建成功")
    else:
        print(f"✅ 输出目录已存在")

    # 读取数据
    try:
        tool_df = pd.read_csv(tool_results_path)
        no_tool_df = pd.read_csv(no_tool_results_path)
        print(f"✅ 成功读取数据:")
        print(f"  使用工具数据: {len(tool_df)} 条")
        print(f"  不使用工具数据: {len(no_tool_df)} 条")
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return None, False
    
    # 检查is_correct列
    if 'is_correct' not in tool_df.columns:
        print("❌ 使用工具数据缺少is_correct列")
        return None, False
    
    if 'is_correct' not in no_tool_df.columns:
        print("❌ 不使用工具数据缺少is_correct列")
        return None, False
    
    # 如果数据量不同，需要对齐
    if len(tool_df) != len(no_tool_df):
        print(f"⚠️  警告: 两个数据集长度不同，将使用较短的长度进行对比")
        min_len = min(len(tool_df), len(no_tool_df))
        tool_df = tool_df.head(min_len)
        no_tool_df = no_tool_df.head(min_len)
    
    # 添加索引以便对比
    tool_df['index'] = range(len(tool_df))
    no_tool_df['index'] = range(len(no_tool_df))
    
    # 获取所有列名，除了index
    tool_cols = [col for col in tool_df.columns if col != 'index']
    no_tool_cols = [col for col in no_tool_df.columns if col != 'index']
    
    # 合并数据进行对比
    comparison_df = pd.merge(
        tool_df[['index'] + tool_cols],
        no_tool_df[['index'] + no_tool_cols],
        on='index',
        suffixes=('_with_tools', '_without_tools')
    )
    
    print(f"✅ 合并后数据: {len(comparison_df)} 条")
    return comparison_df, True

def save_comparison_data(data_df, category, with_tools=True, output_dir="spagent/utils"):
    """
    保存对比数据到CSV文件
    
    Args:
        data_df: 要保存的DataFrame
        category: 分类名称，如"使用工具正确，不使用错误"
        with_tools: True保存with_tools版本，False保存without_tools版本
        output_dir: 输出目录
    
    Returns:
        output_file: 保存的文件路径
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    data_copy = data_df.copy()
    
    if with_tools:
        # 保留with_tools后缀的列，并重命名
        cols_mapping = {}
        for col in data_copy.columns:
            if col.endswith('_with_tools'):
                new_col = col.replace('_with_tools', '')
                cols_mapping[col] = new_col
            elif col in ['question', 'path', 'normalized_ground_truth', 'index']:
                cols_mapping[col] = col
        
        data_copy = data_copy[list(cols_mapping.keys())]
        data_copy = data_copy.rename(columns=cols_mapping)
        
        output_file = f"{output_dir}/使用工具的结果（{category}）.csv"
    else:
        # 保留without_tools后缀的列和共享列，并重命名
        cols_mapping = {}
        for col in data_copy.columns:
            if col.endswith('_without_tools'):
                new_col = col.replace('_without_tools', '')
                cols_mapping[col] = new_col
            elif col in ['question', 'path', 'normalized_ground_truth', 'index']:
                cols_mapping[col] = col
        
        data_copy = data_copy[list(cols_mapping.keys())]
        data_copy = data_copy.rename(columns=cols_mapping)
        
        # 如果原始数据没有某些列，添加空列以保持格式一致
        if 'used_tools' not in data_copy.columns:
            data_copy['used_tools'] = ''
        if 'follow_up_prompt' not in data_copy.columns:
            data_copy['follow_up_prompt'] = ''
        
        output_file = f"{output_dir}/未使用工具的结果（{category}）.csv"
    
    # 删除index列
    if 'index' in data_copy.columns:
        data_copy = data_copy.drop('index', axis=1)
    
    # 保存文件
    data_copy.to_csv(output_file, index=False, encoding='utf-8')
    return output_file

def extract_tool_improved(tool_results_path, no_tool_results_path, output_dir="spagent/utils"):
    """
    提取使用工具正确、不使用工具错误的数据
    
    Args:
        tool_results_path: 使用工具的结果CSV文件路径
        no_tool_results_path: 不使用工具的结果CSV文件路径
        output_dir: 输出目录
    
    Returns:
        success: 是否成功执行
        count: 提取的数据条数
    """
    print("🎯 开始提取使用工具改进的数据...")
    
    # 加载并合并数据
    comparison_df, success = load_and_merge_data(tool_results_path, no_tool_results_path)
    if not success:
        return False, 0
    
    # 筛选使用工具正确，不使用工具错误的数据
    tools_correct_no_tools_wrong = comparison_df[
        (comparison_df['is_correct_with_tools'] == 1) & 
        (comparison_df['is_correct_without_tools'] == 0)
    ]
    
    count = len(tools_correct_no_tools_wrong)
    print(f"📊 找到 {count} 条使用工具改进的数据")
    
    if count > 0:
        # 保存使用工具的结果
        with_tools_file = save_comparison_data(
            tools_correct_no_tools_wrong, 
            "使用工具正确，不使用错误", 
            with_tools=True, 
            output_dir=output_dir
        )
        print(f"✅ 保存使用工具结果: {with_tools_file}")
        
        # 保存不使用工具的结果
        without_tools_file = save_comparison_data(
            tools_correct_no_tools_wrong, 
            "使用工具正确，不使用错误", 
            with_tools=False, 
            output_dir=output_dir
        )
        print(f"✅ 保存未使用工具结果: {without_tools_file}")
        
    else:
        print("⚠️  没有发现使用工具改进的案例")
    
    return True, count

def extract_tool_degraded(tool_results_path, no_tool_results_path, output_dir="spagent/utils"):
    """
    提取使用工具错误、不使用工具正确的数据
    
    Args:
        tool_results_path: 使用工具的结果CSV文件路径
        no_tool_results_path: 不使用工具的结果CSV文件路径
        output_dir: 输出目录
    
    Returns:
        success: 是否成功执行
        count: 提取的数据条数
    """
    print("⚠️  开始提取使用工具恶化的数据...")
    
    # 加载并合并数据
    comparison_df, success = load_and_merge_data(tool_results_path, no_tool_results_path)
    if not success:
        return False, 0
    
    # 筛选使用工具错误，不使用工具正确的数据
    tools_wrong_no_tools_correct = comparison_df[
        (comparison_df['is_correct_with_tools'] == 0) & 
        (comparison_df['is_correct_without_tools'] == 1)
    ]
    
    count = len(tools_wrong_no_tools_correct)
    print(f"📊 找到 {count} 条使用工具恶化的数据")
    
    if count > 0:
        # 保存使用工具的结果
        with_tools_file = save_comparison_data(
            tools_wrong_no_tools_correct, 
            "使用工具错误，不使用正确", 
            with_tools=True, 
            output_dir=output_dir
        )
        print(f"✅ 保存使用工具错误结果: {with_tools_file}")
        
        # 保存不使用工具的结果
        without_tools_file = save_comparison_data(
            tools_wrong_no_tools_correct, 
            "使用工具错误，不使用正确", 
            with_tools=False, 
            output_dir=output_dir
        )
        print(f"✅ 保存未使用工具正确结果: {without_tools_file}")
       
    else:
        print("✅ 没有发现使用工具恶化的案例")
    
    return True, count

def extract_both_wrong(tool_results_path, no_tool_results_path, output_dir="spagent/utils"):
    """
    提取两种方法都错误的数据
    
    Args:
        tool_results_path: 使用工具的结果CSV文件路径
        no_tool_results_path: 不使用工具的结果CSV文件路径
        output_dir: 输出目录
    
    Returns:
        success: 是否成功执行
        count: 提取的数据条数
    """
    print("😞 开始提取两种方法都错误的数据...")
    
    # 加载并合并数据
    comparison_df, success = load_and_merge_data(tool_results_path, no_tool_results_path)
    if not success:
        return False, 0
    
    # 筛选两种方法都错误的数据
    both_wrong = comparison_df[
        (comparison_df['is_correct_with_tools'] == 0) & 
        (comparison_df['is_correct_without_tools'] == 0)
    ]
    
    count = len(both_wrong)
    print(f"📊 找到 {count} 条两种方法都错误的数据")
    
    if count > 0:
        # 保存使用工具的结果
        with_tools_file = save_comparison_data(
            both_wrong, 
            "两种方法都错误", 
            with_tools=True, 
            output_dir=output_dir
        )
        print(f"✅ 保存使用工具错误结果: {with_tools_file}")
        
        # 保存不使用工具的结果
        without_tools_file = save_comparison_data(
            both_wrong, 
            "两种方法都错误", 
            with_tools=False, 
            output_dir=output_dir
        )
        print(f"✅ 保存未使用工具错误结果: {without_tools_file}")
        
        print("✅ 没有发现两种方法都错误的案例")
    
    return True, count

def analyze_all_comparisons(tool_results_path, no_tool_results_path, output_dir="spagent/utils"):
    """
    运行所有三种对比分析
    
    Args:
        tool_results_path: 使用工具的结果CSV文件路径
        no_tool_results_path: 不使用工具的结果CSV文件路径
        output_dir: 输出目录
    
    Returns:
        results: 包含所有分析结果的字典
    """
    print("🚀 开始完整的CSV对比分析...\n")
    
    results = {
        'improved': {'success': False, 'count': 0},
        'degraded': {'success': False, 'count': 0},
        'both_wrong': {'success': False, 'count': 0}
    }
    
    # 运行三种分析
    print("=" * 60)
    success, count = extract_tool_improved(tool_results_path, no_tool_results_path, output_dir)
    results['improved'] = {'success': success, 'count': count}
    
    print("\n" + "=" * 60)
    success, count = extract_tool_degraded(tool_results_path, no_tool_results_path, output_dir)
    results['degraded'] = {'success': success, 'count': count}
    
    print("\n" + "=" * 60)
    success, count = extract_both_wrong(tool_results_path, no_tool_results_path, output_dir)
    results['both_wrong'] = {'success': success, 'count': count}
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("📋 完整分析总结:")
    
    total_cases = sum([results[key]['count'] for key in results if results[key]['success']])
    if total_cases > 0:
        print(f"  🎯 工具改进案例: {results['improved']['count']} 条")
        print(f"  ⚠️  工具恶化案例: {results['degraded']['count']} 条")
        print(f"  😞 两种都错案例: {results['both_wrong']['count']} 条")
        
        if results['improved']['count'] > 0 and results['degraded']['count'] > 0:
            improvement_rate = results['improved']['count'] / (results['improved']['count'] + results['degraded']['count']) * 100
            degradation_rate = results['degraded']['count'] / (results['improved']['count'] + results['degraded']['count']) * 100
            net_improvement = improvement_rate - degradation_rate
            
            print(f"  📈 工具改进率: {improvement_rate:.2f}%")
            print(f"  📉 工具恶化率: {degradation_rate:.2f}%")
            print(f"  📊 净改进率: {net_improvement:.2f}%")
    
    print(f"\n🎉 分析完成！所有结果已保存到: {output_dir}")
    
    return results

# 使用示例
if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="CSV对比分析工具 - 分析使用工具和不使用工具的评测结果差异",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python csv_comparison_tool.py --tool-csv tool_results.csv --baseline-csv baseline_results.csv
  python csv_comparison_tool.py --tool-csv tool.csv --baseline-csv baseline.csv --output-dir=output
        """
    )
    
    # 可选参数
    parser.add_argument(
        "--tool-csv", "-t",
        help="使用工具的结果CSV文件路径"
    )
    parser.add_argument(
        "--baseline-csv", "-b",
        help="不使用工具的结果CSV文件路径（基线）"
    )
    
    # 可选参数
    parser.add_argument(
        "--output-dir", "-o",
        default="spagent/utils",
        help="输出目录 (默认: spagent/utils)"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    print("� CSV对比分析工具")
    print(f"📁 使用工具结果: {args.tool_csv}")
    print(f"📁 基线结果: {args.baseline_csv}")
    print(f"📂 输出目录: {args.output_dir}")
    print("-" * 60)
    
    # 运行完整分析（包含所有3种分析）
    print("🔄 运行完整分析（包含工具改进、工具恶化、两种都错3种分析）...")
    results = analyze_all_comparisons(args.tool_csv, args.baseline_csv, args.output_dir)
    
    # 检查结果
    success_count = sum(1 for result in results.values() if result['success'])
    if success_count == 3:
        print(f"✅ 完成！成功运行了所有 3 种分析")
        print(f"   🎯 工具改进: {results['improved']['count']} 条")
        print(f"   ⚠️  工具恶化: {results['degraded']['count']} 条")
        print(f"   � 两种都错: {results['both_wrong']['count']} 条")
    elif success_count > 0:
        print(f"⚠️  部分完成！成功运行了 {success_count}/3 个分析")
    else:
        print("❌ 所有分析都失败了")
    
    print("\n🎉 程序执行完毕！")