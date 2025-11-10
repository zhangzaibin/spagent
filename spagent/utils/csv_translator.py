#!/usr/bin/env python3
"""
CSV翻译工具
专门用于翻译CSV文件中的特定列（analysis和follow_up_prompt）到中文
"""

import pandas as pd
import os
import argparse
from typing import Any

# 尝试导入翻译库
try:
    import requests
    import json
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️  警告: requests未安装，翻译功能不可用。运行 pip install requests 安装")

def translate_text(text: str, translator: Any = None) -> str:
    """
    翻译文本到中文
    
    Args:
        text: 要翻译的文本
        translator: 翻译器实例（此参数保留兼容性，实际不使用）
    
    Returns:
        翻译后的文本，如果翻译失败则返回原文本
    """
    if not TRANSLATOR_AVAILABLE:
        return text
    
    if not text or not text.strip():
        return text
    
    # 如果文本太长，截断处理
    max_length = 5000  # Google Translate API 限制
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    try:
        # 使用 Google Translate API（免费版本）
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'auto',  # 自动检测源语言
            'tl': 'zh-cn',  # 目标语言：简体中文
            'dt': 't',
            'q': text
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        if result and len(result) > 0 and len(result[0]) > 0:
            translated_text = ''.join([item[0] for item in result[0] if item[0]])
            return translated_text
        
        return text
    
    except Exception as e:
        print(f"⚠️  翻译失败: {e}")
        return text

def translate_csv_file(input_file: str, output_file: str = None, columns_to_translate: list = None):
    """
    翻译CSV文件中指定列的内容
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径，如果为None则自动生成
        columns_to_translate: 需要翻译的列名列表，默认为['analysis', 'follow_up_prompt']
    
    Returns:
        bool: 是否成功完成翻译
    """
    if not TRANSLATOR_AVAILABLE:
        print("❌ 翻译库不可用，无法进行翻译")
        return False
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    # 设置默认需要翻译的列
    if columns_to_translate is None:
        columns_to_translate = ['analysis', 'follow_up_prompt']
    
    # 生成输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_translated.csv"
    
    print(f"🔄 开始翻译CSV文件...")
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出文件: {output_file}")
    print(f"🌐 翻译列: {', '.join(columns_to_translate)}")
    print("-" * 60)
    
    try:
        # 读取CSV文件，添加错误处理
        df = pd.read_csv(input_file, encoding='utf-8', on_bad_lines='skip', engine='python')
        print(f"✅ 成功读取CSV文件，共 {len(df)} 行数据")
        
        # 检查需要翻译的列是否存在
        existing_columns = [col for col in columns_to_translate if col in df.columns]
        missing_columns = [col for col in columns_to_translate if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  以下列不存在，将跳过: {', '.join(missing_columns)}")
        
        if not existing_columns:
            print("❌ 所有指定的翻译列都不存在")
            return False
        
        print(f"✅ 将翻译以下列: {', '.join(existing_columns)}")
        
        # 复制DataFrame
        df_translated = df.copy()
        
        # 翻译每一列
        for column in existing_columns:
            print(f"\n🌐 正在翻译列: {column}")
            
            translated_count = 0
            for idx, value in df[column].items():
                if pd.notna(value) and str(value).strip():
                    try:
                        original_text = str(value)
                        translated_text = translate_text(original_text)  # 移除translator参数
                        
                        # 只有翻译结果与原文不同时才更新
                        if translated_text != original_text:
                            df_translated.at[idx, column] = translated_text
                            translated_count += 1
                        
                        # 显示翻译进度（每10条显示一次）
                        if (idx + 1) % 10 == 0:
                            print(f"  📊 已处理 {idx + 1}/{len(df)} 条记录，已翻译 {translated_count} 条")
                    
                    except Exception as e:
                        print(f"  ⚠️  翻译第{idx}行失败: {e}")
                        continue
            
            print(f"✅ 完成翻译列 {column}，共翻译了 {translated_count} 条记录")
        
        # 保存翻译后的文件
        print(f"\n💾 保存翻译结果到: {output_file}")
        # 使用UTF-8 BOM编码，确保Excel能正确识别中文
        df_translated.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("✅ 翻译完成！")
        print("📝 文件使用UTF-8 BOM编码保存，Excel可以正确显示中文")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理CSV文件时出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CSV翻译工具 - 将CSV文件中的指定列翻译为中文",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 翻译默认列（analysis和follow_up_prompt）
  python csv_translator.py input.csv
  
  # 指定输出文件
  python csv_translator.py input.csv --output output_translated.csv
  
  # 指定要翻译的列
  python csv_translator.py input.csv --columns analysis follow_up_prompt description
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--input_csv",
        help="输入CSV文件路径"
    )
    
    # 可选参数
    parser.add_argument(
        "--output", "-o",
        help="输出CSV文件路径（默认：输入文件名_translated.csv）"
    )
    
    parser.add_argument(
        "--columns", "-c",
        nargs="+",
        default=["analysis", "follow_up_prompt"],
        help="需要翻译的列名（默认：analysis follow_up_prompt）"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    print("🚀 CSV翻译工具")
    print(f"📁 输入文件: {args.input_csv}")
    print(f"📁 输出文件: {args.output or '自动生成'}")
    print(f"🌐 翻译列: {', '.join(args.columns)}")
    print(f"🔧 翻译库状态: {'可用' if TRANSLATOR_AVAILABLE else '不可用'}")
    print("=" * 60)
    
    if not TRANSLATOR_AVAILABLE:
        print("❌ 请先安装requests库: pip install requests")
        return
    
    # 执行翻译
    success = translate_csv_file(
        input_file=args.input_csv,
        output_file=args.output,
        columns_to_translate=args.columns
    )
    
    if success:
        print(f"\n🎉 翻译任务完成！")
        print(f"📄 翻译结果已保存到: {args.output or (os.path.splitext(args.input_csv)[0] + '_translated.csv')}")
    else:
        print(f"\n❌ 翻译任务失败")

if __name__ == "__main__":
    main()