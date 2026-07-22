"""
下载并转换 MMSI-Bench 数据集
将 parquet 格式转换为 BLINK 风格的 JSONL 格式

数据来源: HuggingFace RunsenXu/MMSI-Bench (1000 条样本，单个 test split)
https://huggingface.co/datasets/RunsenXu/MMSI-Bench

使用方法:
    # 自动从 HuggingFace 下载并转换（推荐，无需手动下载 parquet）
    python spagent/utils/download_mmsi.py

    # 使用已经手动下载好的 parquet 文件
    python spagent/utils/download_mmsi.py --parquet_path /path/to/MMSI_Bench.parquet
    python spagent/utils/download_mmsi.py --parquet_path /path/to/MMSI_Bench.parquet --output_dir custom_dataset
"""

import pandas as pd
import json
import os
import argparse
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm

HF_REPO = "RunsenXu/MMSI-Bench"


def _extract_image_bytes(image) -> bytes:
    """把 datasets 里的图片对象（PIL.Image / {"bytes": ...} / 原始 bytes）统一转换成 PNG bytes。"""
    if isinstance(image, dict) and image.get("bytes") is not None:
        return image["bytes"]
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if hasattr(image, "save"):  # PIL.Image
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"无法识别的图片类型: {type(image)}")


def _load_source_dataframe(parquet_path: str, hf_repo: str = HF_REPO, auto_download: bool = True) -> pd.DataFrame:
    """
    加载 MMSI-Bench 数据为 DataFrame。

    优先使用本地 parquet 文件（若存在）；否则（当 auto_download=True 时）
    通过 `datasets.load_dataset` 自动从 HuggingFace 拉取，无需手动下载 parquet。
    """
    if parquet_path and Path(parquet_path).exists():
        print(f"加载本地 parquet 文件: {parquet_path}")
        return pd.read_parquet(parquet_path)

    if not auto_download:
        raise FileNotFoundError(f"找不到文件: {parquet_path}")

    print(f"本地未找到 {parquet_path!r}，从 HuggingFace 自动下载 {hf_repo} ...")
    from datasets import load_dataset

    hf_ds = load_dataset(hf_repo)
    split_name = "test" if "test" in hf_ds else list(hf_ds.keys())[0]
    ds = hf_ds[split_name]
    print(f"✓ HuggingFace 数据集加载成功: split={split_name}, {len(ds)} 条样本")

    df = ds.to_pandas()
    # `images` 列经 to_pandas() 后一般是 list[{"bytes":..., "path":...}]，统一成 list[bytes]
    df["images"] = df["images"].apply(
        lambda imgs: [_extract_image_bytes(img) for img in imgs] if imgs is not None else []
    )
    return df


def parse_answer_from_question(question: str, answer: str) -> str:
    """
    从问题和答案中提取标准答案格式
    
    Args:
        question: 问题文本（可能包含选项）
        answer: 原始答案（可能是字母或完整文本）
    
    Returns:
        标准化的答案字母 (A/B/C/D)
    """
    # 如果答案已经是单个字母，直接返回
    if isinstance(answer, str) and len(answer) == 1 and answer.upper() in ['A', 'B', 'C', 'D']:
        return answer.upper()
    
    # 如果答案是 "(A)" 这种格式
    if isinstance(answer, str) and answer.startswith("(") and answer.endswith(")"):
        letter = answer[1:-1].strip()
        if letter.upper() in ['A', 'B', 'C', 'D']:
            return letter.upper()
    
    # 尝试从问题中的选项匹配答案
    if "Options:" in question:
        options_part = question.split("Options:")[-1].strip()
        # 解析选项: "A: xxx, B: xxx, C: xxx, D: xxx"
        for option in options_part.split(","):
            option = option.strip()
            if ":" in option:
                letter, text = option.split(":", 1)
                letter = letter.strip()
                text = text.strip()
                # 检查答案是否匹配这个选项的文本
                if answer.strip() == text or answer.strip() in text:
                    if letter.upper() in ['A', 'B', 'C', 'D']:
                        return letter.upper()
    
    # 默认返回原答案
    return str(answer)


def format_question_with_choices(question: str) -> str:
    """
    格式化问题，使选项更清晰
    
    Args:
        question: 原始问题文本
    
    Returns:
        格式化后的问题
    """
    if "Options:" in question:
        parts = question.split("Options:")
        question_text = parts[0].strip()
        options_text = parts[1].strip()
        
        # 重新格式化选项
        formatted_question = question_text + "\nSelect from the following choices.\n"
        
        # 解析并格式化每个选项
        for option in options_text.split(","):
            option = option.strip()
            if option:
                formatted_question += f"({option})\n"
        
        return formatted_question
    
    return question


def convert_mmsi_to_blink_format(
    parquet_path: str,
    output_dir: str = "dataset",
    image_folder_name: str = "MMSI_images",
    hf_repo: str = HF_REPO,
    auto_download: bool = True,
) -> int:
    """
    将 MMSI-Bench 转换为 BLINK 格式的 JSONL。

    Args:
        parquet_path: 本地 MMSI_Bench.parquet 文件路径（若不存在且 auto_download=True，
            则自动从 HuggingFace 下载）
        output_dir: 输出目录
        image_folder_name: 图片文件夹名称
        hf_repo: HuggingFace 数据集仓库名（默认: RunsenXu/MMSI-Bench）
        auto_download: 本地文件不存在时是否自动从 HuggingFace 下载

    Returns:
        转换的样本数量
    """
    df = _load_source_dataframe(parquet_path, hf_repo=hf_repo, auto_download=auto_download)
    print(f"✓ 数据集加载成功，共 {len(df)} 条样本")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建图片保存目录
    image_folder = output_path / image_folder_name
    image_folder.mkdir(parents=True, exist_ok=True)
    print(f"✓ 图片文件夹: {image_folder}")
    
    # 转换数据
    all_converted_data = []
    total_processed = 0
    total_images_saved = 0
    
    # 统计任务类型
    task_counts = {}
    
    print("\n开始转换数据...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理样本"):
        try:
            # 处理图片
            image_paths = []
            images_data = row['images']
            
            # images 是一个包含字节数据的数组
            if images_data is not None and len(images_data) > 0:
                for img_idx, img_bytes in enumerate(images_data):
                    try:
                        # 从字节数据创建 PIL 图片
                        img = Image.open(BytesIO(img_bytes))
                        
                        # 生成图片文件名
                        image_filename = f"mmsi_{idx:06d}_img{img_idx + 1}.png"
                        image_path = image_folder / image_filename
                        
                        # 保存图片
                        img.save(image_path)
                        
                        # 保存相对路径
                        image_paths.append(f"{image_folder_name}/{image_filename}")
                        total_images_saved += 1
                        
                    except Exception as e:
                        print(f"\n⚠ 样本 {idx} 图片 {img_idx} 保存失败: {e}")
                        continue
            
            # 获取问题类型作为任务
            question_type = row.get('question_type', 'Unknown')
            task_counts[question_type] = task_counts.get(question_type, 0) + 1
            
            # 格式化问题
            question_text = row['question']
            formatted_question = format_question_with_choices(question_text)
            
            # 解析答案
            answer = parse_answer_from_question(question_text, row['answer'])
            
            # 构建对话内容
            conversations = [
                {
                    "from": "human",
                    "value": formatted_question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
            
            # 构建 JSON 条目（BLINK 格式）
            json_entry = {
                "id": f"MMSI_{idx}",
                "image": image_paths,
                "video": [],
                "conversations": conversations,
                "task": question_type,
                "input_type": "image",
                "output_type": "MCQ",
                "data_source": "MMSI-Bench",
                "sub_task": "",
                "others": {
                    "original_answer": row['answer'],
                    "thought": row.get('thought', ''),
                    "original_id": int(row['id']) if 'id' in row else idx
                }
            }
            
            all_converted_data.append(json_entry)
            total_processed += 1
            
        except Exception as e:
            print(f"\n✗ 处理样本 {idx} 时出错: {e}")
            continue
    
    print(f"\n✓ 数据处理完成！")
    print(f"  - 成功处理: {total_processed} 条样本")
    print(f"  - 保存图片: {total_images_saved} 张")
    
    # 保存 JSONL 文件
    json_path = output_path / "MMSI_All_Tasks.jsonl"
    print(f"\n保存 JSONL 文件: {json_path}")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        for item in all_converted_data:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print(f"✓ JSONL 文件保存成功！")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print("任务类型统计:")
    print(f"{'='*60}")
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task:<45} {count:>4} 条")
    
    print(f"\n{'='*60}")
    print("输出文件:")
    print(f"{'='*60}")
    print(f"  JSONL: {json_path}")
    print(f"  图片目录: {image_folder}")
    print(f"  总样本数: {len(all_converted_data)}")
    print(f"  总图片数: {total_images_saved}")
    
    # 打印第一条数据作为示例
    if all_converted_data:
        print(f"\n{'='*60}")
        print("第一条数据示例:")
        print(f"{'='*60}")
        print(json.dumps(all_converted_data[0], ensure_ascii=False, indent=2))
    
    return total_processed


def main():
    parser = argparse.ArgumentParser(
        description="下载/转换 MMSI-Bench 为 BLINK 格式的 JSONL（默认自动从 HuggingFace 下载）"
    )
    parser.add_argument(
        '--parquet_path',
        type=str,
        default='datasets/spatial-reasoning/MMSI-Bench/MMSI_Bench.parquet',
        help='本地 MMSI_Bench.parquet 文件路径；不存在时自动从 HuggingFace 下载（除非指定 --no_auto_download）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset',
        help='输出目录（默认: dataset）'
    )
    parser.add_argument(
        '--image_folder_name',
        type=str,
        default='MMSI_images',
        help='图片文件夹名称（默认: MMSI_images）'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default=HF_REPO,
        help=f'HuggingFace 数据集仓库名（默认: {HF_REPO}）'
    )
    parser.add_argument(
        '--no_auto_download',
        action='store_true',
        help='禁用自动下载；本地 parquet 文件不存在时直接报错'
    )

    args = parser.parse_args()

    try:
        total = convert_mmsi_to_blink_format(
            parquet_path=args.parquet_path,
            output_dir=args.output_dir,
            image_folder_name=args.image_folder_name,
            hf_repo=args.hf_repo,
            auto_download=not args.no_auto_download,
        )
        print(f"\n🎉 转换成功！共处理 {total} 条样本")
        return 0
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
