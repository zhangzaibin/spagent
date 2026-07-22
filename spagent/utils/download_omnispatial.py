"""
下载并转换 OmniSpatial 数据集（test split）
将 HuggingFace 数据集转换为 BLINK 风格的 JSONL 格式

数据来源: nv-njb/OmniSpatial-Test（qizekun/OmniSpatial 官方 test split 的重新打包版，
可直接通过 datasets.load_dataset 一次性拉取，1533 条样本）
https://huggingface.co/datasets/nv-njb/OmniSpatial-Test
官方仓库: https://huggingface.co/datasets/qizekun/OmniSpatial

原始 schema: id, image, question, options (list[str]), answer (int, 0-based index
into options), task_type, sub_task_type

使用方法:
    python spagent/utils/download_omnispatial.py
    python spagent/utils/download_omnispatial.py --output_dir custom_dataset
"""

import json
import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

HF_REPO = "nv-njb/OmniSpatial-Test"


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


def format_question_with_options(question: str, options) -> str:
    """
    把 question + options 列表格式化成带 (A)(B)(C)... 选项的提示文本，
    与 MindCube / MMSI 的 MCQ 提问风格保持一致。
    """
    formatted = question.strip() + "\nSelect from the following choices.\n"
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        formatted += f"({letter}) {opt}\n"
    return formatted.rstrip("\n")


def convert_omnispatial_to_blink_format(
    output_dir: str = "dataset",
    image_folder_name: str = "OmniSpatial_images",
    hf_repo: str = HF_REPO,
    split: str = "test",
) -> int:
    """
    将 OmniSpatial 转换为 BLINK 格式的 JSONL。

    Args:
        output_dir: 输出目录
        image_folder_name: 图片文件夹名称
        hf_repo: HuggingFace 数据集仓库名（默认: nv-njb/OmniSpatial-Test）
        split: 数据集 split 名称（默认: test）

    Returns:
        转换的样本数量
    """
    print(f"加载 OmniSpatial 数据集: {hf_repo}")
    from datasets import load_dataset

    hf_ds = load_dataset(hf_repo)
    split_name = split if split in hf_ds else list(hf_ds.keys())[0]
    ds = hf_ds[split_name]
    print(f"✓ 数据集加载成功: split={split_name}, 共 {len(ds)} 条样本")

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
    for idx in tqdm(range(len(ds)), total=len(ds), desc="处理样本"):
        row = ds[idx]
        try:
            raw_id = row.get("id", idx)

            # 获取任务类型
            task_type = row.get("task_type", "Unknown")
            sub_task_type = row.get("sub_task_type", "")
            task_counts[task_type] = task_counts.get(task_type, 0) + 1

            # ⚠️ 原始 `id` 字段（"{image_number}_{question_number}"）在 4 个 task_type
            # 分类内各自独立从 0 开始编号，跨分类拼接后并不是全局唯一的（同一个 "0_0"
            # 会在 4 个不同 task_type 里各出现一次，对应完全不同的图片/问题）。
            # 这里用 task_type + 行下标 组合出一个全局唯一的 sample_id，避免：
            #   1) 保存图片时文件名冲突、互相覆盖
            #   2) quick_eval.py 里按 id 做 resume/缓存查找时把不同问题误判为同一条
            sample_id = f"{idx:04d}_{task_type}_{raw_id}"

            # 处理图片
            image_paths = []
            image_raw = row.get("image")
            if image_raw is not None:
                try:
                    img_bytes = _extract_image_bytes(image_raw)
                    img = Image.open(BytesIO(img_bytes))

                    image_filename = f"omnispatial_{sample_id}.png"
                    image_path = image_folder / image_filename
                    img.save(image_path)

                    image_paths.append(f"{image_folder_name}/{image_filename}")
                    total_images_saved += 1
                except Exception as e:
                    print(f"\n⚠ 样本 {sample_id} 图片保存失败: {e}")

            # 格式化问题（question + options → 带 (A)(B)(C)... 的提示文本）
            question_text = row.get("question", "")
            options = row.get("options", []) or []
            formatted_question = format_question_with_options(question_text, options)

            # 解析答案：options 下标 → 字母
            answer_idx = row.get("answer")
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                answer_letter = chr(ord("A") + answer_idx)
            else:
                answer_letter = str(answer_idx)

            # 构建对话内容
            conversations = [
                {
                    "from": "human",
                    "value": formatted_question
                },
                {
                    "from": "gpt",
                    "value": answer_letter
                }
            ]

            # 构建 JSON 条目（BLINK 格式）
            json_entry = {
                "id": f"OmniSpatial_{sample_id}",
                "image": image_paths,
                "video": [],
                "conversations": conversations,
                "task": task_type,
                "input_type": "image",
                "output_type": "MCQ",
                "data_source": "OmniSpatial",
                "sub_task": sub_task_type,
                "others": {
                    "original_answer_index": answer_idx,
                    "options": options,
                    "original_id": raw_id,
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
    json_path = output_path / "OmniSpatial_All.jsonl"
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
        print(f"  {task:<30} {count:>4} 条")

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
        description="下载并转换 OmniSpatial (test split) 为 BLINK 格式的 JSONL"
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
        default='OmniSpatial_images',
        help='图片文件夹名称（默认: OmniSpatial_images）'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default=HF_REPO,
        help=f'HuggingFace 数据集仓库名（默认: {HF_REPO}，可直接 load_dataset）'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='数据集 split 名称（默认: test）'
    )

    args = parser.parse_args()

    try:
        total = convert_omnispatial_to_blink_format(
            output_dir=args.output_dir,
            image_folder_name=args.image_folder_name,
            hf_repo=args.hf_repo,
            split=args.split,
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
