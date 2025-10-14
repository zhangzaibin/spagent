#!/usr/bin/env python3
"""
下载并转换VLM4D数据集为JSONL格式
- 从本地路径复制视频文件
- 支持real与synthetic数据合并
- 将回答映射为选项字母（例如 "B"）
"""

import os
import json
import shutil
from collections import defaultdict

def download_vlm4d(test_mode=False, max_samples=5, start_index=0):
    """下载VLM4D数据集并转换为JSONL格式"""

    source_base = "/media/zzb/AI_save/zzb/spagent/dataset/VLM4D/shijiezhou/VLM4D"
    qa_folder = os.path.join(source_base, "QA")
    video_real = os.path.join(source_base, "videos_real")
    video_synth = os.path.join(source_base, "videos_synthetic")

    target_root = "dataset"
    target_video_folder = os.path.join(target_root, "VLM4D_videos")
    os.makedirs(target_video_folder, exist_ok=True)

    print(f"📂 源路径: {source_base}")
    print(f"📁 目标路径: {os.path.abspath(target_root)}")

    # 检查源路径是否存在
    if not os.path.exists(source_base):
        print(f"❌ 源路径不存在: {source_base}")
        return

    # 加载两个QA文件
    qa_files = {
        "real": os.path.join(qa_folder, "real_mc.json"),
        "synthetic": os.path.join(qa_folder, "synthetic_mc.json")
    }

    all_data = []
    for name, file_path in qa_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到 {file_path}，跳过")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"✅ 加载 {file_path} 共 {len(data)} 条")
            for item in data:
                item["split"] = name
            all_data.extend(data)

    if test_mode:
        all_data = all_data[start_index:start_index+max_samples]
        print(f"🧪 测试模式: 从索引 {start_index} 开始处理 {len(all_data)} 条数据")

    print(f"📊 总数据量: {len(all_data)} 条")

    # 复制视频文件
    copied_videos = set()
    failed_videos = []
    print("\n📹 开始复制视频文件...")

    for sample in all_data:
        video_url = sample["video"]
        # 从URL中提取相对路径
        if "videos_real" in video_url:
            relative_path = video_url.split("videos_real/")[-1]
            source_path = os.path.join(video_real, relative_path)
        elif "videos_synthetic" in video_url:
            relative_path = video_url.split("videos_synthetic/")[-1]
            source_path = os.path.join(video_synth, relative_path)
        else:
            print(f"⚠️ 无法识别视频路径: {video_url}")
            failed_videos.append(video_url)
            continue

        target_filename = f"{sample['split']}_{relative_path.replace('/', '_')}"
        target_path = os.path.join(target_video_folder, target_filename)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if os.path.exists(source_path):
            try:
                if not os.path.exists(target_path):
                    shutil.copy2(source_path, target_path)
                    print(f"  ✅ 复制: {relative_path}")
                copied_videos.add(target_filename)
            except Exception as e:
                print(f"  ❌ 复制失败: {relative_path} - {e}")
                failed_videos.append(video_url)
        else:
            print(f"  ❌ 源文件不存在: {source_path}")
            failed_videos.append(video_url)

    print(f"\n📊 视频复制完成，共复制 {len(copied_videos)} 个文件，失败 {len(failed_videos)} 个")

    # 转换为JSONL格式
    print("\n📝 开始转换为JSONL格式...")
    jsonl_path = os.path.join(target_root, "VLM4D.jsonl")
    converted = []

    for idx, sample in enumerate(all_data):
        try:
            q = sample.get("question", "")
            a_text = sample.get("answer", "")
            choices = sample.get("choices", {})
            video_url = sample.get("video", "")
            split = sample.get("split", "unknown")

            # 安全转换答案为字符串
            a_text_str = str(a_text).strip().lower()

            # 确定本地视频路径
            if "videos_real" in video_url:
                rel = video_url.split("videos_real/")[-1]
            elif "videos_synthetic" in video_url:
                rel = video_url.split("videos_synthetic/")[-1]
            else:
                rel = os.path.basename(video_url)

            video_filename = f"{split}_{rel.replace('/', '_')}"
            video_rel_path = f"VLM4D_videos/{video_filename}"

            # === 核心逻辑：将答案转换为选项字母 ===
            answer_letter = None
            for letter, text in choices.items():
                # 统一比较字符串形式
                if str(text).strip().lower() == a_text_str:
                    answer_letter = letter
                    break
            if answer_letter is None:
                answer_letter = "Unknown"

            # 构建 conversations
            question_text = q
            if choices:
                question_text += "\nSelect from the following choices:\n"
                for k, v in choices.items():
                    question_text += f"({k}) {v}\n"

            conversations = [
                {"from": "human", "value": question_text},
                {"from": "gpt", "value": answer_letter}
            ]

            converted.append({
                "id": sample.get("id", f"VLM4D_{idx}"),
                "image": [],
                "video": [video_rel_path],
                "conversations": conversations,
                "task": sample.get("question_type", "multiple-choice"),
                "input_type": "video",
                "output_type": "MCQ",
                "data_source": "VLM4D",
                "others": {},
                "subtask": split
            })
        except Exception as e:
            print(f"  ❌ 数据转换失败 [{idx}]: {e}")

    # 写入JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in converted:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n✅ JSONL文件已保存至: {os.path.abspath(jsonl_path)}")
    print(f"📈 共保存 {len(converted)} 条数据")

    if converted:
        print("\n📄 示例数据:")
        print(json.dumps(converted[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 示例运行：全量或测试模式
    # download_vlm4d(test_mode=True, max_samples=5)
    download_vlm4d(test_mode=False)
