#!/usr/bin/env python3
"""
下载VSI-Bench数据集并转换为JSONL格式

HuggingFace 仓库 nyu-visionx/VSI-Bench 中视频以三个 zip 包存储：
  arkitscenes.zip / scannet.zip / scannetpp.zip

本脚本：
  1. 从 HF 下载这三个 zip（已存在则跳过）
  2. 解压到 dataset/VSI_videos/
  3. 把元数据转换为 JSONL 格式
"""

from datasets import load_dataset
import json
import os
import shutil
import zipfile
from collections import defaultdict

HF_REPO = "nyu-visionx/VSI-Bench"
# 视频来源：仓库中的 zip 包名称
VIDEO_ZIPS = ["arkitscenes.zip", "scannet.zip", "scannetpp.zip"]


def _download_and_extract_zips(target_video_folder: str, zip_cache_dir: str) -> bool:
    """
    从 HuggingFace 仓库下载三个视频 zip 包并解压到 target_video_folder。
    zip_cache_dir: zip 文件本地缓存目录。
    返回是否全部成功。
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(zip_cache_dir, exist_ok=True)
    os.makedirs(target_video_folder, exist_ok=True)

    all_ok = True
    for zip_name in VIDEO_ZIPS:
        dataset_name = zip_name.replace(".zip", "")
        # 检查是否已经解压过（用一个 .done 标记文件）
        done_marker = os.path.join(target_video_folder, f".{dataset_name}_extracted")
        if os.path.exists(done_marker):
            print(f"  ⏭️  {zip_name} 已解压，跳过")
            continue

        zip_local = os.path.join(zip_cache_dir, zip_name)

        # ── 下载 zip ──────────────────────────────────────────────────────
        if not os.path.exists(zip_local):
            print(f"  ⬇️  下载 {zip_name} ...", flush=True)
            try:
                downloaded = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=zip_name,
                    repo_type="dataset",
                    local_dir=zip_cache_dir,
                )
                if os.path.abspath(downloaded) != os.path.abspath(zip_local):
                    shutil.move(downloaded, zip_local)
                print(f"  ✅ 下载完成: {zip_name} ({os.path.getsize(zip_local)/1024/1024:.1f} MB)")
            except Exception as e:
                print(f"  ❌ 下载失败: {zip_name}: {e}")
                all_ok = False
                continue
        else:
            print(f"  ⏭️  {zip_name} 已缓存，直接解压")

        # ── 解压 zip ──────────────────────────────────────────────────────
        print(f"  📦 解压 {zip_name} → {target_video_folder} ...", flush=True)
        try:
            with zipfile.ZipFile(zip_local, 'r') as zf:
                members = zf.namelist()
                print(f"     包含 {len(members)} 个文件")
                for member in members:
                    # zip 内结构可能是 scene.mp4 或 subdir/scene.mp4
                    basename = os.path.basename(member)
                    if not basename.endswith(".mp4"):
                        continue
                    # 目标路径：{dataset_name}_{scene_name}.mp4
                    target_name = f"{dataset_name}_{basename}"
                    target_path = os.path.join(target_video_folder, target_name)
                    if not os.path.exists(target_path):
                        with zf.open(member) as src, open(target_path, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
            # 写入完成标记
            open(done_marker, 'w').close()
            print(f"  ✅ 解压完成: {zip_name}")
        except Exception as e:
            print(f"  ❌ 解压失败: {zip_name}: {e}")
            all_ok = False

    return all_ok


def download_vsibench(test_mode=False, max_samples=5, start_index=0):
    """下载VSI-Bench数据集并转换为JSONL格式"""

    print(f"开始处理VSI-Bench数据集... {'(测试模式，从索引' + str(start_index) + '开始处理' + str(max_samples) + '条数据)' if test_mode else ''}")

    source_video_base = "dataset/VSI-Bench"   # 保留旧路径（本地手动复制仍可用）
    target_video_folder = "dataset/VSI_videos"
    zip_cache_dir = "dataset/VSI-Bench-zips"
    dataset_folder = "dataset"

    print(f"目标视频文件夹: {os.path.abspath(target_video_folder)}")

    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(target_video_folder, exist_ok=True)
    os.makedirs(source_video_base, exist_ok=True)

    # ── 加载元数据 ────────────────────────────────────────────────────────────
    try:
        parquet_path = os.path.join(source_video_base, "test-00000-of-00001.parquet")
        if os.path.exists(parquet_path):
            print(f"📂 从本地加载数据集: {parquet_path}")
            ds = load_dataset("parquet", data_files={"test": parquet_path})
        else:
            print(f"⚠️  本地parquet文件不存在，从Hub加载...")
            ds = load_dataset(HF_REPO)
        test_data = ds['test']
        print(f"✅ 数据集加载成功！数据量: {len(test_data)}")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return

    if test_mode:
        end_index = min(start_index + max_samples, len(test_data))
        process_data = [test_data[i] for i in range(start_index, end_index)]
    else:
        process_data = test_data

    # ── 统计需要的视频文件 ────────────────────────────────────────────────────
    video_files_needed = set()
    for sample in process_data:
        video_files_needed.add(f"{sample['dataset']}/{sample['scene_name']}.mp4")

    print(f"📊 需要的视频文件数量: {len(video_files_needed)}")

    # ── 下载并解压 zip 包（主要下载路径） ─────────────────────────────────────
    print("\n📹 从 HuggingFace 下载视频 zip 包...")
    _download_and_extract_zips(target_video_folder, zip_cache_dir)

    # ── 从旧的本地路径补充复制（兜底） ────────────────────────────────────────
    for video_rel in sorted(video_files_needed):
        target_path = os.path.join(target_video_folder, video_rel.replace('/', '_'))
        if os.path.exists(target_path):
            continue
        source_path = os.path.join(source_video_base, video_rel)
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, target_path)
            except Exception:
                pass

    # ── 统计实际可用的视频 ────────────────────────────────────────────────────
    copied_videos = set()
    failed_videos = []
    for video_rel in video_files_needed:
        target_path = os.path.join(target_video_folder, video_rel.replace('/', '_'))
        if os.path.exists(target_path):
            copied_videos.add(video_rel)
        else:
            failed_videos.append(video_rel)

    print(f"\n📊 视频获取结果:")
    print(f"  成功: {len(copied_videos)} 个")
    print(f"  失败: {len(failed_videos)} 个")
    if failed_videos:
        print(f"  失败列表: {failed_videos[:5]}{'...' if len(failed_videos) > 5 else ''}")
    
    # 转换为JSONL格式
    print("\n📝 开始转换为JSONL格式...")
    all_converted_data = []
    total_processed = 0
    skipped_no_video = 0
    
    # 统计各类信息
    dataset_stats = defaultdict(int)
    question_type_stats = defaultdict(int)
    
    for idx, sample in enumerate(process_data):
        try:
            dataset_name = sample['dataset']
            scene_name = sample['scene_name']
            video_path = f"{dataset_name}/{scene_name}.mp4"
            
            # 检查视频是否成功复制
            if video_path not in copied_videos:
                skipped_no_video += 1
                continue
            
            # 构建视频路径（相对于dataset目录）
            video_filename = video_path.replace('/', '_')
            video_relative_path = f"VSI_videos/{video_filename}"
            
            # 构建对话内容
            conversations = []
            
            # 获取基础信息
            question = sample.get('question', '')
            ground_truth = sample.get('ground_truth', '')
            options = sample.get('options')
            
            # 判断问题类型并构建相应的问题文本
            question_text = question
            output_type = "text"
            answer = str(ground_truth)
            
            # 判断是否为MCQ类型：ground_truth是字母且options不为None
            is_mcq = (isinstance(ground_truth, str) and 
                     len(ground_truth) == 1 and
                     ground_truth.isalpha() and 
                     options is not None)
            
            # 判断是否为Number类型：ground_truth是数字(或数字字符串)且options为None
            is_number = False
            if options is None:
                try:
                    # 尝试将ground_truth转换为数字
                    float(ground_truth)
                    is_number = True
                except (ValueError, TypeError):
                    is_number = False
            
            if is_mcq:
                # MCQ类型：添加选项信息
                output_type = "MCQ"
                if options and len(options) > 0:
                    question_text += "\nSelect from the following choices.\n"
                    for i, choice in enumerate(options):
                        question_text += f"({chr(65+i)}) {choice}\n"
                # 答案保持为字母格式
                answer = str(ground_truth)
                
            elif is_number:
                # Number类型：直接使用数字答案
                output_type = "Number"
                answer = str(ground_truth)
            else:
                # 其他类型：保持原格式
                output_type = "text"
                answer = str(ground_truth)
            
            # 添加人类问题
            if question_text:
                conversations.append({
                    "from": "human",
                    "value": question_text
                })
            
            # 添加答案
            if ground_truth is not None:
                conversations.append({
                    "from": "gpt",
                    "value": answer
                })
            
            # 构建JSON条目
            json_entry = {
                "id": f"VSIBench_{sample.get('id', idx)}",
                "image": [],  # VSI-Bench是视频数据集，没有静态图像
                "video": [video_relative_path],  # 视频路径
                "conversations": conversations,
                "task": sample.get('question_type', 'unknown'),
                "input_type": "video",
                "output_type": output_type,
                "data_source": "VSI-Bench",
                "others": {},
                "subtask": ""
            }
            
            all_converted_data.append(json_entry)
            total_processed += 1
            
            # 统计信息
            dataset_stats[dataset_name] += 1
            question_type_stats[sample.get('question_type', 'unknown')] += 1
            
            if total_processed % 500 == 0:
                print(f"  已处理 {total_processed} 条数据...")
                
        except Exception as e:
            print(f"  处理数据 {idx} 时出错: {e}")
            continue
    
    print(f"\n📊 数据处理完成!")
    print(f"  总原始数据: {len(process_data)} 条")
    print(f"  成功处理: {total_processed} 条")
    print(f"  跳过(无视频): {skipped_no_video} 条")
    
    # 保存JSONL文件
    json_filename = 'VSI_Bench_test.jsonl' if test_mode else 'VSI_Bench.jsonl'
    json_path = f'dataset/{json_filename}'
    print(f"\n💾 保存JSONL文件到: {os.path.abspath(json_path)}")
    
    if all_converted_data:
        with open(json_path, 'w', encoding='utf-8') as f:
            for item in all_converted_data:
                json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
        print(f"✅ JSONL文件保存成功!")
    else:
        print("⚠️  警告: 没有数据需要保存!")
    
    # 输出统计信息
    print(f"\n📈 数据统计:")
    print(f"  总数据量: {len(all_converted_data)} 条")
    print(f"  JSONL文件: {json_path}")
    print(f"  视频文件夹: {target_video_folder}")
    
    print(f"\n📊 数据来源分布:")
    for dataset, count in sorted(dataset_stats.items()):
        print(f"  {dataset}: {count} 条")
    
    print(f"\n📊 问题类型分布:")
    for question_type, count in sorted(question_type_stats.items()):
        print(f"  {question_type}: {count} 条")
    
    # 打印第一条数据作为示例
    if all_converted_data:
        print(f"\n📄 第一条数据示例:")
        print(json.dumps(all_converted_data[0], ensure_ascii=False, indent=2))
    
    print(f"\n🎉 VSI-Bench数据集处理完成!")


if __name__ == "__main__":
    # download_vsibench(test_mode=True, max_samples=5, start_index=956)
    download_vsibench(test_mode=False)