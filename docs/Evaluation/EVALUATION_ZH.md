# 图像数据集评测 (Image Dataset Evaluation)

所有数据集都需要先下载并转换为统一的JSONL格式，其中jsonl中每条数据包含以下标准字段：
- `id`: 数据样本的唯一标识符
- `image`: 图片路径列表（支持多图像），若没有则为空
- `video`：视频路径列表，若没有则为空
- `conversations`: 对话格式的问答内容，需包含问题选项和答案，如（"conversations": [{"from": "human", "value": "{question}\nSelect from the following choices. (A) .. A (B) .."},{"from": "gpt", "value": "A"}],）
- `task`: 任务类型（如Object_Localization, Depth, Count等）
- `input_type`: 输入类型（通常为"Image"）
- `output_type`: 输出类型（如"MCQ"表示多选题）
- `data_source`: 数据集来源

```bash
# 创建样本数据（可选，用于快速测试）
python dataset/create_json_sample.py --input_file dataset/ERQA_All_Data.jsonl --sample 30

python evaluate_img.py --data_path dataset/BLINK_All_Tasks.jsonl --max_workers 4 --image_base_path dataset --model gpt-4o-mini
```
## 1. BLINK数据集

```bash
# 下载BLINK数据集并转换为JSONL格式
python spagent/utils/download_blink.py
```

## 2. MindCube数据集
```bash
# 下载 MindCube 数据集并转换为 JSONL 格式
# 脚本会自动运行download_MindCube.sh脚本下载数据集并转换为统一的格式
python spagent/utils/download_mindcube.py

# 使用自定义参数
python spagent/utils/download_mindcube.py \
    --input dataset/mindcube/data/raw/MindCube_tinybench.jsonl \
    --output dataset/MindCube_data.jsonl \
    --image-prefix mindcube/data/

# 如果已经运行download_MindCube.sh脚本下载了数据，只想转换格式（跳过下载步骤）
python spagent/utils/download_mindcube.py --skip-download
```

**参数说明**:
- `--input, -i`: 输入的 MindCube JSONL 文件路径（默认: `dataset/mindcube/data/raw/MindCube_tinybench.jsonl`）
- `--output, -o`: 输出的 JSONL 文件路径（默认: `dataset/MindCube_data.jsonl`）
- `--image-prefix, -p`: 图片路径前缀（默认: `mindcube/data/`）
- `--skip-download`: 跳过下载步骤，直接转换已有数据

## 3. CVBench数据集
CVBench专注于计算机视觉的基础能力测试，包括深度估计、目标计数、空间关系等任务。

```bash
# 第一步：下载CVBench图片（需要先保存parquet文件到dataset目录）
# 数据集地址：https://huggingface.co/datasets/nyu-visionx/CV-Bench
python spagent/utils/cvbench_img.py --subset both --root dataset --out dataset/CVBench

# 第二步：转换为JSONL格式
python spagent/utils/download_cvbench.py
```

## 4. ERQA数据集
```bash
# 第一步，下载ERQA原始数据（先保存tfrecord数据到dataset文件夹）
# 数据集地址：https://github.com/embodiedreasoning/ERQA/blob/main/data/erqa.tfrecord
python  python spagent/utils/download_erqa.py
```

## 5. VSI-Bench数据集
```bash
# 下载VSI-Bench原始数据并转为jsonl格式。
# 数据集地址：https://huggingface.co/datasets/nyu-visionx/VSI-Bench
python spagent/utils/download_vsibench.py
```

## 6. VLM4D数据集
```bash
# 下载VLM4D原始数据并转为jsonl格式。
# 数据集地址：https://huggingface.co/datasets/shijiezhou/VLM4D
python spagent/utils/download_vlm4d.py
```

## 7. Omni-Perspective数据集

```bash
# 第一步：从 HuggingFace 下载 parquet 文件到本地目录
# 数据集地址：https://huggingface.co/datasets/Icey444/Omni-perspective
# 需要下载 val-*.parquet 文件到指定目录（默认: /home/ubuntu/Downloads）

# 第二步：转换为 JSONL 格式
python spagent/utils/download_Omni-Perspective.py

# 使用自定义参数
python spagent/utils/download_Omni-Perspective.py \
    --parquet_dir /path/to/parquet/files \
    --save_dir dataset \
    --pattern val-*.parquet
```

**参数说明**:
- `--parquet_dir`: parquet 文件所在目录（默认: `/home/ubuntu/Downloads`）
- `--save_dir`: 保存目录，将创建 `Omni_Perspective_images` 文件夹和 `Omni_Perspective_All.jsonl`（默认: `dataset`）
- `--pattern`: parquet 文件匹配模式（默认: `val-*.parquet`）

## 8. MMSI-Bench数据集

MMSI-Bench 是一个多模态空间推理基准数据集。

```bash
# 第一步：下载 MMSI_Bench.parquet 文件
# 数据集地址https://huggingface.co/datasets/RunsenXu/MMSI-Bench
# 需要将 parquet 文件保存到本地

# 第二步：转换为 JSONL 格式
python spagent/utils/download_mmsi.py

# 使用自定义参数
python spagent/utils/download_mmsi.py \
    --parquet_path /path/to/MMSI_Bench.parquet \
    --output_dir dataset \
    --image_folder_name MMSI_images
```

**参数说明**:
- `--parquet_path`: MMSI_Bench.parquet 文件路径（默认: `/home/ubuntu/datasets/spatial-reasoning/MMSI-Bench/MMSI_Bench.parquet`）
- `--output_dir`: 输出目录（默认: `dataset`）
- `--image_folder_name`: 图片文件夹名称（默认: `MMSI_images`）