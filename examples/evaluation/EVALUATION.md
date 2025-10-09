# 图像数据集评测 (Image Dataset Evaluation)

所有数据集都需要先下载并转换为统一的JSONL格式，其中每条数据包含以下标准字段：
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

## 2. CVBench数据集
CVBench专注于计算机视觉的基础能力测试，包括深度估计、目标计数、空间关系等任务。

```bash
# 第一步：下载CVBench图片（需要先保存parquet文件到dataset目录）
# 数据集地址：https://huggingface.co/datasets/nyu-visionx/CV-Bench
python spagent/utils/cvbench_img.py --subset both --root dataset --out dataset/CVBench

# 第二步：转换为JSONL格式
python spagent/utils/download_cvbench.py
```

## 3. ERQA數據集
```bash
# 第一步，下载ERQA原始数据（先保存tfrecord数据到dataset文件夹）
# 数据集地址：https://github.com/embodiedreasoning/ERQA/blob/main/data/erqa.tfrecord
python  python spagent/utils/download_erqa.py
```

## 3. VSI-Bench数据集
```bash
# 下载VSI-Bench原始数据并转为jsonl格式。
# 数据集地址：https://huggingface.co/datasets/nyu-visionx/VSI-Bench
python spagent/utils/download_vsibench.py
```