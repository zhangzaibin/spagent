# Image Dataset Evaluation

> **中文版本**: [中文文档](EVALUATION_ZH.md) | **English Version**: This document

All datasets need to be downloaded and converted to a unified JSONL format first. Each data entry in the JSONL file contains the following standard fields:
- `id`: Unique identifier for the data sample
- `image`: List of image paths (supports multiple images), empty if none
- `video`: List of video paths, empty if none
- `conversations`: Q&A content in conversation format, must include question options and answers, e.g., `"conversations": [{"from": "human", "value": "{question}\nSelect from the following choices. (A) .. A (B) .."},{"from": "gpt", "value": "A"}]`
- `task`: Task type (e.g., Object_Localization, Depth, Count, etc.)
- `input_type`: Input type (usually "Image")
- `output_type`: Output type (e.g., "MCQ" for multiple choice questions)
- `data_source`: Dataset source

```bash
# Create sample data (optional, for quick testing)
python dataset/create_json_sample.py --input_file dataset/ERQA_All_Data.jsonl --sample 30

python evaluate_img.py --data_path dataset/BLINK_All_Tasks.jsonl --max_workers 4 --image_base_path dataset --model gpt-4o-mini
```

## 1. BLINK Dataset

```bash
# Download BLINK dataset and convert to JSONL format
python spagent/utils/download_blink.py
```

## 2. MindCube Dataset

```bash
# Download MindCube dataset and convert to JSONL format
# The script will automatically run download_MindCube.sh to download the dataset and convert to unified format
python spagent/utils/download_mindcube.py

# Use custom parameters
python spagent/utils/download_mindcube.py \
    --input dataset/mindcube/data/raw/MindCube_tinybench.jsonl \
    --output dataset/MindCube_data.jsonl \
    --image-prefix mindcube/data/

# If you have already run download_MindCube.sh to download the data, only convert format (skip download step)
python spagent/utils/download_mindcube.py --skip-download
```

**Parameter Description**:
- `--input, -i`: Input MindCube JSONL file path (default: `dataset/mindcube/data/raw/MindCube_tinybench.jsonl`)
- `--output, -o`: Output JSONL file path (default: `dataset/MindCube_data.jsonl`)
- `--image-prefix, -p`: Image path prefix (default: `mindcube/data/`)
- `--skip-download`: Skip download step, directly convert existing data

## 3. CVBench Dataset

CVBench focuses on fundamental computer vision capabilities testing, including depth estimation, object counting, spatial relationships, and other tasks.

```bash
# Step 1: Download CVBench images (need to save parquet files to dataset directory first)
# Dataset URL: https://huggingface.co/datasets/nyu-visionx/CV-Bench
python spagent/utils/cvbench_img.py --subset both --root dataset --out dataset/CVBench

# Step 2: Convert to JSONL format
python spagent/utils/download_cvbench.py
```

## 4. ERQA Dataset

```bash
# Step 1: Download ERQA raw data (save tfrecord data to dataset folder first)
# Dataset URL: https://github.com/embodiedreasoning/ERQA/blob/main/data/erqa.tfrecord
python spagent/utils/download_erqa.py
```

## 5. VSI-Bench Dataset

```bash
# Download VSI-Bench raw data and convert to jsonl format.
# Dataset URL: https://huggingface.co/datasets/nyu-visionx/VSI-Bench
python spagent/utils/download_vsibench.py
```

## 6. VLM4D Dataset

```bash
# Download VLM4D raw data and convert to jsonl format.
# Dataset URL: https://huggingface.co/datasets/shijiezhou/VLM4D
python spagent/utils/download_vlm4d.py
```

## 7. Omni-Perspective Dataset

```bash
# Step 1: Download parquet files from HuggingFace to local directory
# Dataset URL: https://huggingface.co/datasets/Icey444/Omni-perspective
# Need to download val-*.parquet files to specified directory (default: /home/ubuntu/Downloads)

# Step 2: Convert to JSONL format
python spagent/utils/download_Omni-Perspective.py

# Use custom parameters
python spagent/utils/download_Omni-Perspective.py \
    --parquet_dir /path/to/parquet/files \
    --save_dir dataset \
    --pattern val-*.parquet
```

**Parameter Description**:
- `--parquet_dir`: Directory containing parquet files (default: `/home/ubuntu/Downloads`)
- `--save_dir`: Save directory, will create `Omni_Perspective_images` folder and `Omni_Perspective_All.jsonl` (default: `dataset`)
- `--pattern`: Parquet file matching pattern (default: `val-*.parquet`)

## 8. MMSI-Bench Dataset

MMSI-Bench is a multimodal spatial reasoning benchmark dataset.

```bash
# Step 1: Download MMSI_Bench.parquet file
# Dataset URL: https://huggingface.co/datasets/RunsenXu/MMSI-Bench
# Need to save the parquet file to local

# Step 2: Convert to JSONL format
python spagent/utils/download_mmsi.py

# Use custom parameters
python spagent/utils/download_mmsi.py \
    --parquet_path /path/to/MMSI_Bench.parquet \
    --output_dir dataset \
    --image_folder_name MMSI_images
```

**Parameter Description**:
- `--parquet_path`: MMSI_Bench.parquet file path (default: `/home/ubuntu/datasets/spatial-reasoning/MMSI-Bench/MMSI_Bench.parquet`)
- `--output_dir`: Output directory (default: `dataset`)
- `--image_folder_name`: Image folder name (default: `MMSI_images`)
