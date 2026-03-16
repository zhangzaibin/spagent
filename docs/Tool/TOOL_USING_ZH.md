# External Experts Module

External Experts 模块包含了专门用于空间智能任务的专业模型，包括深度估计、目标检测、分割、3D重建、视频生成等功能。工具分为两类：本地 server/client 架构和云端 API 直调（Veo、Sora），均支持独立部署和调用。

## 📁 模块结构

```
external_experts/
├── __init__.py                     # 模块初始化
├── checkpoints/                    # 所有模型权重文件
│   └──depth_anything
│   └──grounding_dino
│   └──pi3
│   └──pi3x
│   └──sam2
│   └──vggt
├── GroundingDINO/                  # 开放词汇目标检测
├── SAM2/                          # 图像和视频分割
├── Depth_AnythingV2/              # 深度估计
├── Pi3/                           # 3D重建 (Pi3 & Pi3X)
├── VGGT/                          # 多视角3D重建与相机位姿估计
├── mapanything/                   # 基于深度估计的稠密3D重建
├── moondream/                     # 视觉语言模型
├── Veo/                           # Google Veo 视频生成（API 直调，无需本地服务器）
├── Sora/                          # OpenAI Sora 视频生成（API 直调，无需本地服务器）
└── supervision/                   # YOLO目标检测和标注工具
```

## 🛠️ 工具概览

| 工具名称 | Tool Class | 功能 | 主要用途 | 部署方式 | 主要参数 |
|---------|------------|------|----------|----------|----------|
| **Depth AnythingV2** | `DepthEstimationTool` | 深度估计 | 单目深度估计，分析图像中的3D深度关系 | 本地服务器（20019） | `image_path` |
| **SAM2** | `SegmentationTool` | 图像/视频分割 | 高精度分割任务，精确分割图像中的对象 | 本地服务器（20020） | `image_path`, `point_coords`(可选), `point_labels`(可选), `box`(可选) |
| **GroundingDINO** | `ObjectDetectionTool` | 开放词汇目标检测 | 基于文本描述检测任意物体 | 本地服务器（20022） | `image_path`, `text_prompt`, `box_threshold`, `text_threshold` |
| **Moondream** | `MoondreamTool` | 视觉语言模型 | 图像理解和问答，基于图像内容回答自然语言问题 | 本地服务器（20024） | `image_path`, `task`, `object_name` |
| **Pi3** | `Pi3Tool` | 3D重建 | 从图像生成3D点云和多视角渲染图 | 本地服务器（20030） | `image_path`, `azimuth_angle`, `elevation_angle` |
| **Pi3X** | `Pi3XTool` | 3D重建（增强版） | Pi3升级版，更平滑点云、近似度量尺度、可选多模态条件注入 | 本地服务器（20031） | `image_path`, `azimuth_angle`, `elevation_angle` |
| **VGGT** | `VGGTTool` | 多视角3D重建与相机位姿估计 | 从多张图像或视频帧重建3D点云并估计相机位姿 | 20032 | `image_paths`, `azimuth_angle`, `elevation_angle`, `rotation_reference_camera`, `camera_view` |
| **MapAnything** | `MapAnythingTool` | 基于深度估计的稠密3D重建 | 利用深度图和相机位姿从多张图像重建稠密3D点云 | 20033 | `image_paths`, `azimuth_angle`, `elevation_angle`, `conf_percentile`, `apply_mask` |
| **Supervision** | `SupervisionTool` | 目标检测标注 | YOLO模型和可视化工具，通用目标检测和分割 | 本地 | `image_path`, `task` ("image_det" 或 "image_seg") |
| **YOLO-E** | `YOLOETool` | YOLO-E检测 | 高精度检测，支持自定义类别 | 本地 | `image_path`, `task`, `class_names` |
| **Veo** | `VeoTool` | 视频生成 | 通过 Google Veo（Gemini API）实现文生视频和图生视频 | API 直调（无需服务器） | `prompt`, `image_path`(可选), `duration`, `aspect_ratio` |
| **Sora** | `SoraTool` | 视频生成 | 通过 OpenAI Sora 实现文生视频和图生视频 | API 直调（无需服务器） | `prompt`, `image_path`(可选), `duration`, `resolution`, `aspect_ratio` |

**使用示例**:
- 详细使用示例请参考：[Advanced Examples](../Examples/ADVANCED_EXAMPLES.md)
- 快速入门指南请参考：[Quick Start Guide](../../readme.md#-quick-start)

---

## 📋 详细工具介绍

### 1. Depth AnythingV2 - 深度估计

**功能**: 单目图像深度估计

**特点**:
- 三种模型规格可选
- 高质量深度图生成
- 支持多种输入格式

**文件结构**:
```
Depth_AnythingV2/
├── depth_server.py
├── depth_client.py
├── mock_depth_service.py
└── depth_anything_v2/
```

**模型规格**:
| 模型 | 骨干网络 | 参数量 | 文件大小 | 推理速度 | 精度 |
|------|----------|--------|----------|----------|------|
| Small | ViT-S | ~25M | ~100MB | 快 | 良好 |
| Base | ViT-B | ~97M | ~390MB | 中等 | 高 |
| Large | ViT-L | ~335M | ~1.3GB | 慢 | 很高 |

**权重下载**:
```bash
cd checkpoints/
# Small模型 (~25MB, 最快)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
# Base模型 (~100MB, 平衡) - 推荐
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
# Large模型 (~350MB, 最高质量)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

**资源链接**:
- [官方仓库](https://github.com/DepthAnything/Depth-Anything-V2)
- [论文](https://arxiv.org/abs/2406.09414)

---


### 2. SAM2 - 图像和视频分割

**功能**: 高精度的图像和视频分割模型

**特点**:
- 支持图像和视频分割
- 多种模型规格可选
- 高精度分割效果

**文件结构**:
```
SAM2/
├── sam2_server.py
└── sam2_client.py
```

**模型规格**:
| 模型 | 参数量 | 文件大小 | 用途 |
|------|--------|----------|------|
| Hiera Large | ~224M | ~900MB | 高精度 |
| Hiera Base+ | ~80M | ~320MB | 平衡性能 |
| Hiera Small | ~46M | ~185MB | 快速推理 |

**权重下载**:
#### 使用官方脚本（推荐）
```bash
cd checkpoints/
# 推荐使用官方脚本
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh
```

#### 手动下载
```bash
cd checkpoints/

# SAM2.1 Hiera Large (推荐)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# SAM2.1 Hiera Base+ 
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# SAM2.1 Hiera Small
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

**资源链接**:
- [官方仓库](https://github.com/facebookresearch/sam2)
- [论文](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)

---

### 3. GroundingDINO - 开放词汇目标检测

**功能**: 基于自然语言描述检测图像中的目标物体

**特点**:
- 支持开放词汇检测，无需预定义类别
- 基于Swin-B骨干网络
- 可通过文本描述检测任意物体

**文件结构**:
```
GroundingDINO/
├── grounding_dino_server.py
├── grounding_dino_client.py
└── configs/
    └── GroundingDINO_SwinB_cfg.py
```

**安装**:
```bash
pip install groundingdino_py
```

**权重下载**:
```bash
cd checkpoints/
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

**资源链接**:
- [官方仓库](https://github.com/IDEA-Research/GroundingDINO)
- [论文](https://arxiv.org/abs/2303.05499)

---

### 4. Moondream - 视觉语言模型

**功能**: 视觉语言理解和图像问答

**特点**:
- 图像理解能力
- 自然语言交互
- API接口支持

**文件结构**:
```
moondream/
├── md_server.py          # 服务器端
├── md_client.py          # 客户端
├── md_local.py          # 本地部署
├── __init__.py
└── __pycache__/
```

**安装**:
```bash
pip install moondream
```

**环境配置**:
```bash
export MOONDREAM_API_KEY="your_api_key"
```

**资源链接**:
- [官方网站](https://moondream.ai/)
- [API文档](https://docs.moondream.ai/)

---

### 5. Pi3 - 3D重建服务

**功能**: 基于Pi3模型的3D重建，从图像生成3D点云

**特点**:
- 高质量3D重建
- 支持PLY格式输出
- 可视化支持

**文件结构**:
```
Pi3/
├── pi3/                  # 运行代码
├── example.py            # 原始Pi3运行代码
├── pi3_server.py         # Flask服务器
└── pi3_client.py         # 客户端
```

**环境要求**:
- torch==2.5.1
- torchvision==0.20.1
- numpy==1.26.4

**使用方法**:
```bash
# 可视化生成的PLY文件
python spagent/utils/ply_to_html_viewer.py xxx.ply --output xxx.html --max_points 100000
```

**权重下载**:
```bash
cd checkpoints/pi3
wget https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
```

---

### 5.1 Pi3X - 增强版3D重建服务

**功能**: 基于Pi3X模型的增强版3D重建（Pi3的升级版本）

**特点**:
- 更平滑的点云重建（ConvHead替代LinearPts3d，消除网格伪影）
- 近似度量尺度重建（Metric Scale）
- 可选多模态条件注入（相机位姿、内参、深度）
- 更可靠的连续置信度评分
- 与Pi3完全兼容的API接口（相同的输入/输出格式）

**文件结构**:
```
Pi3/
├── pi3/
│   └── models/
│       ├── pi3.py            # 原始Pi3模型
│       ├── pi3x.py           # Pi3X模型（增强版）
│       └── layers/
│           ├── conv_head.py   # 卷积上采样Head（Pi3X）
│           └── prope.py       # PRoPE位置编码（Pi3X）
├── pi3_server.py              # Pi3 Flask服务器
├── pi3_client.py              # Pi3客户端
├── pi3x_server.py             # Pi3X Flask服务器
└── pi3x_client.py             # Pi3X客户端
```

**权重下载**:
```bash
mkdir -p checkpoints/pi3x
cd checkpoints/pi3x
wget https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors
```

**资源链接**:
- [官方仓库](https://github.com/yyfz/Pi3)
- [Pi3X HuggingFace权重](https://huggingface.co/yyfz233/Pi3X)

---

### 6. VGGT - 多视角3D重建与相机位姿估计

**功能**: 利用VGGT-1B模型从多张图像重建3D点云并估计相机外参/内参

**特点**:
- 支持多视角图像输入（图像列表或视频帧）
- 输出稠密3D点云（PLY格式）及多视角渲染图
- 相机位姿估计（外参矩阵与内参矩阵）
- 基于置信度的点云过滤与马氏距离离群点去除
- 支持自定义观察角度（方位角与仰角）

**文件结构**:
```
VGGT/
├── vggt_server.py        # Flask服务器
├── vggt_client.py        # 客户端
└── vggt/                 # VGGT模型代码
```

**权重下载**:

首次启动时自动从HuggingFace下载：
```bash
# 自动下载（默认）
# python vggt_server.py  →  自动下载 facebook/VGGT-1B

# 或手动下载后指定路径
huggingface-cli download facebook/VGGT-1B --local-dir checkpoints/vggt
```

**资源链接**:
- [官方仓库](https://github.com/facebookresearch/vggt)
- [HuggingFace 模型](https://huggingface.co/facebook/VGGT-1B)

---

### 7. MapAnything - 基于深度估计的稠密3D重建

**功能**: 利用预测深度图和相机位姿从多张图像重建稠密3D点云

**特点**:
- 支持多视角图像或视频帧的稠密3D重建
- 内置边缘过滤与置信度点云掩码
- 输出稠密3D点云（PLY格式）及多视角渲染图
- 接口与Pi3兼容，方便对比实验
- 支持自定义观察角度（方位角与仰角）

**文件结构**:
```
mapanything/
├── mapanything_server.py   # Flask服务器
├── mapanything_client.py   # 客户端
└── mapanything/            # MapAnything模型代码
```

**权重下载**:

首次启动时自动从HuggingFace下载：
```bash
# 自动下载（默认）
# python mapanything_server.py  →  自动下载 facebook/map-anything

# 或手动预下载
huggingface-cli download facebook/map-anything --local-dir ~/.cache/huggingface/hub/models--facebook--map-anything
```

**资源链接**:
- [HuggingFace 模型](https://huggingface.co/facebook/map-anything)

---

### 8. Supervision - 目标检测和标注工具

**功能**: YOLO目标检测和可视化标注工具

**特点**:
- 集成多种YOLO模型
- 丰富的可视化工具
- 标注和后处理功能

**文件结构**:
```
supervision/
├── __init__.py
├── supervision_server.py
├── supervision_client.py
├── sv_yoloe_server.py
├── sv_yoloe_client.py
├── annotator.py
├── yoloe_annotator.py
├── yoloe_test.py 
├── download_weights.py
└── mock_supervision_service.py
```

**安装**:
```bash
pip install supervision
```

**可用模型**:
| 模型文件 | 功能 | 用途 |
|----------|------|------|
| yoloe-v8l-seg.pt | YOLOE v8 Large 分割 | 高精度目标检测和分割 |
| yoloe-v8l-seg-pf.pt | YOLOE v8 Large 分割(优化版) | 性能优化的分割模型 |

**权重下载**:
```bash
python download_weights.py
```

**资源链接**:
- [官方仓库](https://github.com/roboflow/supervision)
- [文档](https://supervision.roboflow.com/)

---

### 9. Veo - 视频生成（Google）

**功能**：通过 Google Veo 模型（Gemini API）实现文生视频和图生视频。**无需启动本地服务器**，直接调用云端 API。

**特点**：
- 支持文生视频（t2v）和图生视频（i2v）
- 生成的 `.mp4` 文件保存在本地 `outputs/` 目录
- 提供 mock 模式，无需 API Key 即可离线测试

**文件结构**：
```
Veo/
├── __init__.py
├── veo_client.py          # 真实 Gemini API 客户端
└── mock_veo_service.py    # 测试用 mock 服务
```

**参数说明**：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | ✅ | — | 视频内容的文字描述 |
| `image_path` | string | ❌ | None | 图生视频的参考图片路径 |
| `duration` | integer | ❌ | 8 | 时长（秒），可选 5 或 8 |
| `aspect_ratio` | string | ❌ | `"16:9"` | 宽高比：`"16:9"` 或 `"9:16"` |

**API Key 配置**：
```bash
export GOOGLE_API_KEY="your_google_api_key"
# 或
export GCP_API_KEY="your_gcp_api_key"
```

**工具测试**：
```bash
# 文生视频（无参考图）
python test/test_tool.py --tool veo \
    --image dummy \
    --prompt "A golden retriever running on a beach at sunset" \
    --duration 8

# 图生视频（有参考图）
python test/test_tool.py --tool veo \
    --image assets/dog.jpeg \
    --prompt "The dog starts running across the field" \
    --duration 8 \
    --aspect_ratio 16:9

# mock 模式（无需 API Key）
python test/test_tool.py --tool veo \
    --image dummy \
    --prompt "test video" \
    --use_mock
```

**评测**：
```bash
# 准备 JSONL 格式数据集（每行一条）：
# {"id":"1","prompt":"A sunset over the ocean","image":[],"task":"t2v","duration":8}
# {"id":"2","prompt":"The dog starts running","image":["assets/dog.jpeg"],"task":"i2v"}

# 使用真实 Veo API 跑评测
python examples/evaluation/evaluate_veo.py \
    --data_path dataset/veo_eval_data.jsonl \
    --model gpt-4o \
    --video_num_frames 4

# 使用 mock 服务跑评测
python examples/evaluation/evaluate_veo.py \
    --data_path dataset/veo_eval_data.jsonl \
    --use_mock \
    --max_samples 5

# 使用已有对话格式数据集
python examples/evaluation/evaluate_veo.py \
    --data_path dataset/tmp.jsonl \
    --image_base_path dataset \
    --model gpt-4o
```

**常用参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `dataset/veo_eval_data.jsonl` | JSONL 评测数据集路径 |
| `--image_base_path` | `.` | 图片路径的基础目录 |
| `--model` | `gpt-4o` | LLM 编排模型 |
| `--max_samples` | 全部 | 限制评测样本数 |
| `--video_num_frames` | `4` | 生成视频中抽取帧数后回传给模型 |
| `--use_mock` | false | 使用 mock 服务（无需 API Key） |
| `--max_iterations` | `3` | 每个样本最大工具调用轮次 |

---

### 10. Sora - 视频生成（OpenAI）

**功能**：通过 OpenAI Sora 模型实现文生视频和图生视频。**无需启动本地服务器**，直接调用云端 API。

**特点**：
- 支持文生视频（t2v）和图生视频（i2v）
- 支持分辨率控制：480p / 720p / 1080p
- 宽高比支持 1:1 方形（在 Veo 基础上新增）
- 生成的 `.mp4` 文件保存在本地 `outputs/` 目录
- 提供 mock 模式，无需 API Key 即可离线测试

**文件结构**：
```
Sora/
├── __init__.py
├── sora_client.py          # 真实 OpenAI API 客户端
└── mock_sora_service.py    # 测试用 mock 服务
```

**参数说明**：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | ✅ | — | 视频内容的文字描述 |
| `image_path` | string | ❌ | None | 图生视频的参考图片路径 |
| `duration` | integer | ❌ | 10 | 时长（秒），范围 5–20 |
| `resolution` | string | ❌ | `"1080p"` | 分辨率：`"480p"`、`"720p"` 或 `"1080p"` |
| `aspect_ratio` | string | ❌ | `"16:9"` | 宽高比：`"16:9"`、`"9:16"` 或 `"1:1"` |

**API Key 配置**：
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

**工具测试**：
```bash
# 文生视频
python test/test_tool.py --tool sora \
    --image dummy \
    --prompt "A cat sitting on a windowsill watching rain fall outside" \
    --duration 10 \
    --resolution 1080p

# 图生视频
python test/test_tool.py --tool sora \
    --image assets/dog.jpeg \
    --prompt "The dog starts running" \
    --duration 10 \
    --aspect_ratio 16:9

# mock 模式（无需 API Key）
python test/test_tool.py --tool sora \
    --image dummy \
    --prompt "test video" \
    --use_mock
```

**评测**：
```bash
# 准备 JSONL 格式数据集（每行一条）：
# {"id":"1","prompt":"A cat playing with yarn","image":[],"task":"t2v","duration":10,"resolution":"1080p"}
# {"id":"2","prompt":"The dog runs","image":["assets/dog.jpeg"],"task":"i2v","resolution":"720p"}

# 使用真实 Sora API 跑评测
python examples/evaluation/evaluate_sora.py \
    --data_path dataset/sora_eval_data.jsonl \
    --model gpt-4o \
    --video_num_frames 4

# 使用 mock 服务跑评测
python examples/evaluation/evaluate_sora.py \
    --data_path dataset/sora_eval_data.jsonl \
    --use_mock \
    --max_samples 5

# 使用已有对话格式数据集
python examples/evaluation/evaluate_sora.py \
    --data_path dataset/tmp.jsonl \
    --image_base_path dataset \
    --model gpt-4o
```

**常用参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `dataset/sora_eval_data.jsonl` | JSONL 评测数据集路径 |
| `--image_base_path` | `.` | 图片路径的基础目录 |
| `--model` | `gpt-4o` | LLM 编排模型 |
| `--max_samples` | 全部 | 限制评测样本数 |
| `--video_num_frames` | `4` | 生成视频中抽取帧数后回传给模型 |
| `--use_mock` | false | 使用 mock 服务（无需 API Key） |
| `--max_iterations` | `3` | 每个样本最大工具调用轮次 |

---

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
# 需要GPU内存 >= 24G
apt-get install tmux
pip install torch torchvision
pip install groundingdino_py supervision moondream
```

创建checkpoints目录：
```bash
mkdir -p checkpoints/{grounding_dino,depth_anything,pi3,pi3x,sam2}
```
### 2. 下载模型权重

每个工具都需要下载相应的模型权重文件，请参考各工具的详细说明。

### 3. 启动服务

如果要使用真实的专家服务而非mock模式，根据需要启动相应的服务器：
```bash
# 深度估计服务
python spagent/external_experts/Depth_AnythingV2/depth_server.py \
  --checkpoint_path checkpoints/depth_anything/depth_anything_v2_vitb.pth \
  --port 20019

# 部署SAM2分割服务，这里面需要将sam的权重名字rename成sam2.1_b.pt，否则会报错
python spagent/external_experts/SAM2/sam2_server.py \
  --checkpoint_path checkpoints/sam2/sam2.1_b.pt \
  --port 20020

# 部署grounding dino
# sometimes the network cannot connect the huggingface, we can reset the huggingfacesource
export HF_ENDPOINT=https://hf-mirror.com

python spagent/external_experts/GroundingDINO/grounding_dino_server.py \
  --checkpoint_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth \
  --port 20022

# 3D重建服务（Pi3）
python spagent/external_experts/Pi3/pi3_server.py \
  --checkpoint_path checkpoints/pi3/model.safetensors \
  --port 20030
  
# 3D重建服务（Pi3X - 增强版，推荐）
python spagent/external_experts/Pi3/pi3x_server.py \
  --checkpoint_path checkpoints/pi3x/model.safetensors \
  --port 20031

# VGGT多视角3D重建服务
python spagent/external_experts/VGGT/vggt_server.py \
  --checkpoint_path checkpoints/vggt \
  --port 20032

# MapAnything稠密3D重建服务（自动下载 facebook/map-anything）
python spagent/external_experts/mapanything/mapanything_server.py \
  --port 20033

# 视觉语言模型服务
python spagent/external_experts/moondream/md_server.py \
  --port 20024
```