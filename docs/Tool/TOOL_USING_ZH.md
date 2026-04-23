# External Experts Module

External Experts 模块包含了专门用于空间智能任务的专业模型，包括深度估计、目标检测、分割、3D重建、视频生成等功能。工具分为两类：本地 server/client 架构和云端 API 直调（Veo、Sora），均支持独立部署和调用。

## 📁 模块结构

```
external_experts/
├── __init__.py                     # 模块初始化
├── checkpoints/                    # 所有模型权重文件
│   └──depth_anything
│   └──grounding_dino
│   └──orient_anything_v2
│   └──pi3
│   └──pi3x
│   └──sam2
│   └──vggt
│   └──Wan2.1-VACE-1.3B
├── GroundingDINO/                  # 开放词汇目标检测
├── SAM2/                          # 图像和视频分割
├── Depth_AnythingV2/              # 深度估计
├── Pi3/                           # 3D重建 (Pi3 & Pi3X)
├── VGGT/                          # 多视角3D重建与相机位姿估计
├── mapanything/                   # 基于深度估计的稠密3D重建
├── moondream/                     # 视觉语言模型
├── OrientAnythingV2/              # 物体朝向与相对旋转估计
├── Veo/                           # Google Veo 视频生成（API 直调，无需本地服务器）
├── Sora/                          # OpenAI Sora 视频生成（API 直调，无需本地服务器）
├── vace/                          # VACE 本地视频生成（首帧驱动流水线，服务端口 20034）
└── supervision/                   # YOLO目标检测和标注工具
```

## 🛠️ 工具概览

| 工具名称 | Tool Class | 功能 | 主要用途 | 部署方式 | 主要参数 |
|---------|------------|------|----------|----------|----------|
| **Depth AnythingV2** | `DepthEstimationTool` | 深度估计 | 单目深度估计，分析图像中的3D深度关系 | 本地服务器（20019） | `image_path` |
| **SAM2** | `SegmentationTool` | 图像/视频分割 | 高精度分割任务，精确分割图像中的对象 | 本地服务器（20020） | `image_path`, `point_coords`(可选), `point_labels`(可选), `box`(可选) |
| **GroundingDINO** | `ObjectDetectionTool` | 开放词汇目标检测 | 基于文本描述检测任意物体 | 本地服务器（20022） | `image_path`, `text_prompt`, `box_threshold`, `text_threshold` |
| **Moondream** | `MoondreamTool` | 视觉语言模型 | 图像理解和问答，基于图像内容回答自然语言问题 | 本地服务器（20024） | `image_path`, `task`, `object_name` |
| **Molmo2** | `Molmo2Tool` | 多模态推理与点选定位 | 通过本地 Molmo2 服务执行图像问答、描述和 point grounding，可选保存标注图 | 本地服务器（20025） | `image_path`, `task`, `prompt`(可选), `save_annotated`(可选), `max_new_tokens`(可选) |
| **Pi3** | `Pi3Tool` | 3D重建 | 从图像生成3D点云和多视角渲染图 | 本地服务器（20030） | `image_path`, `azimuth_angle`, `elevation_angle` |
| **Pi3X** | `Pi3XTool` | 3D重建（增强版） | Pi3升级版，更平滑点云、近似度量尺度、可选多模态条件注入 | 本地服务器（20031） | `image_path`, `azimuth_angle`, `elevation_angle` |
| **VGGT** | `VGGTTool` | 多视角3D重建与相机位姿估计 | 从多张图像或视频帧重建3D点云并估计相机位姿 | 20032 | `image_paths`, `azimuth_angle`, `elevation_angle`, `rotation_reference_camera`, `camera_view` |
| **MapAnything** | `MapAnythingTool` | 基于深度估计的稠密3D重建 | 利用深度图和相机位姿从多张图像重建稠密3D点云 | 20033 | `image_paths`, `azimuth_angle`, `elevation_angle`, `conf_percentile`, `apply_mask` |
| **Supervision** | `SupervisionTool` | 目标检测标注 | YOLO模型和可视化工具，通用目标检测和分割 | 本地 | `image_path`, `task` ("image_det" 或 "image_seg") |
| **YOLO-E** | `YOLOETool` | YOLO-E检测 | 高精度检测，支持自定义类别 | 本地 | `image_path`, `task`, `class_names` |
| **YOLO26** | `YOLO26Tool` | 目标检测 | 基于 Ultralytics YOLO26 的本地快速目标检测，返回边界框、类别标签和置信度 | 本地（无需服务器） | `image_path`, `conf`(可选), `save_annotated`(可选) |
| **Veo** | `VeoTool` | 视频生成 | 通过 Google Veo（Gemini API）实现文生视频和图生视频 | API 直调（无需服务器） | `prompt`, `image_path`(可选), `duration`, `aspect_ratio` |
| **Sora** | `SoraTool` | 视频生成 | 通过 OpenAI Sora 实现文生视频和图生视频 | API 直调（无需服务器） | `prompt`, `image_path`(可选), `duration`, `resolution`, `aspect_ratio` |
| **Orient Anything V2** | `OrientAnythingV2Tool` | 物体朝向与旋转估计 | 估计物体绝对朝向（方位角/仰角/旋转角/对称阶数）以及两视角间的相对位姿（NeurIPS 2025 Spotlight） | 本地服务器（20034） | `image_path`, `task`, `image_path2`(可选) |
| **VACE** | `VaceTool` | 本地视频生成 | 基于单张参考图 + 文本提示词，通过本地 Wan2.1-VACE 首帧流水线生成短视频，返回 `.mp4` 路径 | 本地服务器（20034） | `image_path`, `prompt`, `base`(可选), `task`(可选), `mode`(可选) |

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

### 4.1 Molmo2 - 多模态推理与点选定位服务

**功能**: 通过本地 Molmo2 服务执行图像问答、图像描述和 point grounding。

**特点**:
- 与项目里的其他重型工具保持一致，采用本地 server + HTTP client + mock service 的方式
- 支持三种任务模式：`qa`、`caption`、`point`
- 支持 point grounding，并可选保存标注图
- 提供 mock 模式，便于开发和测试

**文件结构**:
```text
spagent/external_experts/Molmo2/
├── download_weights.py
├── molmo2_server.py
├── molmo2_client.py
├── mock_molmo2_service.py
├── point_utils.py
└── __init__.py
```

**推荐安装**:
```bash
pip install -r requirements.txt
pip install "transformers>=4.57,<5" accelerate sentencepiece huggingface_hub
```

如需参考上游源码安装:
```bash
git clone https://github.com/allenai/molmo2.git
cd molmo2
pip install torchcodec
pip install -e .[all]
```

**下载权重**:
```bash
# 将 Hugging Face 模型快照下载到本地目录
python spagent/external_experts/Molmo2/download_weights.py \
    --repo allenai/Molmo2-4B \
    --local-dir checkpoints/molmo2/Molmo2-4B

# 启动服务前指定模型目录
export MOLMO2_MODEL=checkpoints/molmo2/Molmo2-4B
```

**启动服务**:
```bash
python spagent/external_experts/Molmo2/molmo2_server.py \
    --checkpoint allenai/Molmo2-4B \
    --port 20035
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_path` | string | ✅ | — | 输入图片路径 |
| `task` | string | ❌ | `"qa"` | `qa`、`caption`、`point` 三选一 |
| `prompt` | string | ❌ | 按任务自动填充 | 问答问题、描述指令或 point 指令 |
| `save_annotated` | boolean | ❌ | `True` | 当 `task="point"` 时是否保存标注图 |
| `max_new_tokens` | integer | ❌ | `200` | Molmo2 输出的最大生成长度 |

默认提示词：
- `qa`: `What do you see in this image? Answer briefly.`
- `caption`: `Describe this image.`
- `point`: `Point to the requested location. Output pointing coordinates in the model's standard format.`

**Python 示例**:
```python
from spagent.tools import Molmo2Tool

tool = Molmo2Tool(
    use_mock=False,
    server_url="http://localhost:20025",
    output_dir="outputs/molmo2",
)

qa_result = tool.call(
    image_path="assets/dog.jpeg",
    task="qa",
    prompt="图中是什么动物？",
    max_new_tokens=64,
)
print(qa_result["response_text"])

caption_result = tool.call(
    image_path="assets/dog.jpeg",
    task="caption",
)
print(caption_result["response_text"])

point_result = tool.call(
    image_path="assets/dog.jpeg",
    task="point",
    prompt="请指向这只狗。",
    save_annotated=True,
)
print(point_result["result"]["points_by_image"])
print(point_result["output_path"])  # 默认保存在系统临时目录，除非显式设置 output_dir
```

**直接测试**:
```bash
python test/test_tool.py --tool molmo2 --image assets/dog.jpeg --task qa --prompt "Describe the dog" --server_url http://localhost:20035
python test/test_tool.py --tool molmo2 --image assets/dog.jpeg --task caption --server_url http://localhost:20035
python test/test_tool.py --tool molmo2 --image assets/dog.jpeg --task point --prompt "Point to the dog" --use_mock --save_annotated

# 单元测试
pytest test/test_molmo2_tool.py -v
python -m unittest test.test_molmo2_expert -v
```

**资源链接**:
- [官方仓库](https://github.com/allenai/molmo2)
- [论文](https://arxiv.org/abs/2601.10611)
- [Hugging Face Models](https://huggingface.co/collections/allenai/molmo2)

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

### 11. YOLO26 - 本地目标检测

**功能**：基于 Ultralytics YOLO26 的本地目标检测，无需启动任何服务器，模型在进程内直接运行。

**特点**：
- 检测图像中的物体，返回边界框（xyxy 格式）、类别标签和置信度
- 可选保存带标注框的可视化输出图片
- 支持自定义置信度阈值、IOU 阈值及最大检测数
- 兼容任意 YOLO26 权重文件（nano、small、medium、large 等）

**文件结构**：
```
spagent/tools/
└── yolo26_tool.py          # YOLO26Tool 实现
checkpoints/yolo26/
└── yolo26n.pt              # 默认权重文件（放置于此）
test/yolo26/
├── test_yolo26_tool_real.py   # 真实推理集成测试
└── README.md                  # 测试说明文档
```

**安装依赖**：
```bash
pip install ultralytics opencv-python
```

**权重下载**：
```bash
mkdir -p checkpoints/yolo26
# 从 Ultralytics 下载（以 nano 版本为例）
wget -O checkpoints/yolo26/yolo26n.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

**参数说明**：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_path` | string | ✅ | — | 输入图像路径 |
| `conf` | float | ❌ | `0.25` | 置信度阈值（0–1） |
| `save_annotated` | boolean | ❌ | `True` | 是否将标注图片保存到 `outputs/yolo26/` |

**快速使用**：
```python
from spagent.tools import YOLO26Tool

tool = YOLO26Tool(
    model_path="checkpoints/yolo26/yolo26n.pt",
    device="cpu",          # 或 "cuda:0"
    conf=0.25,
    save_annotated=True,
    output_dir="outputs/yolo26",
)

result = tool.call(image_path="assets/example.png")
print(result["result"]["num_detections"])
print(result["result"]["detections"])   # 每条检测：{bbox_xyxy, class_id, class_name, confidence}
print(result["output_path"])            # 标注图片路径
```

**集成测试**：
```bash
# 真实推理测试（需要权重文件和测试图片）
RUN_REAL_YOLO26_TEST=1 python -m pytest -q test/yolo26/test_yolo26_tool_real.py

# 通过环境变量覆盖默认路径和设备
RUN_REAL_YOLO26_TEST=1 \
YOLO26_MODEL_PATH=checkpoints/yolo26/yolo26n.pt \
YOLO26_DEVICE=cpu \
python -m pytest -q test/yolo26/test_yolo26_tool_real.py
```

**评测**：
```bash
python examples/evaluation/evaluate_yolo26.py \
    --data_path dataset/cvbench_data.jsonl \
    --image_base_path dataset \
    --model_path checkpoints/yolo26/yolo26n.pt \
    --device cpu \
    --model gpt-4o \
    --max_samples 100

# 也可通过环境变量指定权重和设备
YOLO26_MODEL_PATH=checkpoints/yolo26/yolo26n.pt \
YOLO26_DEVICE=cuda:0 \
python examples/evaluation/evaluate_yolo26.py ...
```

**常用参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `checkpoints/yolo26/yolo26n.pt` | YOLO26 权重路径 |
| `--device` | `cpu` | 推理设备（`cpu` 或 `cuda:0`） |
| `--conf` | `0.25` | 检测置信度阈值 |
| `--yolo_output_dir` | `outputs/yolo26` | 标注图片保存目录 |
| `--model` | `gpt-4o` | LLM 编排模型 |
| `--max_samples` | 全部 | 限制评测样本数 |
| `--max_iterations` | `3` | 每个样本最大工具调用轮次 |

**资源链接**：
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLO 文档](https://docs.ultralytics.com/)
### 11. Orient Anything V2 - 物体朝向与旋转估计

**功能**：统一的空间视觉模型，支持三维朝向估计、旋转对称性检测和两视角间相对位姿估计（NeurIPS 2025 Spotlight）

**特点**：
- 无需物体类别标签，完全类别无关
- 绝对朝向：方位角（0-360°）、仰角（-90~90°）、平面内旋转（-180~180°）
- 旋转对称阶数：`symmetry_alpha` ∈ {0, 1, 2, 4}
- 双图模式：输出目标图相对于参考图的相对位姿
- 支持背景去除（基于 `rembg`）
- 提供 mock 模式，无需 GPU 即可开发调试

**文件结构**：
```
OrientAnythingV2/
├── oa_v2_server.py            # Flask 服务器（端口 20034）
├── oa_v2_client.py            # HTTP 客户端
├── mock_oa_v2_service.py      # 离线测试用 mock 服务
├── download_weights.sh        # 权重下载脚本
└── __init__.py
```

**模型规格**：

| 项目 | 详情 |
|------|------|
| 模型 | VGGT_OriAny_Ref（VGGT-1B 骨干 + 朝向预测头） |
| 参数量 | ~5.05 GB |
| 权重文件 | `checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt` |
| 源码仓库 | HuggingFace Space `Viglong/Orient-Anything-V2` |
| 论文 | NeurIPS 2025 Spotlight |

**输出字段**：

| 字段 | 范围 | 说明 |
|------|------|------|
| `azimuth` | 0-360° | 物体正面的绝对方位角 |
| `elevation` | -90~90° | 绝对仰角 |
| `rotation` | -180~180° | 平面内旋转角 |
| `symmetry_alpha` | 0/1/2/4 | 旋转对称阶数（0=不确定，1=双侧对称，2=二重，4=四重） |
| `rel_azimuth` | 0-360° | 目标图相对参考图的方位角差（双图模式） |
| `rel_elevation` | -90~90° | 仰角差（双图模式） |
| `rel_rotation` | -180~180° | 旋转角差（双图模式） |

**环境配置**：
```bash
# 1. Clone HF Space 源码（含 vggt 子包，无需单独安装 vggt）
git clone https://huggingface.co/spaces/Viglong/Orient-Anything-V2 \
    third_party/orient_anything_v2

# 2. 安装额外依赖
pip install rembg timm

# 3. 下载权重
bash spagent/external_experts/OrientAnythingV2/download_weights.sh
# 权重保存至：checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt
```

**启动服务**：
```bash
python spagent/external_experts/OrientAnythingV2/oa_v2_server.py \
    --checkpoint_path checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt \
    --repo_path third_party/orient_anything_v2 \
    --port 20034
```

**工具测试**：
```bash
# mock 模式——无需 GPU 或服务器
python test/test_orient_anything_v2_tool.py \
    --use_mock --task orientation --image_path assets/dog.jpeg

# 单图方向估计，连接真实服务器
python test/test_orient_anything_v2_tool.py \
    --task orientation --image_path assets/dog.jpeg

# 双图相对旋转，连接真实服务器
python test/test_orient_anything_v2_tool.py \
    --task relative_rotation \
    --image_path assets/dog.jpeg --image_path2 assets/example.png
```

**Python 调用示例**：
```python
from spagent.tools.orient_anything_v2_tool import OrientAnythingV2Tool

# mock 模式
tool = OrientAnythingV2Tool(use_mock=True)

# 单图：绝对朝向
result = tool.call(image_path="assets/dog.jpeg", object_category="dog")
# {'success': True, 'result': {'azimuth': 143, 'elevation': -12, 'rotation': 5, 'symmetry_alpha': 1}}

# 双图：绝对朝向 + 相对位姿
result = tool.call(
    image_path="ref.jpg",
    task="relative_rotation",
    image_path2="target.jpg",
    object_category="chair",
)
# {'success': True, 'result': {'azimuth': 143, ..., 'rel_azimuth': 72, 'rel_elevation': 8, 'rel_rotation': -15}}
```

**资源链接**：
- [HuggingFace Space（源码）](https://huggingface.co/spaces/Viglong/Orient-Anything-V2)
- [HuggingFace 权重](https://huggingface.co/Viglong/OriAnyV2_ckpt)

---

---

### 12. VACE - 本地视频生成（首帧驱动流水线）

**功能**：基于单张参考图和文本运动提示词，通过本地部署的 Wan2.1-VACE 模型生成短视频。

**特点**：
- 图生视频（首帧→视频）全部在本地 GPU 上运行，无需云端 API
- 使用 [Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) 模型
- 生成的 `.mp4` 视频保存于 `vace/results/` 目录
- 提供 mock 模式，无需 GPU 即可离线开发测试
- 无需 FlashAttention，自动降级至 PyTorch `scaled_dot_product_attention`

**文件结构**：
```
vace/
├── vace_server.py          # Flask 服务器（端口 20034）
├── vace_client.py          # HTTP 客户端
├── vace/                   # VACE 流水线运行时代码
│   ├── vace_pipeline.py
│   ├── vace_wan_inference.py
│   ├── annotators/
│   ├── configs/
│   └── models/
└── third_party/
    └── Wan2.1/             # 内嵌的 Wan2.1 模型代码
```

**环境要求**：
- NVIDIA GPU（默认分辨率下建议显存 ≥ 8 GB）
- Python 3.11，PyTorch ≥ 2.0

**安装依赖**：
```bash
pip install -r requirements-vace.txt
```

**权重下载**：
```bash
python -m pip install -U "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \
    --local-dir checkpoints/Wan2.1-VACE-1.3B \
    --local-dir-use-symlinks False
```

**启动服务**：
```bash
python spagent/external_experts/vace/vace_server.py \
    --checkpoint_path checkpoints/Wan2.1-VACE-1.3B \
    --port 20034
```

健康检查：
```bash
curl -s http://127.0.0.1:20034/health
```

**参数说明**：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_path` | string | ✅ | — | 首帧参考图片路径 |
| `prompt` | string | ✅ | — | 视频运动/场景描述提示词 |
| `base` | string | ❌ | `"wan"` | VACE 基础模型后端 |
| `task` | string | ❌ | `"frameref"` | VACE 任务名称 |
| `mode` | string | ❌ | `"firstframe"` | 流水线模式（`firstframe`、`lastframe` 等） |

**Python 调用示例**：
```python
from spagent.tools import VaceTool

# mock 模式——无需服务器或 GPU
tool = VaceTool(use_mock=True)

# 真实服务器
tool = VaceTool(use_mock=False, server_url="http://localhost:20034")

result = tool.call(
    image_path="assets/example.png",
    prompt="缓慢向前移动",
)
print(result["output_path"])   # 生成的 .mp4 路径
```

**工具测试**：
```bash
# mock 模式
python test/test_tool.py --tool vace \
    --image assets/example.png \
    --prompt "move forward" \
    --use_mock

# 真实服务器
python test/test_tool.py --tool vace \
    --image assets/example.png \
    --prompt "move forward" \
    --server_url http://localhost:20034
```

**资源链接**：
- [Wan2.1-VACE-1.3B（HuggingFace）](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)
- [VACE GitHub](https://github.com/ali-vilab/VACE)

---

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
# 需要GPU内存 >= 24G
apt-get install tmux
pip install torch torchvision
pip install groundingdino_py supervision moondream
pip install ai2-molmo2 accelerate sentencepiece
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

# Orient Anything V2 朝向估计服务
python spagent/external_experts/OrientAnythingV2/oa_v2_server.py \
  --checkpoint_path checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt \
  --repo_path third_party/orient_anything_v2 \
  --port 20034

# 视觉语言模型服务
python spagent/external_experts/moondream/md_server.py \
  --port 20024

# Molmo2 多模态推理服务
python spagent/external_experts/Molmo2/molmo2_server.py \
  --checkpoint allenai/Molmo2-4B \
  --port 20025

# VACE 本地视频生成服务（Wan2.1-VACE 首帧流水线）
python spagent/external_experts/vace/vace_server.py \
  --checkpoint_path checkpoints/Wan2.1-VACE-1.3B \
  --port 20034
```
