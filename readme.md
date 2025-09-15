# 📌 Introduction

This repository provides **SPAgent** - a flexible and modular **Spatial Intelligence Agent** that integrates **agentic skills** into **multi-modal understanding** using external expert models and LLMs.

## 🆕 **New SPAgent Architecture (v2.0)**

**SPAgent** replaces the old workflow system with a modern, modular architecture:

- ✅ **Modular Tool System** - Mix and match any combination of expert tools
- ✅ **Dynamic Tool Management** - Add/remove tools at runtime
- ✅ **Parallel Tool Execution** - Automatic concurrent processing when possible
- ✅ **Multi-Image Analysis** - Handle single or multiple images seamlessly
- ✅ **Multiple Model Support** - GPT, Qwen, and local VLLM models
- ✅ **Flexible Configuration** - Easy to customize and extend

---

## 📂 Project Structure

| Module | Path | Description |
|--------|------|-------------|
| **SPAgent Core** | `spagent/core/` | 🆕 Main agent architecture:<br>- SPAgent class<br>- Tool base classes<br>- Model wrappers<br>- Unified prompt system |
| **Tools** | `spagent/tools/` | 🆕 Modular expert tools:<br>- DepthEstimationTool<br>- SegmentationTool<br>- ObjectDetectionTool<br>- SupervisionTool<br>- YOLOETool |
| **Models** | `spagent/models/` | 🆕 Model wrappers:<br>- GPTModel<br>- QwenModel<br>- QwenVLLMModel |
| **External Experts** | `spagent/external_experts/` | Specialized models for spatial intelligence:<br>- Depth Estimation (**Depth-AnythingV2**)<br>- Object Detection & Segmentation (**SAM2**)<br>- Open-vocabulary Detection (**GroundingDINO**)<br>- 3D Reconstruction (**Pi3**)<br>- Can run as external APIs |
| **VLLM Models** | `spagent/vllm_models/` | VLLM inference functions & wrappers:<br>- GPT / QwenVL inference<br>- Model loading & serving utilities<br>- Unified API for LLM calls |
| **Examples** | `spagent/examples/` | Example scripts and usage tutorials |
| **Legacy Workflows** | `spagent/workflows/` | ⚠️ **Deprecated** - Old workflow system |

---

## 🚀 Quick Start

### 1. 基础使用 (Basic Usage)

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool

# 创建模型和工具
model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=True),    # 深度估计
    SegmentationTool(use_mock=True)        # 图像分割
]

# 创建智能体
agent = SPAgent(model=model, tools=tools)

# 解决问题
result = agent.solve_problem("image.jpg", "分析这张图片的深度关系和主要对象")
print(result['answer'])
```

### 2. 混合多工具使用 (Multi-Tool Usage)

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import (
    DepthEstimationTool,      # 深度估计
    SegmentationTool,         # 图像分割  
    ObjectDetectionTool,      # 目标检测
    SupervisionTool,          # 监督学习工具
    YOLOETool                 # YOLO-E检测
)

# 创建全功能智能体
model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True),
    ObjectDetectionTool(use_mock=True),
    SupervisionTool(use_mock=True),
    YOLOETool(use_mock=True)
]

agent = SPAgent(model=model, tools=tools, max_workers=4)

# 复杂问题分析
result = agent.solve_problem(
    "image.jpg", 
    "全面分析这张图片：识别所有对象，分析深度关系，并分割重要区域"
)

print(f"答案: {result['answer']}")
print(f"使用的工具: {result['used_tools']}")
print(f"生成的额外图像: {result['additional_images']}")
```

### 3. 动态工具管理 (Dynamic Tool Management)

```python
# 从基础智能体开始
agent = SPAgent(model=GPTModel())

# 动态添加工具
agent.add_tool(DepthEstimationTool(use_mock=True))
agent.add_tool(SegmentationTool(use_mock=True))

# 查看当前工具
print(f"当前工具: {agent.list_tools()}")

# 移除不需要的工具
agent.remove_tool("depth_estimation_tool")

# 更换模型
from spagent.models import QwenModel
agent.set_model(QwenModel(model_name="qwen2.5-vl-7b-instruct"))
```

### 4. 多图像分析 (Multi-Image Analysis)

```python
# 分析多张图像
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
result = agent.solve_problem(
    image_paths, 
    "比较这些图像的差异，分析深度变化和对象分布"
)
```

---
### 5. 图像数据集评测 (Image Dataset Evaluation)

本节介绍如何在图像数据集上评测SPAgent的性能。所有数据集都需要先下载并转换为统一的JSONL格式，其中每条数据包含以下标准字段：
- `id`: 数据样本的唯一标识符
- `image`: 图片路径列表（支持多图像），若没有则为空
- `video`：视频路径列表，若没有则为空
- `conversations`: 对话格式的问答内容，需包含问题选项和答案，如（"conversations": [{"from": "human", "value": "{question}\nSelect from the following choices. (A) .. A (B) .."},{"from": "gpt", "value": "A"}],）
- `task`: 任务类型（如Object_Localization, Depth, Count等）
- `input_type`: 输入类型（通常为"Image"）
- `output_type`: 输出类型（如"MCQ"表示多选题）
- `data_source`: 数据集来源

#### 1. BLINK数据集

```bash
# 下载BLINK数据集并转换为JSONL格式
python spagent/utils/download_blink.py

# 运行评测
python evaluate_img.py --data_path dataset/BLINK_All_Tasks.jsonl --max_workers 4 --image_base_path dataset --model gpt-4o-mini
```

#### 2. CVBench数据集
CVBench专注于计算机视觉的基础能力测试，包括深度估计、目标计数、空间关系等任务。

```bash
# 第一步：下载CVBench图片（需要先保存parquet文件到dataset目录）
# 数据集地址：https://huggingface.co/datasets/nyu-visionx/CV-Bench
python spagent/utils/cvbench_img.py --subset both --root dataset --out dataset/CVBench

# 第二步：转换为JSONL格式
python spagent/utils/download_cvbench.py

# 第三步：创建样本数据（可选，用于快速测试）
python dataset/create_cvbench_sample.py

# 运行评测
python evaluate_img.py --data_path dataset/cvbench_data.jsonl --max_samples 30 --max_workers 4 --image_base_path dataset --model gpt-4o-mini
```

## 🛠️ 安装和配置 (Installation & Setup)

### 1. 环境准备 (Environment Setup)

```bash
# 创建Python 3.11环境 (其他版本可能有兼容性问题)
conda create -n spagent python=3.11
conda activate spagent

# 安装依赖
pip install -r requirements.txt
pip install "httpx[socks]"
```

### 2. API配置 (API Configuration)

```bash
# OpenAI API
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="http://35.220.164.252:3888/v1/"

# Qwen API (申请地址: https://bailian.console.aliyun.com)
export DASHSCOPE_API_KEY="your_api_key"

# moondream API（申请地址：https://moondream.ai）
export MOONDREAM_API_KEY="your_api_key"

# 测试API连接
python spagent/vllm_models/qwen.py
```

### 3. 下载模型权重 (Download Model Weights)

创建checkpoints目录：
```bash
mkdir -p checkpoints/{grounding_dino,depth_anything,pi3,sam2}
```

#### Depth-Anything V2 (深度估计)
```bash
# 选择一个模型 (推荐Base版本)
cd checkpoints/depth_anything

# Small (~25MB, 最快)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# Base (~100MB, 平衡) - 推荐
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

# Large (~350MB, 最高质量)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

#### SAM2 (图像分割)
```bash
cd checkpoints/sam2

# 自动下载所有模型
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh

# 或手动下载推荐模型
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

#### GroundingDINO (目标检测)
```bash
cd checkpoints/grounding_dino
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

```

### 4. 部署外部专家服务 (Deploy External Expert Services)

如果要使用真实的专家服务而非mock模式：

```bash
# 需要GPU内存 >= 24G
apt-get install tmux

# 部署深度估计服务
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
  --model_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth \
  --port 20022

# 部署moondream
python spagent/external_experts/GroundingDINO/grounding_dino_server.py 
  --port 20024
```

---

## 🎯 运行示例 (Run Examples)

### 新SPAgent示例 (New SPAgent Examples)

```bash
cd spagent

# 基础SPAgent使用示例
python examples/spagent_example.py assets/example.png "分析这张图片"

# 使用真实图片测试
python examples/spagent_example.py your_image.jpg "描述图片中的对象和深度关系"
```

### 工具定义示例 (Tool Definition Examples)

#### 1. 深度分析专用智能体
```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool

# 专注深度分析的智能体
model = GPTModel(model_name="gpt-4o-mini")
depth_tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True)  # 辅助分割
]

depth_agent = SPAgent(model=model, tools=depth_tools)
result = depth_agent.solve_problem(
    "image.jpg", 
    "分析图片的深度分布，哪些物体离相机近，哪些远？"
)
```

#### 2. 目标检测专用智能体
```python
from spagent.tools import ObjectDetectionTool, SupervisionTool, YOLOETool

# 专注目标检测的智能体
detection_tools = [
    ObjectDetectionTool(use_mock=True),
    SupervisionTool(use_mock=True),
    YOLOETool(use_mock=True),
    SegmentationTool(use_mock=True)  # 辅助分割
]

detection_agent = SPAgent(model=model, tools=detection_tools)
result = detection_agent.solve_problem(
    "image.jpg", 
    "检测并识别图片中的所有对象，包括位置和类型"
)
```

#### 3. 自定义工具组合
```python
# 创建空智能体，逐步添加工具
agent = SPAgent(model=GPTModel())

# 根据需要添加工具
if need_depth:
    agent.add_tool(DepthEstimationTool(use_mock=True))

if need_detection:
    agent.add_tool(ObjectDetectionTool(use_mock=True))
    
if need_segmentation:
    agent.add_tool(SegmentationTool(use_mock=True))

# 使用配置好的智能体
result = agent.solve_problem("image.jpg", "根据可用工具分析图片")
```

---

## 🔧 工具混合策略 (Tool Mixing Strategies)

### 1. 并行工具执行 (Parallel Tool Execution)
SPAgent会自动检测可以并行执行的工具：

```python
# 这个问题会触发多个工具并行执行
result = agent.solve_problem(
    "image.jpg",
    "同时进行深度估计、目标检测和图像分割"  # 会并行执行3个工具
)
```

### 2. 条件工具选择 (Conditional Tool Selection)
模型会根据问题自动选择需要的工具：

```python
# 只会使用深度相关的工具
result1 = agent.solve_problem("image.jpg", "分析深度关系")

# 只会使用检测相关的工具  
result2 = agent.solve_problem("image.jpg", "检测车辆和行人")

# 会使用多种工具
result3 = agent.solve_problem("image.jpg", "全面分析图片")
```

### 3. 工具链组合 (Tool Chain Combination)
```python
# 复杂工具链：检测 → 分割 → 深度分析
result = agent.solve_problem(
    "image.jpg",
    """
    首先检测图片中的主要对象，
    然后对检测到的对象进行精确分割，
    最后分析这些对象的深度关系
    """
)
```

---

## 📖 可用工具列表 (Available Tools)

| 工具类 | 功能 | 用途 | 参数 |
|--------|------|------|------|
| `DepthEstimationTool` | 深度估计 | 分析图像的3D深度关系 | `image_path` |
| `SegmentationTool` | 图像分割 | 精确分割图像中的对象 | `image_path`, `point_coords`(可选), `box`(可选) |
| `ObjectDetectionTool` | 目标检测 | 基于文本描述检测对象 | `image_path`, `text_prompt`, `box_threshold`, `text_threshold` |
| `SupervisionTool` | 监督检测 | 通用目标检测和分割 | `image_path`, `task` ("image_det"或"image_seg") |
| `YOLOETool` | YOLO-E检测 | 自定义类别的高精度检测 | `image_path`, `task`, `class_names` |

## 🤖 可用模型 (Available Models)

| 模型类 | 描述 | 推荐用途 |
|--------|------|----------|
| `GPTModel` | OpenAI GPT模型 | 通用视觉理解，最佳效果 |
| `QwenModel` | 通义千问VL模型 | 中文理解优秀 |
| `QwenVLLMModel` | 本地部署的Qwen VLLM | 本地推理，保护隐私 |

---

## 📊 性能优势 (Performance Benefits)

### 新架构 vs 旧Workflow系统

| 特性 | 旧Workflow | 新SPAgent | 改进 |
|------|------------|-----------|------|
| 代码复用 | 每个组合需要单独的workflow类 | 单一SPAgent类处理所有组合 | **90%代码减少** |
| 工具组合 | 固定组合，难以修改 | 任意组合，动态调整 | **无限灵活性** |
| 并行执行 | 串行执行工具 | 自动并行执行 | **3-5x性能提升** |
| 扩展性 | 添加工具需要修改多个类 | 添加工具只需实现Tool接口 | **易于扩展** |
| 维护性 | 大量重复代码 | 清晰的模块分离 | **易于维护** |

---

## 🔄 从旧系统迁移 (Migration from Old System)

详细迁移指南请查看：[MIGRATION_GUIDE.md](spagent/MIGRATION_GUIDE.md)

### 快速迁移示例：

**旧代码:**
```python
from workflows.mix_workflow import MixedExpertWorkflow
workflow = MixedExpertWorkflow(use_mock=True)
result = workflow.run_workflow("image.jpg", "分析图片")
```

**新代码:**
```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool, ObjectDetectionTool

model = GPTModel()
tools = [DepthEstimationTool(use_mock=True), SegmentationTool(use_mock=True), ObjectDetectionTool(use_mock=True)]
agent = SPAgent(model=model, tools=tools)
result = agent.solve_problem("image.jpg", "分析图片")
```

---

## 🧪 测试和开发 (Testing & Development)

### Mock模式测试
```python
# 使用mock模式进行快速测试（不需要部署实际服务）
tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True),
    ObjectDetectionTool(use_mock=True)
]
```

### 真实服务模式
```python
# 使用真实部署的服务
tools = [
    DepthEstimationTool(use_mock=False, server_url="http://localhost:20019"),
    SegmentationTool(use_mock=False, server_url="http://localhost:20020"),
    ObjectDetectionTool(use_mock=False, server_url="http://localhost:30969")
]
```

---

## ⚠️ 注意事项 (Important Notes)

1. **Python版本**: 建议使用Python 3.11，其他版本可能有兼容性问题
2. **内存要求**: 真实模式需要GPU内存 >= 24GB
3. **网络配置**: 确保API密钥和服务器地址配置正确
4. **并发控制**: 可通过`max_workers`参数控制并行工具数量

---

## 🔍 External Experts
| 工具名称 | 类型 | 主要功能 | 备注 |
| --- | --- | --- | --- |
| **Depth-AnythingV2** | 3D | 单目深度估计 | 将 2D 图像转为像素级深度图 |
| **SAM2** | 2D | 图像分割 | Segment Anything 模型第二代，交互式或自动分割 |
| **Supervision** | 2D | 视觉任务辅助工具库 | 用于目标检测、分割结果可视化和后处理 |
| **GroundingDINO** | 2D | 文本驱动目标检测 | 基于自然语言进行检测和框选 |
| **Pi3** | 3D | 点云生成与处理 | 将图像或多视角输入转为 3D 表示 |

## 📈 Future Roadmap

- [ ] 支持更多专家工具
- [ ] 添加工具执行策略配置
- [ ] 实现工具结果缓存
- [ ] 支持流式处理
- [ ] 添加性能监控
- [ ] 完善文档和教程





