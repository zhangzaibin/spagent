# 📌 Introduction

This repository integrates **agentic skills** into **multi-modal understanding** using external expert models and LLMs.

---

## 📂 Project Structure

| Module | Path | Description |
|--------|------|-------------|
| **External Experts** | `spagent/external_experts/` | Specialized models for spatial intelligence:<br>- Depth Estimation (**Depth-AnythingV2**)<br>- Object Detection & Segmentation (**SAM2**)<br>- Open-vocabulary Detection (**GroundingDINO**)<br>- 3D Reconstruction (**Pi3**)<br>- Can run as external APIs |
| **VLLM Models** | `spagent/vllm_models/` | VLLM inference functions & wrappers:<br>- GPT / QwenVL inference<br>- Model loading & serving utilities<br>- Unified API for LLM calls |
| **Workflows** | `spagent/workflows/` | Orchestrates complete workflows:<br>- Combines LLM + external experts<br>- Defines spatial reasoning pipelines<br>- Manages data flow |
| **Examples** | `spagent/examples/` | Example scripts, each showing a usage tutorial (e.g., `depth_workflow_example_usage.py`) |

---

## 🚀 Quick Start

### 1. Prepare APIs
```bash
# OpenAI API
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="http://35.220.164.252:3888/v1/"

# Qwen API (apply at https://bailian.console.aliyun.com)
export DASHSCOPE_API_KEY="your_api_key"

# Test Qwen API
python spagent/vllm_models/qwen.py

# prepare VLLM in the iiau A800 server
vllm serve /13693266743/models/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 20004 --served-model-name 'qwen-vl' 

# Then, you can run
python spagent/vllm_models/qwen_vllm.py

# 现在我已经在A800上部署了，ip什么都是固定的，直接跑就行，24小时内都能用，过时间我再部署。
```



### 2. Install Dependencies

```bash
# create env of pyhon 3.11. note that other python version may have bugs.
conda create -n spagent python=3.11
pip install -r requirements.txt
pip install "httpx[socks]"
```

### 3. Download Model Weights

Create checkpoints directories first:
```bash
mkdir -p checkpoints/{grounding_dino,depth_anything,pi3,sam2}

```

---

####  (1)  **Depth-Anything V2** - *Monocular Depth Estimation*

Choose one model based on your performance needs:

| Model | Size | Performance | Download |
|-------|------|-------------|----------|
| **Small** | ~25MB | Fastest | `cd checkpoints/depth_anything && wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth` |
| **Base** | ~100MB | Balanced | `cd checkpoints/depth_anything && wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth` |
| **Large** | ~350MB | Best Quality | `cd checkpoints/depth_anything && wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth` |

#### (2) **SAM2** - *Segment Anything Model 2*

Advanced image and video segmentation:

Option 1: Auto-download All Models
```bash
cd checkpoints/sam2
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh
```

Option 2: Manual Selection
| Model | Size | Performance | Download |
|-------|------|-------------|----------|
| **Large** | ~900MB | Best Quality | `cd checkpoints/sam2 && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` |
| **Base+** | ~230MB | Balanced | `cd checkpoints/sam2 && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt` |
| **Small** | ~50MB | Fastest | `cd checkpoints/sam2 && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt` |

#### (3) **Grounding DINO** - *Open-Vocabulary Object Detection*

Natural language object detection with text prompts:

```bash
cd checkpoints/grounding_dino
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```
- **Features**: Detect any object with text descriptions

#### (4) **Pi3** - *3D Reconstruction & Point Cloud Generation*

High-quality 3D scene reconstruction:

```bash
cd checkpoints/pi3
wget https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
```
- **Features**: Single image to 3D point cloud

#### (5) **Supervision & YOLO-World** - *Computer Vision Toolkit*

Real-time object detection and tracking:

- **Supervision models**: Auto-downloaded when running server/client
- **YOLO-World weights**: 
  ```bash
  python download_weights.py
  ```

---

### 4. Deploy External Experts
```
# prepare a GPU with >= 24G
apt-get install tmux # install tmux, use tmux to create two terminal

#deploy depth anything v2
python spagent/external_experts/Depth_AnythingV2/depth_server.py --checkpoint_path checkpoints/depth_anything/depth_anything_v2_vitb.pth --port 20019

# deploy sam2
python spagent/external_experts/SAM2/sam2_server.py --checkpoint_path checkpoints/sam2/sam2.1_b.pt --port 20020

```

### 4. Run Examples

```bash
# depth workflow
cd spagent
python examples/depth_workflow_example_usage.py
```

## 📊 Evaluation

### prepare BLINK dataset
```
dataset/
├── blink_data.jsonl          # BLINK数据集文件
└── BLINK/                    # 图像文件夹
    ├── 02bf928316cf55ddda3d9e938b89f7624db742364c4dd89eb4e3fddb55f51f9a.jpg
    ├── ebb9c1c41b0fe3ff0d65cfc4ef3e2d26e4aefba3be654213a2aeab56d6546443.jpg
    └── ...
```


### Evaluate gpt-4o-mini on BLINK
```
python spagent/examples/straight_evaluation_gpt.py
```




## 🔍 External Experts
| 工具名称 | 类型 | 主要功能 | 备注 |
| --- | --- | --- | --- |
| **Depth-AnythingV2** | 3D | 单目深度估计 | 将 2D 图像转为像素级深度图 |
| **SAM2** | 2D | 图像分割 | Segment Anything 模型第二代，交互式或自动分割 |
| **Supervision** | 2D | 视觉任务辅助工具库 | 用于目标检测、分割结果可视化和后处理 |
| **GroundingDINO** | 2D | 文本驱动目标检测 | 基于自然语言进行检测和框选 |
| **Pi3** | 3D | 点云生成与处理 | 将图像或多视角输入转为 3D 表示 |


## 🧠 Models

| models |
| --- |
| **GPT** |
| **QwenVL** |
| **Local vllm** |

## ✅ Todo





