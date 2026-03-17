<div align="center">

<img src="assets/logo.png.jpg" alt="SPAgent Logo" width="300">

# 🌍 SPAgent: Agent in the Physical & Spatial World

### Think3D: Thinking with Space for Spatial Reasoning

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://zhangzaibin.github.io/spagent/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.13029-b31b1b.svg)](https://arxiv.org/abs/2601.13029)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/collections/jialianjie/spagent)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/spagent)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

</div>

---

## 📌 Introduction

We introduce **SPAgent**, a foundation agent designed for perception, reasoning, and action in the physical and spatial world. SPAgent equips agents with an open-ended ecosystem of tools spanning 2D, 3D, world modeling, agentic search, social simulation, and beyond, enabling grounded understanding, spatial reasoning, and flexible interaction in complex real-world environments.

## 📋 Table of Contents

- [Documentation](#-documentation)
- [SPAgent Features](#-spagent-features)
- [Project Structure](#-project-structure)
- [External Experts](#-external-experts)
- [Installation & Setup](#️-installation--setup)
- [Quick Start](#-quick-start)
- [Testing & Development](#-testing--development)
- [Reinforcement Learning Training](#-reinforcement-learning-training)
- [Important Notes](#️-important-notes)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Tool Reference](docs/Tool/TOOL_USING.md)** | External expert tools API and deployment guide |
| **[Evaluation Guide](docs/Evaluation/EVALUATION.md)** | Dataset download and evaluation usage |
| **[Advanced Examples](docs/Examples/ADVANCED_EXAMPLES.md)** | Specialized agents, tool mixing, and RL training |

##  **SPAgent Features**

**SPAgent** provides a modern, modular architecture with the following features:

- ✅ **Modular Tool System** - Mix and match any combination of expert tools
- ✅ **Dynamic Tool Management** - Add/remove tools at runtime
- ✅ **Parallel Tool Execution** - Automatic concurrent processing when possible
- ✅ **Multi-Image Analysis** - Handle single or multiple images seamlessly
- ✅ **Multiple Model Support** - GPT, Qwen, and local VLLM models
- ✅ **Customizable System Prompt** - Per-agent prompt templates; built-in 3D spatial and general vision presets
- ✅ **Flexible Configuration** - Easy to customize and extend
- ✅ **Reinforcement Learning** - Support reinforcement learning

## 📂 Project Structure

| Module | Path | Description |
|--------|------|-------------|
| **SPAgent Core** | `spagent/core/` | Core agent architecture:<br>- SPAgent class and agent logic<br>- Tool base classes and registry<br>- Model base classes and wrappers<br>- Unified prompt system (built-in `SPATIAL_3D_SYSTEM_PROMPT` / `GENERAL_VISION_SYSTEM_PROMPT` templates, fully customisable via `system_prompt` parameter)<br>- Data collection utilities |
| **Tools** | `spagent/tools/` | Modular expert tool implementations:<br>- DepthEstimationTool<br>- SegmentationTool<br>- ObjectDetectionTool<br>- SupervisionTool<br>- YOLOETool<br>- MoondreamTool<br>- Pi3Tool<br>- Pi3XTool<br>- VGGTTool<br>- MapAnythingTool<br>- **VeoTool** (Google Veo, API-based)<br>- **SoraTool** (OpenAI Sora, API-based) |
| **Models** | `spagent/models/` | Model wrappers for different backends:<br>- GPTModel (OpenAI API)<br>- QwenModel (DashScope API)<br>- QwenVLLMModel (local VLLM) |
| **External Experts** | `spagent/external_experts/` | Specialized expert models with client/server architecture:<br>- Depth Estimation (**Depth-AnythingV2**)<br>- Image/Video Segmentation (**SAM2**)<br>- Open-vocabulary Detection (**GroundingDINO**)<br>- Vision Language Model (**Moondream**)<br>- 3D Point Cloud Reconstruction (**Pi3** / **Pi3X**)<br>- Multi-view 3D Reconstruction & Pose Estimation (**VGGT**)<br>- Dense 3D Reconstruction via Depth Estimation (**MapAnything**)<br>- YOLO-E Detection & Annotation (**Supervision**)<br>- Video Generation (**Veo** / **Sora**, API-based, no local server needed)<br>- Each includes client/server implementations and can run as external APIs |
| **VLLM Models** | `spagent/vllm_models/` | VLLM inference utilities and wrappers:<br>- GPT API wrapper<br>- Qwen API wrapper<br>- Local VLLM inference for Qwen models |
| **Examples** | `examples/` | Example scripts and usage tutorials:<br>- Evaluation scripts for datasets<br>- Quick start examples<br>- Tool definition examples |
| **Test** | `test/` | Test scripts for tools and models:<br>- Direct tool testing without LLM Agent (`test_tool.py`) — supports Pi3, Depth, Segmentation, Detection, Veo, Sora<br>- Pi3 tool testing with video frame extraction (`test_pi3_llm.py`)<br>- System prompt construction verification (`test_prompt.py`) |
| **Train** | `train/` | Reinforcement learning training scripts:<br>- GRPO training configurations<br>- LoRA merge and model compression utilities<br>- System prompts for different training modes |

## 🔍 External Experts

| Tool Name | Type | Main Function | Deployment | Notes |
| --- | --- | --- | --- | --- |
| **Depth-AnythingV2** | 2D | Monocular Depth Estimation | Local server (20019) | Convert 2D images to pixel-level depth maps |
| **SAM2** | 2D | Image Segmentation | Local server (20020) | Segment Anything Model 2nd generation, interactive or automatic segmentation |
| **GroundingDINO** | 2D | Open-vocabulary Object Detection | Local server (20022) | Detect arbitrary objects based on text descriptions |
| **Moondream** | 2D | Vision Language Model | Local server (20024) | Small and efficient visual Q&A model, supports image description and Q&A |
| **Pi3** | 3D | 3D Point Cloud Reconstruction | Local server (20030) | Generate 3D point clouds and multi-view rendered images from images |
| **Pi3X** | 3D | 3D Point Cloud Reconstruction (Enhanced) | Local server (20031) | Upgraded Pi3 with smoother point clouds, metric scale, and optional multimodal conditioning |
| **VGGT** | 3D | Multi-view 3D Point Cloud Reconstruction & Camera Pose Estimation | 20032 | Reconstruct 3D point clouds and estimate camera extrinsics/intrinsics from multiple images using [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B); supports both image lists and video frame input |
| **MapAnything** | 3D | Dense 3D Point Cloud Reconstruction via Depth Estimation | 20033 | Reconstruct dense 3D point clouds from multiple images using depth maps and camera poses with [facebook/map-anything](https://huggingface.co/facebook/map-anything); interface compatible with Pi3 for easy comparison |
| **Supervision** | 2D | Object Detection Annotation | Local | YOLO models and visualization tools, used for result visualization and post-processing |
| **Veo** | Video | Text/Image-to-Video Generation | API (no server) | Google Veo via Gemini API; requires `GOOGLE_API_KEY`; supports t2v and i2v |
| **Sora** | Video | Text/Image-to-Video Generation | API (no server) | OpenAI Sora; requires `OPENAI_API_KEY`; supports t2v, i2v, and 1:1 aspect ratio |

## 🛠️ Installation & Setup

### 1. Environment Setup

```bash
# Create Python 3.11 environment (other versions may have compatibility issues)
conda create -n spagent python=3.11
conda activate spagent

# Install dependencies
pip install -r requirements.txt
pip install "httpx[socks]"
```

### 2. API Configuration

```bash
# OpenAI API (also used by SoraTool)
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"

# Qwen API (Apply at: https://bailian.console.aliyun.com)
export DASHSCOPE_API_KEY="your_api_key"

# Moondream API (Apply at: https://moondream.ai)
export MOONDREAM_API_KEY="your_api_key"

# Google Gemini API (used by VeoTool)
export GOOGLE_API_KEY="your_google_api_key"
# or alternatively
export GCP_API_KEY="your_gcp_api_key"

# Test API connection
python spagent/vllm_models/qwen.py
```

### 3. Deploy External Expert Services

For detailed external expert tools usage guide, please refer to: **[External Experts Tool Usage Guide](docs/Tool/TOOL_USING.md)**


## 🚀 Quick Start

### 1. Basic Usage

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool

# Create model and tools
model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=True),    # Depth estimation
    SegmentationTool(use_mock=True)        # Image segmentation
]

# Create agent
agent = SPAgent(model=model, tools=tools)

# Solve problem
result = agent.solve_problem("image.jpg", "Analyze the depth relationships and main objects in this image")
print(result['answer'])
```

### 2. Multi-Tool Usage

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import (
    DepthEstimationTool,      # Depth estimation
    SegmentationTool,         # Image segmentation  
    ObjectDetectionTool,      # Object detection
    SupervisionTool,          # Supervision tool
    YOLOETool,                # YOLO-E detection
    MoondreamTool,            # Visual Q&A
    Pi3Tool,                  # 3D reconstruction
    Pi3XTool                  # 3D reconstruction (enhanced)
)

# Create full-featured agent
model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True),
    ObjectDetectionTool(use_mock=True),
    SupervisionTool(use_mock=True),
    YOLOETool(use_mock=True)
]

agent = SPAgent(model=model, tools=tools, max_workers=4)

# Complex problem analysis
result = agent.solve_problem(
    "image.jpg", 
    "Comprehensively analyze this image: identify all objects, analyze depth relationships, and segment important regions"
)

print(f"Answer: {result['answer']}")
print(f"Used tools: {result['used_tools']}")
print(f"Additional images: {result['additional_images']}")
```

### 3. Custom System Prompt

`SPAgent` accepts an optional `system_prompt` parameter. Pass one of the built-in
templates or supply your own string. A `{tools_json}` placeholder is replaced
automatically with the live tool schema; if omitted, the tools block is appended.

```python
from spagent.core.prompts import GENERAL_VISION_SYSTEM_PROMPT, SPATIAL_3D_SYSTEM_PROMPT

# General vision agent (GroundingDINO + SAM2, no 3D instructions)
agent = SPAgent(
    model=GPTModel(model_name="gpt-4o"),
    tools=[ObjectDetectionTool(...), SegmentationTool(...)],
    system_prompt=GENERAL_VISION_SYSTEM_PROMPT,
)

# 3D spatial agent (default, same as omitting system_prompt)
agent = SPAgent(model=..., tools=[Pi3XTool(...)], system_prompt=SPATIAL_3D_SYSTEM_PROMPT)

# Fully custom prompt
agent = SPAgent(model=..., tools=tools,
                system_prompt="You are a specialist.\n<tools>\n{tools_json}\n</tools>\n...")
```

The same parameter is forwarded by `evaluate_tool_config`:

```python
evaluate_tool_config(..., system_prompt=GENERAL_VISION_SYSTEM_PROMPT)
```

### 4. Dynamic Tool Management

```python
# Start with a basic agent
agent = SPAgent(model=GPTModel())

# Dynamically add tools
agent.add_tool(DepthEstimationTool(use_mock=True))
agent.add_tool(SegmentationTool(use_mock=True))

# View current tools
print(f"Current tools: {agent.list_tools()}")

# Remove unnecessary tools
agent.remove_tool("depth_estimation_tool")

# Change model
from spagent.models import QwenModel
agent.set_model(QwenModel(model_name="qwen2.5-vl-7b-instruct"))
```

### 5. Multi-Image Analysis

```python
# Analyze multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
result = agent.solve_problem(
    image_paths, 
    "Compare the differences between these images, analyze depth changes and object distribution"
)
```

### 6. Video Generation with Veo / Sora

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import VeoTool, SoraTool

model = GPTModel(model_name="gpt-4o")

# Text-to-video with Google Veo
agent = SPAgent(model=model, tools=[VeoTool()])
result = agent.solve_problem(
    "dummy",
    "Generate a video of a golden retriever running on a beach at sunset",
    video_num_frames=4   # frames sampled from the output video for evaluation
)
print(result['answer'])

# Image-to-video with OpenAI Sora
agent = SPAgent(model=model, tools=[SoraTool()])
result = agent.solve_problem(
    "assets/dog.jpeg",
    "Make the dog start running across the field",
    video_num_frames=4
)
print(result['answer'])
```

### 7. Image Dataset Evaluation

For detailed image dataset evaluation usage guide, please refer to: **[Image Dataset Evaluation Usage Guide](docs/Evaluation/EVALUATION.md)**

**Basic Evaluation Commands:**

```bash
# Normal evaluation
python examples/evaluation/evaluate_img.py --data_path path/to/json --model gpt/qwen3-vl-4b --max_samples 15 --max_iterations 3 --task "your task name"

# Evaluation without tools (clean version)
python examples/evaluation/evaluate_img_wotools.py --data_path path/to/json --model gpt/qwen3-vl-4b --max_samples 15 --max_iterations 1 --task "your task name"

# Collect data for SFT
python examples/evaluation/evaluate_img_with_data_collection.py --data_path path/to/json --model gpt/qwen3-vl-4b --max_samples 15 --max_iterations 3 --enable_data_collection

# Example: Evaluate on BLINK dataset
python examples/evaluation/evaluate_img.py --data_path dataset/Multi-view_Reasoning_BLINK_subset.jsonl --max_samples 20 --model gpt-4.1 --max_iterations 4


# Evaluation examples with the video generation tool. 

# Evaluate Veo on a custom prompt dataset
python examples/evaluation/evaluate_veo.py \
    --data_path dataset/veo_eval_data.jsonl \
    --model gpt-4o \
    --video_num_frames 4


# Evaluate Veo with mock service (no API key needed)
python examples/evaluation/evaluate_veo.py \
    --data_path dataset/veo_eval_data.jsonl \
    --use_mock --max_samples 5

# Evaluate Sora on a custom prompt dataset
python examples/evaluation/evaluate_sora.py \
    --data_path dataset/sora_eval_data.jsonl \
    --model gpt-4o \
    --video_num_frames 4

# Evaluate Sora with mock service
python examples/evaluation/evaluate_sora.py \
    --data_path dataset/sora_eval_data.jsonl \
    --use_mock --max_samples 5
```


For more advanced usage patterns, specialized agents, tool mixing strategies, video analysis, and reinforcement learning training, please refer to: **[Advanced Examples](docs/Examples/ADVANCED_EXAMPLES.md)**

## 🧪 Testing & Development

### Direct Tool Testing (without LLM Agent)

Use `test/test_tool.py` to directly test any external expert tool — no LLM or Agent involved. This is useful for verifying tool deployment, debugging, and development.

```bash
# Test Pi3: input an image and render from a custom angle
python test/test_tool.py --tool pi3 --image assets/dog.jpeg --azimuth 45 --elevation -30

# Test Pi3X (enhanced version with smoother point clouds and metric scale)
python test/test_tool.py --tool pi3x --image assets/dog.jpeg --azimuth 45 --elevation -30

# Specify a custom server address
python test/test_tool.py --tool pi3 --image assets/dog.jpeg --azimuth 45 --elevation -30 --server_url http://10.7.8.94:20030

# Use first-person camera view mode
python test/test_tool.py --tool pi3 --image assets/dog.jpeg --azimuth 90 --elevation 0 --camera_view

# Multiple input images
python test/test_tool.py --tool pi3x --image img1.jpg img2.jpg --azimuth 45 --elevation -30

# Test Veo (text-to-video, requires GOOGLE_API_KEY)
python test/test_tool.py --tool veo \
    --image dummy \
    --prompt "A golden retriever running on a beach at sunset" \
    --duration 8

# Test Veo (image-to-video)
python test/test_tool.py --tool veo \
    --image assets/dog.jpeg \
    --prompt "The dog starts running across the field" \
    --duration 8

# Test Veo with mock service (no API key needed)
python test/test_tool.py --tool veo --image dummy --prompt "test" --use_mock

# Test Sora (text-to-video, requires OPENAI_API_KEY)
python test/test_tool.py --tool sora \
    --image dummy \
    --prompt "A timelapse of a city skyline at night" \
    --duration 5 \
    --resolution 1280x720

# Test Sora (image-to-video)
python test/test_tool.py --tool sora \
    --image assets/dog.jpeg \
    --prompt "The dog starts running" \
    --duration 5

# Test Sora with mock service (no API key needed)
python test/test_tool.py --tool sora --image dummy --prompt "test" --use_mock
```

You can also call the test function directly in Python:

```python
from test.test_tool import test_pi3

output_path = test_pi3(
    image_paths=["assets/dog.jpeg"],
    azimuth_angle=45,
    elevation_angle=-30,
    server_url="http://localhost:20030"
)
print(f"Rendered image saved to: {output_path}")
```

| Test Script | Description |
|-------------|-------------|
| `test/test_tool.py` | Direct tool testing without LLM Agent (Pi3, Depth, Segmentation, Detection, Veo, Sora) |
| `test/test_pi3_llm.py` | Pi3 integration testing through Agent + LLM |
| `test/test_prompt.py` | Verify system prompt construction — no server or API key needed |

```bash
python test/test_prompt.py                  # run all cases
python test/test_prompt.py --case general   # general vision prompt
python test/test_prompt.py --case 3d        # 3D spatial prompt
```

### Real Service Mode
```python
# Use real deployed services
tools = [
    DepthEstimationTool(use_mock=False, server_url="http://localhost:20019"),
    SegmentationTool(use_mock=False, server_url="http://localhost:20020"),
    ObjectDetectionTool(use_mock=False, server_url="http://localhost:30969")
]
```

### Video Analysis Testing

Test Pi3 tool with video frame extraction:

```python
# test/test_pi3_llm.py - Video analysis with Pi3 3D reconstruction
from spagent.core.spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import Pi3Tool

# Configure model and Pi3 tool
model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)
tools = [Pi3Tool(use_mock=False, server_url="http://localhost:20030")]

agent = SPAgent(model=model, tools=tools, max_workers=4)

# Analyze video frames
result = agent.solve_problem(
    frame_paths,  # List of extracted frame paths
    "Based on these frames from a video, please answer: Which direction did the object move?",
    video_path="path/to/video.mp4",  # Optional: for Pi3 to extract more frames
    pi3_num_frames=50  # Number of frames for Pi3 analysis
)
```

## 🎯 Reinforcement Learning Training

SPAgent supports GRPO (Group Relative Policy Optimization) reinforcement learning training using [ms-swift](https://github.com/modelscope/ms-swift).

### Training Scripts

| Script | Description |
|--------|-------------|
| `train/train_grpo.sh` | Standard GRPO training with tool calling |
| `train/train_grpo_all_angles.sh` | GRPO training with all angle combinations |
| `train/train_grpo_notool.sh` | GRPO training without tool calling (baseline) |
| `train/merge_lora.sh` | Merge LoRA adapters into base model |
| `train/compress_model.sh` | Compress trained model checkpoints |

### Basic Training Command

```bash
# Standard GRPO training
cd train
bash train_grpo.sh

# Training without tools (baseline)
bash train_grpo_notool.sh

# Training with all angle combinations
bash train_grpo_all_angles.sh
```

### Key Training Parameters

```bash
swift rlhf \
    --rlhf_type grpo \
    --model path/to/Qwen3-VL-4B-Instruct \
    --external_plugins plugin/plugin.py \
    --multi_turn_scheduler spagent_tool_call_scheduler \
    --max_turns 3 \
    --reward_funcs external_r1v_acc external_multiturn_format \
    --reward_weights 1.0 1.0 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset path/to/training_data.jsonl \
    --num_generations 8 \
    --temperature 0.6 \
    --deepspeed zero2 \
    --output_dir output/grpo_experiment
```

### Post-Training

```bash
# Merge LoRA weights into base model
swift export \
    --adapters output/grpo_xxx/checkpoint-xxx \
    --merge_lora true

# Compress model checkpoint for deployment
bash train/compress_model.sh
```

## ⚠️ Important Notes

1. **Python Version**: Python 3.11 is recommended, other versions may have compatibility issues
2. **Memory Requirements**: Real mode requires GPU memory >= 24GB
3. **Network Configuration**: Ensure API keys and server addresses are configured correctly
4. **Concurrency Control**: Control the number of parallel tools via the `max_workers` parameter

## 📝 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{zhang2026think3d,
  title={Think3D: Thinking with Space for Spatial Reasoning},
  author={Zhang, Zaibin and Wu, Yuhan and Jia, Lianjie and Wang, Yifan and Zhang, Zhongbo and Li, Yijiang and Ran, Binghao and Zhang, Fuxi and Sun, Zhuohan and Yin, Zhenfei and others},
  journal={arXiv preprint arXiv:2601.13029},
  year={2026}
}
```

## ⭐ Star History

If you find **SPAgent** useful for your research or projects, please consider giving us a ⭐ star! Your support helps us continue improving and maintaining this project.

<div align="center">

**🌟 Thank you for your support! 🌟**

[![Star History Chart](https://api.star-history.com/svg?repos=zhangzaibin/spagent&type=Date)](https://star-history.com/#zhangzaibin/spagent&Date)

</div>
