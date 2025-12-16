# 📌 Introduction

This repository provides **SPAgent** - a flexible and modular **Spatial Intelligence Agent** that integrates **agentic skills** into **multi-modal understanding** using external expert models and LLMs.

## 🆕 **SPAgent Features**

**SPAgent** replaces the old workflow system with a modern, modular architecture:

- ✅ **Modular Tool System** - Mix and match any combination of expert tools
- ✅ **Dynamic Tool Management** - Add/remove tools at runtime
- ✅ **Parallel Tool Execution** - Automatic concurrent processing when possible
- ✅ **Multi-Image Analysis** - Handle single or multiple images seamlessly
- ✅ **Multiple Model Support** - GPT, Qwen, and local VLLM models
- ✅ **Flexible Configuration** - Easy to customize and extend
- ✅ **Reinforcement Learning** - Support reinforcement learning

## 📂 Project Structure

| Module | Path | Description |
|--------|------|-------------|
| **SPAgent Core** | `spagent/core/` | Core agent architecture:<br>- SPAgent class and agent logic<br>- Tool base classes and registry<br>- Model base classes and wrappers<br>- Unified prompt system<br>- Data collection utilities |
| **Tools** | `spagent/tools/` | Modular expert tool implementations:<br>- DepthEstimationTool<br>- SegmentationTool<br>- ObjectDetectionTool<br>- SupervisionTool<br>- YOLOETool<br>- MoondreamTool<br>- Pi3Tool |
| **Models** | `spagent/models/` | Model wrappers for different backends:<br>- GPTModel (OpenAI API)<br>- QwenModel (DashScope API)<br>- QwenVLLMModel (local VLLM) |
| **External Experts** | `spagent/external_experts/` | Specialized expert models with client/server architecture:<br>- Depth Estimation (**Depth-AnythingV2**)<br>- Image/Video Segmentation (**SAM2**)<br>- Open-vocabulary Detection (**GroundingDINO**)<br>- Vision Language Model (**Moondream**)<br>- 3D Point Cloud Reconstruction (**Pi3**)<br>- YOLO-E Detection & Annotation (**Supervision**)<br>- Each includes client/server implementations and can run as external APIs |
| **VLLM Models** | `spagent/vllm_models/` | VLLM inference utilities and wrappers:<br>- GPT API wrapper<br>- Qwen API wrapper<br>- Local VLLM inference for Qwen models |
| **Examples** | `examples/` | Example scripts and usage tutorials:<br>- Evaluation scripts for datasets<br>- Quick start examples<br>- Tool definition examples |
| **Test** | `test/` | Test scripts for tools and models:<br>- Pi3 tool testing with video frame extraction<br>- Integration tests |
| **Train** | `train/` | Reinforcement learning training scripts:<br>- GRPO training configurations<br>- LoRA merge and model compression utilities<br>- System prompts for different training modes |

## 🔍 External Experts

| Tool Name | Type | Main Function | Default Port | Notes |
| --- | --- | --- | --- | --- |
| **Depth-AnythingV2** | 3D | Monocular Depth Estimation | 20019 | Convert 2D images to pixel-level depth maps |
| **SAM2** | 2D | Image Segmentation | 20020 | Segment Anything Model 2nd generation, interactive or automatic segmentation |
| **GroundingDINO** | 2D | Open-vocabulary Object Detection | 20022 | Detect arbitrary objects based on text descriptions |
| **Moondream** | 2D | Vision Language Model | 20024 | Small and efficient visual Q&A model, supports image description and Q&A |
| **Pi3** | 3D | 3D Point Cloud Reconstruction | 20030 | Generate 3D point clouds and multi-view rendered images from a single image |
| **Supervision** | 2D | Object Detection Annotation | - | YOLO models and visualization tools, used for result visualization and post-processing |

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
# OpenAI API
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"

# Qwen API (Apply at: https://bailian.console.aliyun.com)
export DASHSCOPE_API_KEY="your_api_key"

# Moondream API (Apply at: https://moondream.ai)
export MOONDREAM_API_KEY="your_api_key"

# Test API connection
python spagent/vllm_models/qwen.py
```

### 3. Deploy External Expert Services

For detailed external expert tools usage guide, please refer to: [External Experts Tool Usage Guide](docs/Tool/TOOL_USING.md)


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
    Pi3Tool                   # 3D reconstruction
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

### 3. Dynamic Tool Management

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

### 4. Multi-Image Analysis

```python
# Analyze multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
result = agent.solve_problem(
    image_paths, 
    "Compare the differences between these images, analyze depth changes and object distribution"
)
```

### 5. Image Dataset Evaluation

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
```

## 📚 Documentation

### Advanced Usage

For more advanced usage patterns, specialized agents, and tool mixing strategies, please refer to:
- **[Advanced Examples](docs/Examples/ADVANCED_EXAMPLES.md)** - Specialized agents, command line usage, and complex workflows
- **[Tool Reference](docs/Tool/TOOL_USING.md)** - Detailed tool API reference and deployment guide

## 🧪 Testing & Development

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

