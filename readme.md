<div align="center">

<img src="assets/logo.png" alt="SPAgent Logo" width="300">

# 🌍 SPAgent: Agent in the Physical & Spatial World

### Think3D: Thinking with Space for Spatial Reasoning

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://zhangzaibin.github.io/spagent/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.13029-b31b1b.svg)](https://arxiv.org/abs/2601.13029)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/collections/jialianjie/spagent)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/jialianjie/Think3DQA)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

</div>

---

## 📌 Introduction

**SPAgent** is a foundation agent for perception, reasoning, and action in the physical and spatial world. It provides a modular, open-ended ecosystem of expert tools spanning 2D vision, 3D reconstruction, world modeling, video/image generation, and beyond — enabling grounded spatial reasoning and flexible interaction in complex real-world environments.

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Tool Reference](docs/Tool/TOOL_USING.md)** | External expert tools API and deployment guide |
| **[External Experts](docs/Tool/EXTERNAL_EXPERTS.md)** | Full list of supported expert models and default ports |
| **[Advanced Examples](docs/Examples/ADVANCED_EXAMPLES.md)** | `step()` API, AgentMemory, video/image gen, RL training, testing |
| **[Reproduce Results](docs/Evaluation/REPRODUCE.md)** | End-to-end recipe to reproduce benchmark numbers |
| **[Quick Eval](docs/Evaluation/QUICK_EVAL.md)** | `quick_eval.py` reference and shell-script shortcuts |
| **[Dataset Preparation](docs/Evaluation/EVALUATION.md)** | Per-benchmark dataset download and JSONL conversion |
| **[Adding New Tools](docs/ADDING_NEW_TOOLS.md)** | Guide for extending SPAgent with new expert tools |

## ✅ Features

- **Modular Tool System** — Mix and match any combination of expert tools
- **Dynamic Tool Management** — Add/remove tools at runtime
- **Parallel Tool Execution** — Automatic concurrent processing when possible
- **Multi-Image Analysis** — Handle single or multiple images seamlessly
- **Multiple Model Support** — GPT, Qwen, and local VLLM models
- **Customizable System Prompt** — Per-agent templates; built-in 3D spatial and general vision presets
- **Multimodal Agent Memory** — `AgentMemory` records every turn: text, images, tool calls, and results
- **Multi-turn Stateful Conversations** — Pass `AgentMemory` across `step()` calls; save/load sessions
- **Reinforcement Learning** — GRPO training support via ms-swift

## 🛠️ Installation & Setup

### 1. Environment

```bash
conda create -n spagent python=3.11
conda activate spagent

pip install -r requirements.txt
pip install "httpx[socks]"
```

### 2. API Keys

```bash
# OpenAI (also used by SoraTool)
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"

# Qwen / DashScope (apply at: https://bailian.console.aliyun.com)
export DASHSCOPE_API_KEY="your_api_key"

# Moondream (apply at: https://moondream.ai)
export MOONDREAM_API_KEY="your_api_key"

# Google Gemini (used by VeoTool)
export GOOGLE_API_KEY="your_google_api_key"
```

### 3. Deploy Expert Servers

See **[Tool Reference](docs/Tool/TOOL_USING.md)** for per-tool deployment instructions (Depth, SAM2, GroundingDINO, Pi3, Molmo2, OrientAnythingV2, WildDet3D, FlowSeek, PaddleOCR-VL, Sana, VACE, …).

## 🚀 Quick Start

### Basic Usage

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool

model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True),
]

agent = SPAgent(model=model, tools=tools)
result = agent.solve_problem("image.jpg", "Analyze depth and main objects in this image")
print(result['answer'])
```

### Multi-Tool Agent

```python
from spagent.tools import (
    DepthEstimationTool, SegmentationTool,
    ZoomObjectTool,      # GroundingDINO: crop close-up for attribute inspection
    LocalizeObjectTool,  # GroundingDINO: bbox annotation for spatial/counting
    SupervisionTool, YOLOETool,
)

tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True),
    ZoomObjectTool(use_mock=True),
    LocalizeObjectTool(use_mock=True),
    SupervisionTool(use_mock=True),
    YOLOETool(use_mock=True),
]

agent = SPAgent(model=GPTModel(model_name="gpt-4o-mini"), tools=tools, max_workers=4)
result = agent.solve_problem("image.jpg", "Comprehensively analyze this image")
print(result['answer'])
print(result['used_tools'])
```

### Real Expert Services

```python
tools = [
    DepthEstimationTool(use_mock=False, server_url="http://localhost:20019"),
    SegmentationTool(use_mock=False,    server_url="http://localhost:20020"),
    ZoomObjectTool(use_mock=False,      server_url="http://localhost:20022"),
    LocalizeObjectTool(use_mock=False,  server_url="http://localhost:20022"),
]
```

For **video generation**, **image generation**, **multi-turn AgentMemory**, **custom system prompts**, **RL training**, and detailed testing, see **[Advanced Examples](docs/Examples/ADVANCED_EXAMPLES.md)**.

## 📊 Evaluation

`scripts/quick_eval.py` is the unified entry point. It runs SPAgent over VLMEvalKit benchmarks and local datasets with automatic resuming and per-sample traces.

**60-second smoke test (no servers needed):**

```bash
python scripts/quick_eval.py --model gpt-4.1-mini --datasets MMStar --limit 5
```

**With tools:**

```bash
python scripts/quick_eval.py \
    --model gpt-4.1-mini --tools zoom localize \
    --datasets MMStar VStarBench --limit 50 \
    --detection-url http://localhost:20022
```

Full recipes → **[REPRODUCE.md](docs/Evaluation/REPRODUCE.md)** · **[QUICK_EVAL.md](docs/Evaluation/QUICK_EVAL.md)** · **[EVALUATION.md](docs/Evaluation/EVALUATION.md)**

## ⚠️ Important Notes

- **Python Version**: Python 3.11 recommended; other versions may have compatibility issues
- **GPU Memory**: Real mode requires ≥ 24 GB GPU memory
- **Concurrency**: Control parallel tool execution via `max_workers`

## 📝 Citation

```bibtex
@article{zhang2026think3d,
  title={Think3D: Thinking with Space for Spatial Reasoning},
  author={Zhang, Zaibin and Wu, Yuhan and Jia, Lianjie and Wang, Yifan and Zhang, Zhongbo and Li, Yijiang and Ran, Binghao and Zhang, Fuxi and Sun, Zhuohan and Yin, Zhenfei and others},
  journal={arXiv preprint arXiv:2601.13029},
  year={2026}
}
```

## ⭐ Star History

<div align="center">

**🌟 Thank you for your support! 🌟**

[![Star History Chart](https://api.star-history.com/svg?repos=zhangzaibin/spagent&type=Date)](https://star-history.com/#zhangzaibin/spagent&Date)

</div>
