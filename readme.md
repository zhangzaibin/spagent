# Introduction

This repo aims to integrate **agentic skills** into spatial intelligence.

这个项目的主要目的是将智能体技术和空间智能结合。

## Project Structure

The project is organized into three main modules:

### 1. External Experts (`spagent/external_experts/`)
This module contains specialized expert models for spatial intelligence tasks:
- **Depth Estimation**: Depth-AnythingV2 for depth prediction
- **Object Detection**: SAM2 for segmentation and detection
- These experts can be deployed as external APIs and integrated into the workflow

### 2. VLLM Models (`spagent/vllm_models/`)
This module contains VLLM inference functions and model wrappers:
- GPT model inference functions
- Model loading and serving utilities
- Provides standardized interfaces for large language model interactions

### 3. Workflows (`spagent/workflows/`)
This module orchestrates the complete spatial intelligence workflows:
- Imports and combines components from `vllm_models` and `external_experts`
- Defines end-to-end workflows for spatial reasoning tasks
- Handles data flow between different expert models and LLM components

# Install
```
pip install -r requirements.txt
pip install "httpx[socks]"
```

# Workflow
![workflow](assets/image.png)

# TODO
## External Experts
- [x] Depth-AnythingV2
- [ ] SAM2

## VLLM Models
- [x] GPT inference functions
- [ ] Additional model support

## Workflows
- [ ] Add workflow examples
    - [ ] Depth estimation workflow
    - [ ] Object detection workflow
    - [ ] Multi-modal reasoning workflow
- [ ] Add evaluation scripts
- [ ] Add workflow documentation
