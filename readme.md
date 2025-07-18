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

### 4. Examples (`spagent/examples/`)
启动示例脚本的入口，每个脚本就是一个使用教程，比如现在要使用depth_workflow
```
cd spagent
python examples/depth_workflow_example_usage.py
```

# Quick Start

## Prepare OpenAI API
```
export OPENAI_API_KEY="you api key here (我之前给过)"
export OPENAI_BASE_URL="http://35.220.164.252:3888/v1/" 
```


## Install
```
# 安装的包很少，主要是一些api的服务
pip install -r requirements.txt
pip install "httpx[socks]"
```

## Run
```
# depth workflow
cd spagent
python examples/depth_workflow_example_usage.py
```

# Workflow
https://b14esv5etcu.feishu.cn/docx/RvVFdkjiro52bnxgRVgcRXUqnpx#share-KQ73doO7IoSt4rx2gqIc6lXmnTf

# TODO
## External Experts
- [x] Depth-AnythingV2
- [ ] SAM2

## VLLM Models
- [x] GPT inference functions
- [ ] Additional model support

## Workflows
- [x] Add workflow examples
    - [x] Depth estimation workflow
    - [ ] Object detection workflow
- [ ] Add evaluation scripts
- [ ] Add documentation
