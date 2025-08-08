# 📌 Introduction

This repository integrates **agentic skills** into **multi-modal understanding** using external expert models and LLMs.

---

## 📂 Project Structure

| Module | Path | Description |
|--------|------|-------------|
| **External Experts** | `spagent/external_experts/` | Specialized models for spatial intelligence:<br>- Depth Estimation (**Depth-AnythingV2**)<br>- Object Detection & Segmentation (**SAM2**)<br>- Can run as external APIs |
| **VLLM Models** | `spagent/vllm_models/` | VLLM inference functions & wrappers:<br>- GPT / QwenVL inference<br>- Model loading & serving utilities<br>- Unified API for LLM calls |
| **Workflows** | `spagent/workflows/` | Orchestrates complete workflows:<br>- Combines LLM + external experts<br>- Defines spatial reasoning pipelines<br>- Manages data flow |
| **Examples** | `spagent/examples/` | Example scripts, each showing a usage tutorial (e.g., `depth_workflow_example_usage.py`) |

---

## 🚀 Quick Start

### 1 Prepare APIs
```bash
# OpenAI API
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="http://35.220.164.252:3888/v1/"

# Qwen API (apply at https://bailian.console.aliyun.com)
export DASHSCOPE_API_KEY="your_api_key"

# Test Qwen API
python spagent/vllm_models/qwen.py

# prepare VLLM in the iiau A800 server
vllm serve /13693266743/models/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 20004 --served-model-name 'qwen-2' --enable-auto-tool-choice --tool-call-parser hermes

# Then, you can run
python spagent/vllm_models/qwen_vllm.py

# 现在我已经在A800上部署了，ip什么都是固定的，直接跑就行，24小时内都能用，过时间我再部署。
```





### 2 Install
```
# 安装的包很少，主要是一些api的服务
pip install -r requirements.txt
pip install "httpx[socks]"
```

### 3 Run
```
# depth workflow
cd spagent
python examples/depth_workflow_example_usage.py

```

## 📊 Evaluation

## prepare BLINK dataset
```
dataset/
├── blink_data.jsonl          # BLINK数据集文件
└── BLINK/                    # 图像文件夹
    ├── 02bf928316cf55ddda3d9e938b89f7624db742364c4dd89eb4e3fddb55f51f9a.jpg
    ├── ebb9c1c41b0fe3ff0d65cfc4ef3e2d26e4aefba3be654213a2aeab56d6546443.jpg
    └── ...
```


## Evaluate gpt-4o-mini on BLINK
```
python spagent/examples/straight_evaluation_gpt.py
```



## 📜  Workflow
[feishu link](https://b14esv5etcu.feishu.cn/docx/RvVFdkjiro52bnxgRVgcRXUqnpx#share-KQ73doO7IoSt4rx2gqIc6lXmnTf)


## ✅ TODO
## External Experts
- [x] Depth-AnythingV2
- [x] SAM2
- [x] Supervision
- [x] GroundingDINO
- [ ] MoonDream2

## Models
- [x] GPT
- [x] QwenVL
- [ ] Local vllm deployment

## Workflows
- [x] Add workflow examples
    - [x] Depth estimation workflow
    - [x] SAM2 workflow
    - [x] supervision workflow
- [x] Add evaluation scripts
    - [x] gpt
    - [x] depth workflow
- [ ] Add documentation
