# SPAgent 数据采集指南

[English](#english-version) | [中文](#中文版本)

---

## 中文版本

### 概述

SPAgent 提供了完整的训练数据采集功能，用于收集多模态大模型训练所需的数据。该功能可以：

- ✅ 记录每次推理的输入（图片+文本prompt）和输出（模型回复）
- ✅ 支持多轮推理，每轮推理作为独立样本
- ✅ 只保存成功的会话（多轮执行完且成功）
- ✅ 自动保存图片、文本和上下文信息
- ✅ 支持多种导出格式（JSONL、JSON、ShareGPT）

### 核心概念

#### 1. Session（会话）
一个完整的问答过程，可能包含多轮推理。只有成功的会话才会被保存。

#### 2. Sample（样本）
会话中的每次推理步骤，包含：
- 输入图片列表
- 完整的 prompt
- 模型的 response
- 上下文信息（工具调用、历史等）

#### 3. 成功判定
如果最终回复包含 `<answer></answer>` 标签，则认为会话成功。

### 快速开始

#### 1. 初始化 DataCollector

```python
from spagent.core.data_collector import DataCollector

collector = DataCollector(
    output_dir="training_data",  # 数据保存目录
    save_images=True,            # 是否复制图片
    auto_save=True               # 自动保存成功的会话
)
```

#### 2. 将 DataCollector 传递给 SPAgent

```python
from spagent.core.spagent import SPAgent
from spagent.core.model import Model

model = Model(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    model_path="/path/to/model"
)

agent = SPAgent(
    model=model,
    tools=[],  # 你的工具列表
    data_collector=collector  # 启用数据采集
)
```

#### 3. 运行推理（自动采集数据）

```python
result = agent.solve_problem(
    image_path="path/to/image.jpg",
    question="这张图片中有什么物体？",
    max_iterations=3
)
```

就这么简单！数据会自动采集和保存。

### 数据结构

#### 目录结构

```
training_data/
├── sessions/                    # 所有会话
│   ├── session_20250124_143022_abc123/
│   │   ├── session_metadata.json   # 会话元数据
│   │   ├── samples/                # 各个推理样本
│   │   │   ├── sample_1.json
│   │   │   ├── sample_2.json
│   │   │   └── sample_3.json
│   │   └── images/                 # 会话中的所有图片
│   │       ├── original_image.jpg
│   │       ├── depth_result.jpg
│   │       └── pi3_result.png
│   └── session_20250124_143145_def456/
│       └── ...
├── images/                      # (可选) 全局图片目录
└── statistics.json              # 采集统计信息
```

#### Session Metadata 格式

```json
{
  "session_id": "session_20250124_143022_abc123",
  "question": "这张图片中有多少个物体？",
  "original_images": ["path/to/image.jpg"],
  "success": true,
  "final_answer": "图片中有3个物体...",
  "error_message": null,
  "start_time": "2025-01-24T14:30:22",
  "end_time": "2025-01-24T14:30:45",
  "num_iterations": 3,
  "samples": [
    {
      "sample_id": "session_20250124_143022_abc123_iter_1",
      "iteration": 1,
      "images": ["path/to/image.jpg"],
      "prompt": "完整的输入prompt...",
      "response": "模型的完整回复...",
      "context": {
        "tool_calls_history": [],
        "tool_results_history": {},
        "additional_images_history": []
      },
      "timestamp": "2025-01-24T14:30:23"
    },
    {
      "sample_id": "session_20250124_143022_abc123_iter_2",
      "iteration": 2,
      "images": ["path/to/image.jpg", "path/to/depth.jpg"],
      "prompt": "继续分析的prompt...",
      "response": "第二轮回复...",
      "context": {
        "tool_calls_history": [{"name": "depth_tool", ...}],
        "tool_results_history": {"depth_tool": {...}},
        "additional_images_history": ["path/to/depth.jpg"]
      },
      "timestamp": "2025-01-24T14:30:35"
    }
  ],
  "metadata": {
    "iterations": 3,
    "num_tool_calls": 2,
    "used_tools": ["depth_tool_iter1", "pi3_tool_iter2"],
    "num_additional_images": 2
  }
}
```

### 导出训练数据

#### 1. JSONL 格式（推荐用于大规模训练）

```python
collector.export_for_training(
    output_file="training_data/train.jsonl",
    format="jsonl"
)
```

每行一个样本：
```jsonl
{"sample_id": "...", "iteration": 1, "images": [...], "prompt": "...", "response": "..."}
{"sample_id": "...", "iteration": 2, "images": [...], "prompt": "...", "response": "..."}
```

#### 2. JSON 格式

```python
collector.export_for_training(
    output_file="training_data/train.json",
    format="json"
)
```

所有样本在一个数组中。

#### 3. ShareGPT 格式（多模态训练）

```python
collector.export_for_training(
    output_file="training_data/train_sharegpt.json",
    format="sharegpt"
)
```

ShareGPT 格式示例：
```json
[
  {
    "id": "session_20250124_143022_abc123_iter_1",
    "images": ["path/to/image1.jpg", "path/to/image2.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "这张图片中有什么？"
      },
      {
        "from": "gpt",
        "value": "图片中包含..."
      }
    ]
  }
]
```

### 统计信息

```python
# 获取统计信息
stats = collector.get_statistics()
print(stats)
# {
#     "total_sessions": 100,
#     "successful_sessions": 85,
#     "failed_sessions": 15,
#     "total_samples": 255,
#     "success_rate": 0.85,
#     "avg_samples_per_success": 3.0
# }

# 保存统计信息到文件
collector.save_statistics()
```

### 高级用法

#### 手动控制会话

如果需要更精细的控制，可以手动管理会话：

```python
# 创建 collector，关闭自动保存
collector = DataCollector(
    output_dir="training_data",
    auto_save=False  # 手动控制
)

# 手动开始会话
session_id = collector.start_session(
    question="问题",
    image_paths=["image.jpg"]
)

# 记录推理步骤
collector.record_inference(
    iteration=1,
    images=["image.jpg"],
    prompt="完整的prompt",
    response="模型回复",
    context={"custom_key": "custom_value"}
)

# 手动结束会话
collector.end_session(
    success=True,
    final_answer="最终答案",
    metadata={"extra_info": "..."}
)
```

#### 自定义成功判定

默认情况下，包含 `<answer>` 标签的回复被视为成功。如果需要自定义：

```python
# 在 solve_problem 之后手动判定
result = agent.solve_problem(...)

# 自定义判定逻辑
is_success = your_custom_validation(result["answer"])

if agent.data_collector:
    if is_success:
        agent.data_collector.end_session(
            success=True,
            final_answer=result["answer"]
        )
    else:
        agent.data_collector.end_session(
            success=False,
            error_message="Custom validation failed"
        )
```

### 多轮推理说明

对于多轮推理（max_iterations > 1）：

1. **每轮都是独立样本**：每次模型推理都会生成一个独立的训练样本
2. **上下文累积**：后续轮次的 prompt 包含前面轮次的信息
3. **全部成功才保存**：只有整个会话成功完成，所有样本才会被保存

示例：如果设置 `max_iterations=3`，模型进行了3轮推理并最终成功：
- 会生成 3 个训练样本（或更多，如果有 final synthesis 步骤）
- 每个样本记录该轮的输入图片、prompt、response
- 所有 3 个样本都会被保存（因为整个会话成功）

### 最佳实践

#### 1. 批量采集

```python
test_cases = [
    {"image": "img1.jpg", "question": "问题1"},
    {"image": "img2.jpg", "question": "问题2"},
    # ...
]

collector = DataCollector("training_data")
agent = SPAgent(model, tools=[], data_collector=collector)

for case in test_cases:
    try:
        agent.solve_problem(
            image_path=case["image"],
            question=case["question"],
            max_iterations=3
        )
    except Exception as e:
        logger.error(f"Error: {e}")

# 保存统计
collector.save_statistics()
```

#### 2. 定期备份

```python
import shutil
from datetime import datetime

# 定期备份采集的数据
backup_dir = f"backup_{datetime.now().strftime('%Y%m%d')}"
shutil.copytree("training_data", backup_dir)
```

#### 3. 数据质量检查

```python
import json
from pathlib import Path

def check_data_quality(sessions_dir):
    """检查采集数据的质量"""
    for session_dir in Path(sessions_dir).iterdir():
        metadata_path = session_dir / "session_metadata.json"
        with open(metadata_path) as f:
            session = json.load(f)
        
        # 检查样本数量
        if len(session["samples"]) < 1:
            print(f"Warning: {session['session_id']} has no samples")
        
        # 检查答案长度
        if len(session["final_answer"]) < 10:
            print(f"Warning: {session['session_id']} has short answer")
        
        # 检查图片是否存在
        images_dir = session_dir / "images"
        if not images_dir.exists():
            print(f"Warning: {session['session_id']} has no images")

check_data_quality("training_data/sessions")
```

### 注意事项

1. **磁盘空间**：如果 `save_images=True`，会复制所有图片，注意磁盘空间
2. **并发安全**：当前实现不支持多进程并发写入同一个 collector
3. **失败会话**：失败的会话不会被保存，只有成功的会话才保存
4. **大规模采集**：建议分批采集，定期备份数据

---

## English Version

### Overview

SPAgent provides complete training data collection functionality for multimodal model training. Features include:

- ✅ Records input (images + text prompt) and output (model response) for each inference
- ✅ Supports multi-turn inference, each turn as an independent sample
- ✅ Only saves successful sessions (completed multi-turn execution)
- ✅ Automatically saves images, text, and context information
- ✅ Supports multiple export formats (JSONL, JSON, ShareGPT)

### Core Concepts

#### 1. Session
A complete Q&A process that may contain multiple inference rounds. Only successful sessions are saved.

#### 2. Sample
Each inference step in a session, containing:
- List of input images
- Complete prompt
- Model response
- Context information (tool calls, history, etc.)

#### 3. Success Criteria
A session is considered successful if the final response contains `<answer></answer>` tags.

### Quick Start

#### 1. Initialize DataCollector

```python
from spagent.core.data_collector import DataCollector

collector = DataCollector(
    output_dir="training_data",  # Data save directory
    save_images=True,            # Whether to copy images
    auto_save=True               # Auto-save successful sessions
)
```

#### 2. Pass DataCollector to SPAgent

```python
from spagent.core.spagent import SPAgent
from spagent.core.model import Model

model = Model(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    model_path="/path/to/model"
)

agent = SPAgent(
    model=model,
    tools=[],  # Your tool list
    data_collector=collector  # Enable data collection
)
```

#### 3. Run Inference (Data Collected Automatically)

```python
result = agent.solve_problem(
    image_path="path/to/image.jpg",
    question="What objects are in this image?",
    max_iterations=3
)
```

That's it! Data is automatically collected and saved.

### Data Structure

See the Chinese version above for detailed data structure and examples.

### Export Training Data

#### JSONL Format
```python
collector.export_for_training(
    output_file="training_data/train.jsonl",
    format="jsonl"
)
```

#### JSON Format
```python
collector.export_for_training(
    output_file="training_data/train.json",
    format="json"
)
```

#### ShareGPT Format
```python
collector.export_for_training(
    output_file="training_data/train_sharegpt.json",
    format="sharegpt"
)
```

### Statistics

```python
stats = collector.get_statistics()
print(stats)

collector.save_statistics()
```

### Advanced Usage

#### Manual Session Control

```python
collector = DataCollector(
    output_dir="training_data",
    auto_save=False  # Manual control
)

session_id = collector.start_session(
    question="Question",
    image_paths=["image.jpg"]
)

collector.record_inference(
    iteration=1,
    images=["image.jpg"],
    prompt="Full prompt",
    response="Model response",
    context={"custom_key": "custom_value"}
)

collector.end_session(
    success=True,
    final_answer="Final answer",
    metadata={"extra_info": "..."}
)
```

### Best Practices

See the Chinese version above for detailed best practices.

### Notes

1. **Disk Space**: If `save_images=True`, all images will be copied
2. **Concurrency**: Current implementation doesn't support multi-process concurrent writes
3. **Failed Sessions**: Failed sessions are not saved, only successful ones
4. **Large-scale Collection**: Recommend batch collection with periodic backups


