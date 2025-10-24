# 简洁数据格式使用指南

## 问题说明

原始导出格式包含了太多系统提示、工具定义等信息，不利于训练。

**原始格式问题**：
```json
{
  "from": "human",
  "value": "You are a helpful assistant...\n\n# Tools\n<tools>...(大量工具定义)...\n\nQuestion: 实际问题在这里..."
}
```

## ✅ 新增简洁格式

现在提供 **4 种导出格式**：

### 1. **train_simple.jsonl** (推荐！最简洁)

每行一个样本，只包含核心信息：

```jsonl
{"id":"session_xxx_iter_1","images":["img1.jpg","img2.jpg"],"question":"Images to analyze:\n- img1.jpg\n- img2.jpg\n\nQuestion:\nThe images are frames from a video...","answer":"<think>分析过程...</think>\n<tool_call>工具调用...</tool_call>","iteration":1,"context":{...}}
```

**格式说明**：
- `id`: 样本唯一标识
- `images`: 图片路径列表
- `question`: 提取的核心问题（不含系统提示和工具定义）
- `answer`: 模型完整回复
- `iteration`: 第几轮推理
- `context`: 上下文信息（工具调用历史等）

### 2. **train_full.jsonl** (完整原始数据)

包含完整 prompt（含系统提示、工具定义）：

```jsonl
{"sample_id":"session_xxx_iter_1","iteration":1,"images":[...],"prompt":"完整的系统提示+工具定义+问题...","response":"模型回复...","context":{...}}
```

### 3. **train_sharegpt_simple.json** (ShareGPT 简洁版)

```json
[
  {
    "id": "session_xxx_iter_1",
    "images": ["img1.jpg", "img2.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "Images to analyze:\n- img1.jpg\n- img2.jpg\n\nQuestion:\n这是核心问题..."
      },
      {
        "from": "gpt",
        "value": "<think>思考...</think>\n<tool_call>工具调用...</tool_call>"
      }
    ]
  }
]
```

### 4. **train_sharegpt_full.json** (ShareGPT 完整版)

包含完整系统提示，与原来的格式相同。

## 使用方法

### 自动生成（运行评估时）

```bash
python examples/evaluation/evaluate_img.py \
    --data_path dataset/data.jsonl \
    --max_samples 10 \
    --model gpt-4o \
    --max_iterations 3 \
    --enable_data_collection
```

会自动生成所有 4 种格式：
```
training_data/xxx/
├── train_simple.jsonl           ← 推荐！最简洁
├── train_full.jsonl             ← 完整原始数据
├── train_sharegpt_simple.json   ← ShareGPT 简洁版
└── train_sharegpt_full.json     ← ShareGPT 完整版
```

### 手动导出

```python
from spagent.core import DataCollector

collector = DataCollector("training_data")

# 方法 1: 简洁格式 (推荐！)
collector.export_for_training(
    output_file="train_simple.jsonl",
    format="simple"  # 核心：使用 simple 格式
)

# 方法 2: ShareGPT 简洁版
collector.export_for_training(
    output_file="train_sharegpt_simple.json",
    format="sharegpt",
    simple_format=True  # 核心：设置 simple_format=True
)

# 方法 3: ShareGPT 完整版（原格式）
collector.export_for_training(
    output_file="train_sharegpt_full.json",
    format="sharegpt",
    simple_format=False  # 或者不设置，默认为 False
)
```

## 格式对比

### 第1轮推理 - 初始问题

#### Simple 格式：
```json
{
  "id": "session_xxx_iter_1",
  "images": ["img1.jpg", "img2.jpg"],
  "question": "Images to analyze:\n- img1.jpg\n- img2.jpg\n\nQuestion:\nThe images are frames from a video. Is the camera moving left or right?",
  "answer": "<think>需要分析相机运动...</think>\n<tool_call>{\"name\": \"pi3_tool\", \"arguments\": {...}}</tool_call>",
  "iteration": 1
}
```

#### Full 格式（原始）：
```json
{
  "sample_id": "session_xxx_iter_1",
  "prompt": "You are a helpful assistant...\n\n# Tools\n<tools>[...几百行工具定义...]\n\nPlease analyze...\n\nQuestion:\nThe images are frames from a video...",
  "response": "<think>...</think>\n<tool_call>...</tool_call>",
  "iteration": 1
}
```

### 第2轮推理 - 基于工具结果继续

#### Simple 格式：
```json
{
  "id": "session_xxx_iter_2",
  "images": ["pi3_result.png"],
  "question": "Original Question:\nThe images are frames from a video. Is the camera moving left or right?\n\nPrevious Tool Results:\n- pi3_tool_iter1: Successfully executed\n  └─ Viewing angle: azimuth=0.0°, elevation=0.0°\n\nAvailable Images:\n- outputs/pi3_result.png\n\nPlease continue your analysis.",
  "answer": "<answer>Based on the 3D reconstruction, the camera is moving left...</answer>",
  "iteration": 2
}
```

## 训练时如何使用

### 使用 Simple 格式训练

```python
import json

# 读取简洁格式数据
with open('train_simple.jsonl') as f:
    for line in f:
        sample = json.loads(line)
        
        images = sample['images']      # 图片列表
        question = sample['question']  # 核心问题
        answer = sample['answer']      # 模型回答
        iteration = sample['iteration'] # 第几轮
        
        # 构建训练样本
        # 输入 = 图片 + 问题
        # 输出 = 回答
        training_sample = {
            "input": {
                "images": images,
                "text": question
            },
            "output": answer
        }
```

### 使用 ShareGPT Simple 格式训练

```python
import json

# 读取 ShareGPT 简洁格式
with open('train_sharegpt_simple.json') as f:
    data = json.load(f)

for sample in data:
    images = sample['images']
    conversations = sample['conversations']
    
    # conversations[0]['value'] = 核心问题（无系统提示）
    # conversations[1]['value'] = 模型回答
    
    human_input = conversations[0]['value']
    gpt_output = conversations[1]['value']
    
    # 用于训练
    ...
```

## 实际示例

运行后查看生成的文件：

```bash
# 查看简洁格式（每行一个样本）
head -n 1 training_data/*/train_simple.jsonl | jq

# 输出示例：
{
  "id": "session_20251024_161042_e3e520c6_iter_1",
  "images": [
    "/media/zzb/AI_save/zzb/spagent/dataset/own_dataset/gay_images/1_1.jpg",
    "/media/zzb/AI_save/zzb/spagent/dataset/own_dataset/gay_images/1_2.jpg"
  ],
  "question": "Images to analyze:\n- /media/.../1_1.jpg\n- /media/.../1_2.jpg\n\nQuestion:\nThe images are frames from a video. The video is shooting a static scene. The camera is either moving clockwise (left) or counter-clockwise (right) around the object. The first image is from the beginning of the video and the second image is from the end. Is the camera moving left or right when shooting the video? Select from the following options.\n(A) left\n(B) right\n\nThink step by step to analyze the question and provide a detailed answer.",
  "answer": "<think>\nTo determine whether the camera moved left (clockwise) or right (counterclockwise) around the object, I need to analyze the change in perspective between the two images...\n</think>\n<tool_call>\n{\"name\": \"pi3_tool\", \"arguments\": {\"image_path\": [...], \"azimuth_angle\": 0, \"elevation_angle\": 0}}\n</tool_call>",
  "iteration": 1,
  "context": {
    "tool_calls_history": [],
    "tool_results_history": {},
    "additional_images_history": []
  }
}
```

## 多轮推理示例

一个完整的多轮会话会生成多个样本：

### 样本 1（第1轮）：
```json
{
  "id": "session_xxx_iter_1",
  "images": ["img1.jpg", "img2.jpg"],
  "question": "原始问题...",
  "answer": "<think>思考...</think>\n<tool_call>调用工具...</tool_call>",
  "iteration": 1
}
```

### 样本 2（第2轮，基于工具结果）：
```json
{
  "id": "session_xxx_iter_2",
  "images": ["img1.jpg", "img2.jpg", "pi3_result.png"],
  "question": "Original Question: 原始问题...\n\nPrevious Tool Results:\n- pi3_tool: 成功\n\nPlease continue...",
  "answer": "<answer>基于3D重建的最终答案...</answer>",
  "iteration": 2
}
```

## 优势总结

| 格式 | 优点 | 适用场景 |
|------|------|----------|
| **train_simple.jsonl** | 最简洁，只含核心问答 | 微调多模态模型，只需要问答对 |
| train_full.jsonl | 包含完整提示词 | 需要学习工具调用的完整上下文 |
| **train_sharegpt_simple.json** | 简洁对话格式 | LLaMA、Qwen 等对话模型训练 |
| train_sharegpt_full.json | 完整对话格式 | 需要系统提示的对话训练 |

## 推荐使用

**对于大多数多模态模型训练**，推荐使用：
1. **train_simple.jsonl** - 最简洁直接
2. **train_sharegpt_simple.json** - 如果你的训练框架使用 ShareGPT 格式

这两种格式去掉了冗余的系统提示和工具定义，只保留核心的：
- 输入：图片 + 问题 + 上一轮结果（如果有）
- 输出：模型回答

## 立即使用

```bash
# 重新运行评估，会自动生成 4 种格式
python examples/evaluation/evaluate_img.py \
    --data_path dataset/data.jsonl \
    --max_samples 10 \
    --model gpt-4o \
    --max_iterations 3 \
    --enable_data_collection

# 检查生成的简洁格式
head -n 1 training_data/*/train_simple.jsonl | jq .question
head -n 1 training_data/*/train_simple.jsonl | jq .answer
```

现在的数据格式直接、简洁，非常适合训练！🎉


