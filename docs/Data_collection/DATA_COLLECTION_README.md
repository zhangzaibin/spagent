# SPAgent 数据采集功能 - 完整指南

## 📋 目录

1. [功能概述](#功能概述)
2. [快速开始](#快速开始)
3. [完整实现说明](#完整实现说明)
4. [使用场景](#使用场景)
5. [文件结构](#文件结构)
6. [示例代码](#示例代码)

---

## 功能概述

SPAgent 现已支持完整的训练数据采集功能，用于收集多模态大模型训练数据。

### ✨ 核心特性

- ✅ **自动采集**：每次推理自动记录输入（图片+prompt）和输出（response）
- ✅ **多轮支持**：每轮推理作为独立样本，完整记录推理链
- ✅ **智能过滤**：只保存成功的会话（有完整答案的样本）
- ✅ **上下文完整**：保存工具调用、结果、额外图片等所有信息
- ✅ **多种格式**：支持 JSONL、JSON、ShareGPT 等训练格式
- ✅ **统计监控**：实时统计采集进度和成功率

### 🎯 设计目标

您提出的需求：
> "每次推理都要将其保存下来，包括图片文本等所有 memory 和输出的 language。只要成功的样本。多轮执行完成功了，所有的数据样本就都是成功的正样本。"

✅ **已完整实现**

---

## 快速开始

### 方式一：基础使用（3 行代码）

```python
from spagent.core import SPAgent, Model, DataCollector

# 1. 创建 DataCollector
collector = DataCollector(
    output_dir="training_data",
    save_images=True,
    auto_save=True
)

# 2. 传递给 SPAgent
agent = SPAgent(
    model=Model("Qwen/Qwen2-VL-7B-Instruct", "/path/to/model"),
    tools=[],  # 你的工具列表
    data_collector=collector
)

# 3. 正常使用（自动采集数据）
result = agent.solve_problem(
    image_path="image.jpg",
    question="这张图片中有什么？",
    max_iterations=3
)

# 数据已自动保存到 training_data/
```

### 方式二：集成到评估脚本

```bash
# 运行评估并同时采集训练数据
python examples/evaluation/evaluate_img_with_data_collection.py \
    --data_path dataset/data.jsonl \
    --max_samples 100 \
    --model gpt-4o \
    --enable_data_collection \
    --max_iterations 3
```

详见：[examples/evaluation/HOW_TO_ADD_DATA_COLLECTION.md](examples/evaluation/HOW_TO_ADD_DATA_COLLECTION.md)

---

## 完整实现说明

### 1. 核心模块

#### `spagent/core/data_collector.py` (新增)

实现了三个核心类：

1. **InferenceSample**：单次推理样本
   - 记录图片、prompt、response、上下文
   - 每轮推理生成一个样本

2. **SessionData**：完整会话数据
   - 包含多个 InferenceSample
   - 记录会话级别的元数据

3. **DataCollector**：数据收集器
   - 管理会话和样本
   - 导出多种格式
   - 提供统计功能

#### `spagent/core/spagent.py` (已修改)

集成了 DataCollector：
- 在 `solve_problem` 开始时启动会话
- 每次推理后记录样本
- 结束时判定成功并保存

### 2. 数据采集流程

```
用户调用 solve_problem()
    ↓
DataCollector.start_session()  # 开始会话
    ↓
┌─────────────────────────────┐
│  多轮推理循环                │
│  ┌─────────────────────┐    │
│  │ 模型推理            │    │
│  │  ↓                  │    │
│  │ record_inference()  │    │  # 记录本轮样本
│  │  ↓                  │    │
│  │ 工具调用（可选）    │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
    ↓
提取答案，判定成功
    ↓
DataCollector.end_session()    # 结束会话
    ↓
如果成功：保存所有样本
如果失败：丢弃所有样本
```

### 3. 成功判定逻辑

```python
def _extract_answer(self, response: str) -> Optional[str]:
    """提取 <answer> 标签中的内容"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    return match.group(1).strip() if match else None

# 在 solve_problem 结尾
extracted_answer = self._extract_answer(final_response)
success = extracted_answer is not None  # 有 answer 标签即成功
```

**关键点**：
- ✅ 只要模型返回 `<answer>` 标签，就认为成功
- ✅ 无论预测是否正确，都保存（可用于分析和负采样）
- ✅ 整个会话成功，所有轮次的样本都保存

### 4. 多轮推理示例

假设 `max_iterations=3`，实际执行了 3 轮推理：

**会话流程**：
```
Iteration 1: 初始推理
  → Sample 1: {images: [img1.jpg], prompt: "...", response: "..."}

Iteration 2: 调用工具后继续推理
  → Sample 2: {images: [img1.jpg, depth.jpg], prompt: "...", response: "..."}

Iteration 3: 进一步分析
  → Sample 3: {images: [img1.jpg, depth.jpg, pi3.png], prompt: "...", response: "..."}

Final Synthesis: 综合最终答案
  → Sample 4: {images: [...], prompt: "...", response: "<answer>...</answer>"}
```

**结果**：
- 如果 Final Synthesis 包含 `<answer>` 标签 → **成功**，保存所有 4 个样本
- 如果没有 `<answer>` 标签 → **失败**，不保存任何样本

---

## 使用场景

### 场景 1：批量采集训练数据

```python
from spagent.core import SPAgent, Model, DataCollector

# 初始化
collector = DataCollector("training_data")
agent = SPAgent(model, tools, data_collector=collector)

# 批量处理
test_cases = [
    {"image": "img1.jpg", "question": "问题1"},
    {"image": "img2.jpg", "question": "问题2"},
    # ... 更多样本
]

for case in test_cases:
    try:
        agent.solve_problem(
            case["image"],
            case["question"],
            max_iterations=3
        )
    except Exception as e:
        print(f"Error: {e}")

# 导出数据
collector.export_for_training("train.jsonl", format="jsonl")
collector.save_statistics()

stats = collector.get_statistics()
print(f"成功率: {stats['success_rate']:.1%}")
print(f"总样本: {stats['total_samples']}")
```

### 场景 2：评估时同时采集

```bash
# 在评估的同时采集训练数据
python examples/evaluation/evaluate_img_with_data_collection.py \
    --data_path dataset/cvbench_data.jsonl \
    --max_samples 1000 \
    --model gpt-4o \
    --enable_data_collection \
    --max_iterations 3
```

输出：
- 评估结果：`spagent_evaluation_results_*.json`
- 训练数据：`training_data/depth_detection_segmentation_*/`

### 场景 3：手动控制采集

```python
# 高级：完全手动控制
collector = DataCollector("training_data", auto_save=False)

session_id = collector.start_session(question, images)

# 自定义推理流程
for iteration in range(3):
    response = custom_inference(...)
    
    collector.record_inference(
        iteration=iteration,
        images=current_images,
        prompt=prompt,
        response=response,
        context=custom_context
    )

# 自定义成功判定
if custom_success_criteria(response):
    collector.end_session(success=True, final_answer=response)
else:
    collector.end_session(success=False, error_message="未达标")
```

---

## 文件结构

### 新增文件

```
spagent/
├── core/
│   ├── data_collector.py          # 数据采集核心模块
│   ├── spagent.py                 # 已修改：集成 DataCollector
│   └── __init__.py                # 已修改：导出 DataCollector
│
├── examples/
│   ├── data_collection_example.py              # 详细示例
│   ├── quick_start_data_collection.py          # 快速开始示例
│   └── evaluation/
│       ├── evaluate_img_with_data_collection.py  # 集成版评估脚本
│       └── HOW_TO_ADD_DATA_COLLECTION.md         # 集成指南
│
└── docs/
    ├── DATA_COLLECTION.md                      # 用户使用文档
    └── DATA_COLLECTION_IMPLEMENTATION.md       # 实现细节文档
```

### 生成的数据结构

```
training_data/
├── sessions/                    # 所有会话
│   ├── session_20250124_143022_abc123/
│   │   ├── session_metadata.json   # 会话完整信息
│   │   ├── samples/                # 各个推理样本
│   │   │   ├── sample_1.json       # 第1轮推理
│   │   │   ├── sample_2.json       # 第2轮推理
│   │   │   └── sample_3.json       # 第3轮推理
│   │   └── images/                 # 所有相关图片
│   │       ├── original.jpg
│   │       ├── depth_result.jpg
│   │       └── pi3_result.png
│   └── session_20250124_143145_def456/
│       └── ...
│
├── statistics.json              # 采集统计信息
├── train.jsonl                  # JSONL 格式训练数据
└── train_sharegpt.json          # ShareGPT 格式训练数据
```

---

## 示例代码

### 示例 1：完整的采集流程

```python
from spagent.core import SPAgent, Model, DataCollector
from spagent.tools import DepthEstimationTool, Pi3Tool

# 1. 创建 DataCollector
collector = DataCollector(
    output_dir="my_training_data",
    save_images=True,
    auto_save=True
)

# 2. 创建 SPAgent
model = Model("Qwen/Qwen2-VL-7B-Instruct", "/path/to/model")
tools = [
    DepthEstimationTool(server_url="http://localhost:20019"),
    Pi3Tool(server_url="http://localhost:20030")
]

agent = SPAgent(
    model=model,
    tools=tools,
    data_collector=collector
)

# 3. 运行推理（自动采集）
result = agent.solve_problem(
    image_path="test_image.jpg",
    question="这个场景中有多少个物体？从不同角度看是否有遮挡？",
    max_iterations=3
)

print(f"答案: {result['answer']}")
print(f"使用的工具: {result['used_tools']}")
print(f"迭代次数: {result['iterations']}")

# 4. 查看统计
stats = collector.get_statistics()
print(f"\n采集统计:")
print(f"  总会话数: {stats['total_sessions']}")
print(f"  成功会话: {stats['successful_sessions']}")
print(f"  总样本数: {stats['total_samples']}")
print(f"  成功率: {stats['success_rate']:.1%}")

# 5. 导出训练数据
collector.export_for_training("my_training_data/train.jsonl", format="jsonl")
collector.export_for_training("my_training_data/train_sharegpt.json", format="sharegpt")
collector.save_statistics()

print(f"\n✅ 训练数据已保存到: my_training_data/")
```

### 示例 2：查看采集的数据

```python
import json
from pathlib import Path

# 读取会话数据
session_dir = Path("training_data/sessions/session_xxx")
with open(session_dir / "session_metadata.json") as f:
    session = json.load(f)

print(f"问题: {session['question']}")
print(f"成功: {session['success']}")
print(f"最终答案: {session['final_answer']}")
print(f"推理轮数: {session['num_iterations']}")

# 查看各个样本
for i, sample in enumerate(session['samples'], 1):
    print(f"\n=== 样本 {i} ===")
    print(f"迭代: {sample['iteration']}")
    print(f"图片数: {len(sample['images'])}")
    print(f"Prompt (前100字): {sample['prompt'][:100]}...")
    print(f"Response (前100字): {sample['response'][:100]}...")
```

### 示例 3：分析采集数据

```python
import json
from pathlib import Path
from collections import Counter

def analyze_training_data(data_dir):
    """分析采集的训练数据"""
    sessions_dir = Path(data_dir) / "sessions"
    
    stats = {
        "total_sessions": 0,
        "successful_sessions": 0,
        "total_samples": 0,
        "iteration_distribution": Counter(),
        "tool_usage": Counter()
    }
    
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        
        with open(session_dir / "session_metadata.json") as f:
            session = json.load(f)
        
        stats["total_sessions"] += 1
        
        if session["success"]:
            stats["successful_sessions"] += 1
            stats["total_samples"] += len(session["samples"])
            stats["iteration_distribution"][session["num_iterations"]] += 1
            
            # 统计工具使用
            for tool in session["metadata"].get("used_tools", []):
                stats["tool_usage"][tool] += 1
    
    # 打印报告
    print("=" * 60)
    print("训练数据分析报告")
    print("=" * 60)
    print(f"总会话数: {stats['total_sessions']}")
    print(f"成功会话: {stats['successful_sessions']}")
    print(f"总样本数: {stats['total_samples']}")
    print(f"成功率: {stats['successful_sessions']/stats['total_sessions']:.1%}")
    
    print(f"\n迭代次数分布:")
    for num_iters, count in sorted(stats['iteration_distribution'].items()):
        print(f"  {num_iters} 轮: {count} 个会话")
    
    print(f"\n工具使用统计:")
    for tool, count in stats['tool_usage'].most_common():
        print(f"  {tool}: {count} 次")
    
    return stats

# 使用
analyze_training_data("training_data")
```

---

## 导出格式说明

### 1. JSONL 格式

每行一个样本，便于流式加载：

```jsonl
{"sample_id": "session_xxx_iter_1", "iteration": 1, "images": ["img.jpg"], "prompt": "...", "response": "..."}
{"sample_id": "session_xxx_iter_2", "iteration": 2, "images": ["img.jpg", "depth.jpg"], "prompt": "...", "response": "..."}
```

### 2. ShareGPT 格式

适用于多模态对话模型训练：

```json
[
  {
    "id": "session_xxx_iter_1",
    "images": ["img1.jpg", "img2.jpg"],
    "conversations": [
      {"from": "human", "value": "问题文本..."},
      {"from": "gpt", "value": "回复文本..."}
    ]
  }
]
```

### 3. JSON 格式

所有样本在一个数组中：

```json
[
  {
    "sample_id": "...",
    "iteration": 1,
    "images": [...],
    "prompt": "...",
    "response": "..."
  }
]
```

---

## 最佳实践

### ✅ 推荐做法

1. **小规模测试**：先用 10-20 个样本测试数据采集是否正常
2. **定期备份**：大规模采集时定期备份 `training_data/` 目录
3. **监控磁盘空间**：启用 `save_images=True` 会占用较多空间
4. **保留所有样本**：包括错误的样本，用于分析和改进
5. **版本管理**：为不同批次的数据使用不同的输出目录

### ⚠️ 注意事项

1. **并发限制**：多进程采集时每个进程使用独立的 `output_dir`
2. **异常处理**：即使采集失败，评估仍会继续
3. **数据清理**：定期清理不需要的旧数据
4. **路径问题**：建议使用绝对路径或 `save_images=True`

---

## 技术支持

- 详细文档：`docs/DATA_COLLECTION.md`
- 实现说明：`docs/DATA_COLLECTION_IMPLEMENTATION.md`
- 示例代码：`examples/data_collection_example.py`
- 集成指南：`examples/evaluation/HOW_TO_ADD_DATA_COLLECTION.md`

---

## 总结

✅ **完整实现了您的需求**：
- 每次推理都记录（图片+文本+输出）
- 只保存成功的样本
- 多轮执行成功后，所有样本都保存为正样本
- 包含完整的 memory 和 context 信息

🎯 **使用简单**：
- 只需 3 行代码即可启用
- 自动采集，无需手动干预
- 支持批量评估时同时采集

📦 **功能完善**：
- 多种导出格式
- 统计和监控
- 灵活的成功判定
- 完整的文档和示例

现在您可以开始使用 SPAgent 采集高质量的多模态训练数据了！


