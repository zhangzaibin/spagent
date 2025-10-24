# SPAgent 数据采集功能实现方案

## 概述

本文档详细说明了为 SPAgent 框架实现的训练数据采集功能的设计和实现方案。

## 需求分析

### 核心需求

1. **采集内容**：记录每次推理的输入（图片+文本prompt）和输出（模型回复）
2. **采集粒度**：每次推理作为一个独立的训练样本
3. **多轮支持**：一个问题可能需要多次推理，每轮推理都是独立样本
4. **成功过滤**：只保留成功完成的多轮对话中的所有样本
5. **完整上下文**：保存图片、文本、工具调用等所有相关信息

### 使用场景

- 为多模态大模型训练收集高质量数据
- 评估模型在多轮推理任务上的表现
- 分析工具调用的效果
- 构建特定领域的训练数据集

## 架构设计

### 模块结构

```
spagent/
├── core/
│   ├── spagent.py          # 主Agent类（已修改）
│   ├── data_collector.py   # 数据采集模块（新增）
│   └── __init__.py         # 导出接口（已修改）
├── examples/
│   ├── data_collection_example.py           # 详细示例
│   └── quick_start_data_collection.py       # 快速开始
└── docs/
    ├── DATA_COLLECTION.md                   # 使用文档
    └── DATA_COLLECTION_IMPLEMENTATION.md    # 本文档
```

### 核心类设计

#### 1. InferenceSample（推理样本）

```python
class InferenceSample:
    - sample_id: str           # 唯一标识符
    - iteration: int           # 迭代序号
    - images: List[str]        # 使用的图片路径列表
    - prompt: str              # 完整的输入prompt
    - response: str            # 模型的完整回复
    - context: Dict[str, Any]  # 上下文信息（工具调用、历史等）
    - timestamp: str           # 时间戳
```

**设计考虑**：
- 每个样本包含完整的单次推理信息
- context 字段灵活存储额外信息
- 时间戳便于追踪和调试

#### 2. SessionData（会话数据）

```python
class SessionData:
    - session_id: str               # 会话唯一标识
    - question: str                 # 原始问题
    - original_images: List[str]    # 原始输入图片
    - samples: List[InferenceSample]  # 推理样本列表
    - success: bool                 # 会话是否成功
    - final_answer: str             # 最终答案
    - error_message: str            # 错误信息（如果失败）
    - metadata: Dict[str, Any]      # 额外元数据
```

**设计考虑**：
- 会话是样本的容器，代表一个完整的问答过程
- 只有成功的会话才会被保存
- 元数据字段支持自定义扩展

#### 3. DataCollector（数据收集器）

```python
class DataCollector:
    - output_dir: Path          # 数据输出目录
    - save_images: bool         # 是否复制图片
    - auto_save: bool           # 自动保存成功会话
    - current_session: SessionData  # 当前活动会话
    
    # 主要方法
    - start_session(question, image_paths)  # 开始新会话
    - record_inference(...)                  # 记录推理步骤
    - end_session(success, final_answer)    # 结束会话
    - export_for_training(format)           # 导出训练数据
    - get_statistics()                       # 获取统计信息
```

**设计考虑**：
- 支持自动和手动两种模式
- 提供多种导出格式（JSONL、JSON、ShareGPT）
- 内置统计功能

## 实现细节

### 1. 集成到 SPAgent

在 `SPAgent.solve_problem()` 方法中集成数据采集：

```python
def solve_problem(self, image_path, question, max_iterations=3, **kwargs):
    # 1. 开始会话
    if self.data_collector:
        self.data_collector.start_session(question, image_paths)
    
    # 2. 多轮推理循环
    while iteration < max_iterations:
        # 执行模型推理
        response = self.model.inference(...)
        
        # 记录推理数据
        if self.data_collector:
            self.data_collector.record_inference(
                iteration=iteration,
                images=current_images,
                prompt=prompt,
                response=response,
                context={...}
            )
        
        # ... 工具调用、结果处理 ...
    
    # 3. 结束会话
    if self.data_collector:
        extracted_answer = self._extract_answer(final_response)
        success = extracted_answer is not None
        
        self.data_collector.end_session(
            success=success,
            final_answer=extracted_answer or final_response,
            metadata={...}
        )
```

### 2. 数据采集时机

采集发生在以下关键点：

1. **初始推理**：第一次调用模型（iteration 1）
2. **后续迭代**：每次额外的推理迭代（iteration 2, 3, ...）
3. **最终综合**：如果需要生成最终答案（final synthesis）
4. **基线综合**：如果启用了基线比较（baseline synthesis）

每个时机都会生成一个独立的 InferenceSample。

### 3. 成功判定逻辑

```python
def _extract_answer(self, response: str) -> Optional[str]:
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# 在 solve_problem 结尾
extracted_answer = self._extract_answer(final_response)
success = extracted_answer is not None  # 有answer标签即成功
```

**设计考虑**：
- 默认依赖 `<answer>` 标签判定成功
- 可以通过手动模式实现自定义判定逻辑

### 4. 数据存储结构

```
training_data/
├── sessions/                              # 所有会话
│   ├── session_20250124_143022_abc123/   # 单个会话目录
│   │   ├── session_metadata.json         # 会话完整信息
│   │   ├── samples/                       # 推理样本
│   │   │   ├── sample_1.json
│   │   │   ├── sample_2.json
│   │   │   └── sample_3.json
│   │   └── images/                        # 相关图片
│   │       ├── original.jpg
│   │       ├── depth_result.jpg
│   │       └── pi3_result.png
│   └── session_20250124_143145_def456/
│       └── ...
├── images/                                 # (可选) 全局图片
├── statistics.json                        # 统计信息
├── train.jsonl                            # 导出的训练数据
└── train_sharegpt.json                    # ShareGPT格式
```

**设计优势**：
- 每个会话独立目录，便于管理和清理
- 样本单独存储，便于加载和处理
- 图片本地化，避免路径依赖

### 5. 导出格式

#### JSONL 格式（推荐）
```jsonl
{"sample_id": "...", "iteration": 1, "images": [...], "prompt": "...", "response": "..."}
{"sample_id": "...", "iteration": 2, "images": [...], "prompt": "...", "response": "..."}
```

**适用场景**：大规模训练，流式加载

#### ShareGPT 格式
```json
[
  {
    "id": "session_xxx_iter_1",
    "images": ["img1.jpg", "img2.jpg"],
    "conversations": [
      {"from": "human", "value": "question"},
      {"from": "gpt", "value": "answer"}
    ]
  }
]
```

**适用场景**：LLaMA、Qwen 等多模态模型训练

## 使用流程

### 自动模式（推荐）

```python
# 1. 初始化
collector = DataCollector("training_data", auto_save=True)
agent = SPAgent(model, tools, data_collector=collector)

# 2. 运行推理（自动采集）
result = agent.solve_problem(image, question, max_iterations=3)

# 3. 导出数据
collector.export_for_training("train.jsonl", format="jsonl")
```

### 手动模式（高级）

```python
# 1. 初始化
collector = DataCollector("training_data", auto_save=False)

# 2. 手动控制
collector.start_session(question, images)
collector.record_inference(iteration=1, images=..., prompt=..., response=...)
collector.end_session(success=True, final_answer="...")
```

## 关键特性

### 1. 多轮推理支持

- 每轮推理作为独立样本
- 保留完整的推理链条
- 上下文信息累积记录

**示例**：3轮推理会生成3-4个样本（可能包含final synthesis）

### 2. 只保存成功样本

- 失败会话不保存，节省存储空间
- 确保训练数据质量
- 可通过统计信息追踪成功率

### 3. 图片管理

- 可选择是否复制图片到输出目录
- 自动处理工具生成的额外图片
- 避免重复复制相同图片

### 4. 灵活的上下文记录

每个样本的 context 包含：
- `tool_calls_history`: 工具调用历史
- `tool_results_history`: 工具执行结果
- `additional_images_history`: 生成的额外图片
- 自定义字段（如 `type: "final_synthesis"`）

### 5. 统计与监控

```python
stats = collector.get_statistics()
# {
#     "total_sessions": 100,
#     "successful_sessions": 85,
#     "failed_sessions": 15,
#     "total_samples": 255,
#     "success_rate": 0.85,
#     "avg_samples_per_success": 3.0
# }
```

## 性能考虑

### 内存使用

- 当前会话数据保存在内存中
- 会话结束时写入磁盘
- 对于长时间运行，建议定期检查磁盘空间

### 磁盘空间

- 如果 `save_images=True`，每个会话会复制所有相关图片
- 建议：
  - 小规模采集：`save_images=True`
  - 大规模采集：`save_images=False`，仅保存路径

### 并发限制

- 当前实现不支持多进程并发写入同一个 DataCollector
- 解决方案：每个进程使用独立的 output_dir

## 扩展方向

### 1. 自定义成功判定

```python
# 当前：基于 <answer> 标签
# 未来：支持自定义验证函数
def custom_validator(response, expected):
    # 自定义逻辑
    return is_valid

collector = DataCollector(validator=custom_validator)
```

### 2. 增量导出

```python
# 当前：导出所有成功会话
# 未来：支持增量导出新会话
collector.export_for_training(
    output_file="train.jsonl",
    format="jsonl",
    since="2025-01-24"  # 只导出该日期之后的
)
```

### 3. 数据版本管理

```python
# 未来：支持数据版本控制
collector = DataCollector(
    output_dir="training_data",
    version="v1.0"
)
```

### 4. 在线监控

```python
# 未来：实时监控采集进度
collector.start_monitoring(port=8000)
# 访问 http://localhost:8000 查看统计
```

## 最佳实践

### 1. 批量采集

```python
collector = DataCollector("training_data")
agent = SPAgent(model, tools, data_collector=collector)

for case in test_cases:
    try:
        agent.solve_problem(case["image"], case["question"])
    except Exception as e:
        logger.error(f"Error: {e}")

collector.save_statistics()
```

### 2. 定期备份

```python
import shutil
from datetime import datetime

# 每天备份
backup_dir = f"backup_{datetime.now().strftime('%Y%m%d')}"
shutil.copytree("training_data", backup_dir)
```

### 3. 数据质量检查

```python
def check_quality(sessions_dir):
    for session_dir in Path(sessions_dir).iterdir():
        with open(session_dir / "session_metadata.json") as f:
            session = json.load(f)
        
        # 检查样本数量
        if len(session["samples"]) < 1:
            print(f"Warning: {session['session_id']} has no samples")
        
        # 检查答案质量
        if len(session["final_answer"]) < 10:
            print(f"Warning: {session['session_id']} has short answer")
```

### 4. 多格式导出

```python
# 导出多种格式供不同训练框架使用
collector.export_for_training("train.jsonl", format="jsonl")
collector.export_for_training("train_sharegpt.json", format="sharegpt")
collector.export_for_training("train.json", format="json")
```

## 测试

### 单元测试

```python
def test_data_collector():
    collector = DataCollector("test_data", auto_save=False)
    
    # 测试会话创建
    session_id = collector.start_session("test question", ["img.jpg"])
    assert collector.current_session is not None
    
    # 测试记录推理
    collector.record_inference(1, ["img.jpg"], "prompt", "response")
    assert len(collector.current_session.samples) == 1
    
    # 测试结束会话
    collector.end_session(True, "final answer")
    assert collector.current_session is None
```

### 集成测试

```python
def test_integration():
    collector = DataCollector("test_data")
    model = Model("test-model", "path")
    agent = SPAgent(model, [], data_collector=collector)
    
    result = agent.solve_problem("test.jpg", "test question")
    
    stats = collector.get_statistics()
    assert stats["total_sessions"] > 0
```

## 故障排除

### 问题1：会话未保存

**症状**：运行后没有生成数据

**可能原因**：
- 会话失败（没有 `<answer>` 标签）
- `auto_save=False` 且未手动调用 `end_session`

**解决**：检查 logs，确认成功判定逻辑

### 问题2：图片路径错误

**症状**：导出的数据中图片路径无效

**解决**：
- 设置 `save_images=True` 以复制图片
- 或使用绝对路径

### 问题3：磁盘空间不足

**症状**：采集中途失败

**解决**：
- 设置 `save_images=False`
- 定期清理旧数据
- 使用更大的存储设备

## 总结

本实现提供了完整的训练数据采集解决方案，具有以下优势：

✅ **完整性**：记录每次推理的所有信息  
✅ **灵活性**：支持自动和手动两种模式  
✅ **可靠性**：只保存成功样本，确保质量  
✅ **易用性**：简单的 API，最少 3 行代码即可使用  
✅ **可扩展性**：支持多种导出格式和自定义扩展  

该功能已完全集成到 SPAgent 框架中，可直接用于生产环境的数据采集任务。


