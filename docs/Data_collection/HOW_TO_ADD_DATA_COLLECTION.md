# 如何为评估脚本添加数据采集功能

本文档说明如何将 DataCollector 集成到现有的评估脚本中。

## 方案一：最小修改（推荐）

只需 3 步修改原有的 `evaluate_img.py`：

### 步骤 1：导入 DataCollector

在文件顶部添加导入：

```python
from spagent.core import DataCollector
```

### 步骤 2：修改 `evaluate_tool_config` 函数

在 `spagent_evaluation.py` 的 `evaluate_tool_config` 函数中添加 DataCollector 支持：

```python
def evaluate_tool_config(
    config_name: str,
    tools: List[Any],
    data_path: str,
    image_base_path: str,
    model: str = "gpt-4o-mini",
    max_samples: int = None,
    max_workers: int = 4,
    max_iterations: int = 3,
    enable_data_collection: bool = False,  # 新增参数
    data_output_dir: str = None  # 新增参数
) -> Dict[str, Any]:
    
    # ... 现有代码 ...
    
    # === 新增：初始化 DataCollector ===
    collector = None
    if enable_data_collection:
        if data_output_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            data_output_dir = f"training_data/{config_name}_{timestamp}"
        
        collector = DataCollector(
            output_dir=data_output_dir,
            save_images=True,
            auto_save=True  # 自动保存成功的会话
        )
        print(f"✓ Data collection enabled: {data_output_dir}")
    
    # === 修改：创建 SPAgent 时传入 DataCollector ===
    agent = SPAgent(
        model=GPTModel(model_name=model),
        tools=tools,
        max_workers=max_workers,
        data_collector=collector  # 添加这一行
    )
    
    # ... 现有的评估循环代码 ...
    
    # === 新增：评估结束后导出数据 ===
    if enable_data_collection and collector:
        print(f"\n{'='*60}")
        print("Data Collection Summary")
        print(f"{'='*60}")
        
        stats = collector.get_statistics()
        print(f"Total sessions:      {stats['total_sessions']}")
        print(f"Successful sessions: {stats['successful_sessions']}")
        print(f"Total samples:       {stats['total_samples']}")
        
        # 保存统计信息
        collector.save_statistics()
        
        # 导出训练数据
        collector.export_for_training(
            output_file=f"{data_output_dir}/train.jsonl",
            format="jsonl"
        )
        collector.export_for_training(
            output_file=f"{data_output_dir}/train_sharegpt.json",
            format="sharegpt"
        )
        print(f"✓ Training data exported to: {data_output_dir}/")
    
    return {
        # ... 现有返回值 ...
        "data_collection_enabled": enable_data_collection,  # 新增
        "data_collection_dir": data_output_dir if enable_data_collection else None  # 新增
    }
```

### 步骤 3：修改 main 函数调用

在 `evaluate_img.py` 的 main 函数中添加参数：

```python
def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Server')
    # ... 现有参数 ...
    
    # 新增：数据采集参数
    parser.add_argument('--enable_data_collection', action='store_true',
                        help='Enable training data collection')
    parser.add_argument('--data_output_dir', type=str, default=None,
                        help='Directory for training data (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # ... 现有代码 ...
    
    # 修改：调用时传入数据采集参数
    for config_name, tools in TOOL_CONFIGS.items():
        results = evaluate_tool_config(
            config_name=config_name,
            tools=tools,
            data_path=args.data_path,
            image_base_path=args.image_base_path,
            model=args.model,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            max_iterations=args.max_iterations,
            enable_data_collection=args.enable_data_collection,  # 新增
            data_output_dir=args.data_output_dir  # 新增
        )
        all_results[config_name] = results
```

## 使用方法

### 不启用数据采集（默认行为）

```bash
python examples/evaluation/evaluate_img.py \
    --data_path dataset/cvbench_data.jsonl \
    --max_samples 10 \
    --model gpt-4o
```

### 启用数据采集

```bash
python examples/evaluation/evaluate_img.py \
    --data_path dataset/cvbench_data.jsonl \
    --max_samples 10 \
    --model gpt-4o \
    --enable_data_collection
```

### 指定数据输出目录

```bash
python examples/evaluation/evaluate_img.py \
    --data_path dataset/cvbench_data.jsonl \
    --max_samples 10 \
    --model gpt-4o \
    --enable_data_collection \
    --data_output_dir my_training_data
```

## 方案二：使用独立脚本（已提供）

我们已经创建了一个独立的脚本 `evaluate_img_with_data_collection.py`，它包含了完整的数据采集功能。

使用方法：

```bash
python examples/evaluation/evaluate_img_with_data_collection.py \
    --data_path dataset/cvbench_data.jsonl \
    --max_samples 10 \
    --model gpt-4o \
    --enable_data_collection \
    --max_iterations 3
```

## 输出结构

启用数据采集后，会生成以下目录结构：

```
training_data/
└── depth_detection_segmentation_gpt_4o_20250124_143022/
    ├── sessions/
    │   ├── session_xxx/
    │   │   ├── session_metadata.json
    │   │   ├── samples/
    │   │   │   ├── sample_1.json
    │   │   │   ├── sample_2.json
    │   │   │   └── sample_3.json
    │   │   └── images/
    │   │       ├── original.jpg
    │   │       └── pi3_result.png
    │   └── ...
    ├── statistics.json
    ├── train.jsonl
    └── train_sharegpt.json
```

## 数据采集行为说明

### 自动保存条件

使用 `auto_save=True` 时，只有满足以下条件的会话才会被保存：
- 模型成功返回了包含 `<answer>` 标签的回复
- 推理过程没有抛出异常

**注意**：无论预测是否正确，只要成功完成推理就会保存。这是因为：
1. **正确的样本**：可以作为正例训练
2. **错误的样本**：也可以用于分析模型弱点或进行负采样训练

### 只保存正确样本（可选）

如果你只想保存预测正确的样本，需要使用手动控制模式：

```python
collector = DataCollector(
    output_dir=data_output_dir,
    save_images=True,
    auto_save=False  # 手动控制
)

# 在评估循环中
result = evaluate_single_sample(agent, sample, ...)

if result["success"] and result["is_correct"]:
    # 只有正确的样本才保存
    # 注意：这需要修改 SPAgent 的内部逻辑
    pass
```

**建议**：默认保存所有成功的样本，后续可以根据准确性标签进行过滤。

## 多轮推理的数据采集

对于 `max_iterations > 1` 的情况：

1. **每轮推理都是独立样本**
   - Iteration 1: 初始推理
   - Iteration 2: 基于工具结果的第二轮推理
   - Iteration 3: 进一步推理
   - Final synthesis: 最终答案综合

2. **所有样本都会被保存**
   - 如果整个会话成功，所有迭代的样本都保存
   - 如果会话失败，所有样本都不保存

3. **训练数据格式**
   ```json
   {
     "sample_id": "session_xxx_iter_1",
     "iteration": 1,
     "images": ["image.jpg"],
     "prompt": "完整的输入prompt...",
     "response": "模型的回复...",
     "context": {
       "tool_calls_history": [],
       "tool_results_history": {},
       "additional_images_history": []
     }
   }
   ```

## 高级：自定义成功判定

如果你想自定义哪些样本应该被保存（例如只保存准确率高的样本），可以：

### 方法1：评估后过滤

```python
# 正常运行评估并采集所有数据
# 然后编写脚本过滤 sessions 目录

import json
from pathlib import Path

sessions_dir = Path("training_data/xxx/sessions")
for session_dir in sessions_dir.iterdir():
    metadata = json.load(open(session_dir / "session_metadata.json"))
    
    # 自定义过滤逻辑
    if not meets_quality_criteria(metadata):
        # 删除不符合标准的会话
        shutil.rmtree(session_dir)

# 重新导出训练数据
collector.export_for_training(...)
```

### 方法2：在评估过程中过滤

修改 `evaluate_single_sample` 函数，在调用 `agent.solve_problem` 前后进行判断：

```python
def evaluate_single_sample_with_filtering(agent, sample, ...):
    # 临时禁用 auto_save
    agent.data_collector.auto_save = False
    
    # 运行评估
    agent_result = agent.solve_problem(...)
    
    # 判断是否保存
    if should_save(agent_result):
        # 手动标记成功并保存
        agent.data_collector.end_session(
            success=True,
            final_answer=agent_result["answer"]
        )
    else:
        # 标记失败，不保存
        agent.data_collector.end_session(
            success=False,
            error_message="Filtered by quality criteria"
        )
```

## 监控和统计

在评估过程中，你可以随时查看数据采集统计：

```python
stats = collector.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Total samples: {stats['total_samples']}")
print(f"Avg samples per session: {stats['avg_samples_per_success']:.2f}")
```

## 注意事项

1. **磁盘空间**：启用 `save_images=True` 会复制所有图片
2. **评估速度**：数据采集会略微增加 I/O 开销
3. **并发安全**：如果使用多进程评估，每个进程应使用独立的 DataCollector
4. **数据清理**：定期清理旧的训练数据目录

## 故障排除

### 问题：没有生成任何数据

**可能原因**：
- 所有会话都失败了（没有 `<answer>` 标签）
- `enable_data_collection=False`

**解决方法**：
- 检查评估日志，确认是否有成功的会话
- 确保 `--enable_data_collection` 参数已传递

### 问题：数据量太大

**解决方法**：
- 设置 `save_images=False`，只保存图片路径
- 定期导出并清理旧数据
- 使用外部存储或更大的磁盘

## 示例：完整的评估与数据采集流程

```bash
# 1. 运行评估并采集数据
python examples/evaluation/evaluate_img.py \
    --data_path dataset/cvbench_data.jsonl \
    --max_samples 100 \
    --model gpt-4o \
    --max_iterations 3 \
    --enable_data_collection \
    --data_output_dir my_training_data

# 2. 检查采集结果
ls -lh my_training_data/
cat my_training_data/statistics.json

# 3. 查看训练数据格式
head -n 1 my_training_data/train.jsonl | jq

# 4. 数据质量检查（可选）
python scripts/check_data_quality.py my_training_data/

# 5. 使用训练数据
# 将 train.jsonl 或 train_sharegpt.json 用于模型训练
```

## 总结

- ✅ **最小修改**：只需 3 处改动即可启用数据采集
- ✅ **兼容性**：不影响现有评估流程
- ✅ **灵活性**：可选启用/禁用数据采集
- ✅ **完整性**：保存所有推理步骤和上下文
- ✅ **多格式**：支持 JSONL、JSON、ShareGPT 等格式

建议从小规模测试开始（如 `--max_samples 10`），确认数据采集正常后再进行大规模评估。


