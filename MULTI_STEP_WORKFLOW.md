# Multi-Step Workflow Feature

## 概述

SPAgent现在支持多步workflow功能，允许模型进行多次迭代的推理和工具调用。这对于需要从多个角度查看3D渲染结果的任务特别有用。

## 主要改进

### 1. 多步迭代支持

- **新增参数**: `max_iterations` - 控制最大迭代次数（默认为1，保持向后兼容）
- **智能终止**: 当模型提供最终答案时，即使未达到最大迭代次数也会终止
- **状态追踪**: 跟踪所有迭代中的工具调用、结果和生成的图像

### 2. 角度可调的Pi3工具

Pi3Tool现在支持自定义视角参数：

- **azimuth_angle**: 方位角（左右旋转），范围：-180° 到 180°
  - 负值：向左旋转
  - 正值：向右旋转
  - 默认值：0°（正面视图）

- **elevation_angle**: 仰角（上下旋转），范围：-90° 到 90°
  - 负值：向下看
  - 正值：向上看
  - 默认值：0°（水平视图）

### 3. 智能Prompt系统

- **初始Prompt**: 第一次迭代使用标准系统提示
- **延续Prompt**: 后续迭代包含：
  - 之前的工具执行结果
  - 已生成的图像列表
  - 剩余迭代次数
  - 明确指导模型可以调用更多工具或提供最终答案

## 使用方法

### 基础用法（单次迭代）

```python
from spagent.core.spagent import SPAgent
from spagent.core.model import Model
from spagent.tools.pi3_tool import Pi3Tool

# 初始化
model = Model(model_name="Qwen2-VL-7B-Instruct")
agent = SPAgent(model=model)

# 添加工具
pi3_tool = Pi3Tool(use_mock=False, server_url="http://localhost:20021")
agent.add_tool(pi3_tool)

# 单次推理（默认行为，向后兼容）
result = agent.solve_problem(
    image_path="path/to/image.jpg",
    question="分析这个物体的3D结构",
    max_iterations=1  # 可以省略，默认为1
)
```

### 多步Workflow

```python
# 允许多次迭代
result = agent.solve_problem(
    image_path="path/to/image.jpg",
    question="""从多个角度分析这个物体的3D结构。
    首先查看正面视图，然后从左侧查看，最后从顶部查看。""",
    max_iterations=3  # 允许最多3次迭代
)

# 检查实际使用的迭代次数
print(f"完成的迭代次数: {result['iterations']}")
print(f"工具调用次数: {len(result['tool_calls'])}")
print(f"使用的工具: {result['used_tools']}")
print(f"生成的图像: {len(result['additional_images'])}")
```

### 指定特定视角

```python
result = agent.solve_problem(
    image_path="path/to/image.jpg",
    question="""使用以下角度生成3D重建：
    1. 正面视图 (azimuth=0, elevation=0)
    2. 左侧45度视图 (azimuth=-45, elevation=0)
    3. 俯视图 (azimuth=0, elevation=45)
    
    然后从所有这些视角分析结构。""",
    max_iterations=3
)
```

## 工作原理

### 迭代流程

```
迭代1:
  ├─ 模型推理 → 工具调用1 (azimuth=0, elevation=0)
  └─ 执行工具 → 生成图像1

迭代2:
  ├─ 模型看到图像1 → 决定需要不同角度
  ├─ 工具调用2 (azimuth=-45, elevation=0)
  └─ 执行工具 → 生成图像2

迭代3:
  ├─ 模型看到图像1和图像2 → 决定需要俯视图
  ├─ 工具调用3 (azimuth=0, elevation=45)
  └─ 执行工具 → 生成图像3

最终推理:
  └─ 基于所有图像生成综合答案
```

### 返回值结构

```python
{
    "answer": "最终答案文本",
    "initial_response": "第一次推理的响应",
    "tool_calls": [
        {"name": "pi3_tool", "arguments": {...}},
        ...
    ],
    "tool_results": {
        "pi3_tool_iter1": {...},
        "pi3_tool_iter2": {...},
        ...
    },
    "used_tools": ["pi3_tool_iter1", "pi3_tool_iter2", ...],
    "additional_images": ["path1.png", "path2.png", ...],
    "iterations": 3,
    "prompts": {...}
}
```

## Pi3工具调用示例

### 模型如何调用工具

在推理过程中，模型可以这样调用Pi3工具：

```xml
<tool_call>
{
  "name": "pi3_tool",
  "arguments": {
    "image_path": ["path/to/image.jpg"],
    "azimuth_angle": -45,
    "elevation_angle": 10
  }
}
</tool_call>
```

### 角度参数说明

| 参数 | 范围 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| azimuth_angle | -180° ~ 180° | 0° | 水平旋转角度 | -45°(左), 0°(正面), 45°(右) |
| elevation_angle | -90° ~ 90° | 0° | 垂直旋转角度 | -30°(俯视), 0°(水平), 30°(仰视) |

### 常用视角组合

```python
# 正面视图
{"azimuth_angle": 0, "elevation_angle": 0}

# 左侧视图
{"azimuth_angle": -45, "elevation_angle": 0}

# 右侧视图
{"azimuth_angle": 45, "elevation_angle": 0}

# 俯视图
{"azimuth_angle": 0, "elevation_angle": 45}

# 仰视图
{"azimuth_angle": 0, "elevation_angle": -45}

# 左上视图
{"azimuth_angle": -45, "elevation_angle": 30}
```

## 提前终止

如果模型在达到`max_iterations`之前就有足够的信息，它可以提供最终答案来提前终止workflow：

```xml
<think>
我已经从正面和左侧看到了足够的信息来回答这个问题。
</think>

<answer>
这个物体是一个立方体，边长约为10cm...
</answer>
```

## 注意事项

1. **向后兼容**: 默认`max_iterations=1`，保持原有行为
2. **性能考虑**: 每次迭代都会调用模型推理，增加`max_iterations`会增加处理时间
3. **图像累积**: 每次迭代生成的图像都会被保存，用于后续推理
4. **工具命名**: 每次迭代的工具结果会添加`_iterN`后缀，便于区分

## 完整示例

参考 `examples/multi_step_workflow_example.py` 获取完整的使用示例。

## 依赖服务

使用Pi3工具时需要先启动Pi3服务器：

```bash
# 启动Pi3服务器
cd spagent/external_experts/Pi3
python pi3_server.py --checkpoint_path /path/to/pi3/checkpoint --port 20021
```

## 调试

启用详细日志来查看每次迭代的详细信息：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

日志会显示：
- 每次迭代的开始和结束
- 工具调用详情
- 生成的图像路径
- 模型响应内容

