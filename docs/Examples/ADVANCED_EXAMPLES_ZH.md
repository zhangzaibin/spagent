# 高级使用示例

> **English Version**: [English Document](ADVANCED_EXAMPLES.md) | **中文版本**: 当前文档

本指南提供了 SPAgent 的高级使用示例和策略，包括专门化代理、工具混合策略和复杂工作流。

## 📋 目录

- [命令行示例](#命令行示例)
- [专门化代理示例](#专门化代理示例)
- [工具混合策略](#工具混合策略)
- [视频分析测试](#视频分析测试)
- [强化学习训练](#强化学习训练)

## 命令行示例

### 基本命令行用法

```bash
# 在数据集上运行评估
python examples/evaluation/evaluate_img.py \
    --data_path dataset/your_data.jsonl \
    --model gpt-4o-mini \
    --max_samples 10 \
    --max_iterations 3

# 无工具评估（基线）
python examples/evaluation/evaluate_img_wotools.py \
    --data_path dataset/your_data.jsonl \
    --model gpt-4o-mini \
    --max_samples 10

# 评估时收集训练数据
python examples/evaluation/evaluate_img_with_data_collection.py \
    --data_path dataset/your_data.jsonl \
    --model gpt-4o-mini \
    --max_samples 10 \
    --enable_data_collection
```

## 专门化代理示例

通过选择适当的工具组合，创建针对特定任务的专门化代理。

### 1. 深度分析专门化代理

构建专注于深度分析任务的代理：

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool

# 专门用于深度分析的代理
model = GPTModel(model_name="gpt-4o-mini")
depth_tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True)  # 辅助分割
]

depth_agent = SPAgent(model=model, tools=depth_tools)
result = depth_agent.solve_problem(
    "image.jpg", 
    "Analyze the depth distribution of the image: which objects are close to the camera and which are far?"
)
```

### 2. 物体检测专门化代理

创建针对物体检测任务优化的代理：

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import ObjectDetectionTool, SupervisionTool, YOLOETool, SegmentationTool

# 专门用于物体检测的代理
model = GPTModel(model_name="gpt-4o-mini")
detection_tools = [
    ObjectDetectionTool(use_mock=True),
    SupervisionTool(use_mock=True),
    YOLOETool(use_mock=True),
    SegmentationTool(use_mock=True)  # 辅助分割
]

detection_agent = SPAgent(model=model, tools=detection_tools)
result = detection_agent.solve_problem(
    "image.jpg", 
    "Detect and identify all objects in the image, including their positions and types"
)
```

### 3. 自定义工具组合

根据需求条件性地添加工具，动态构建代理：

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, ObjectDetectionTool, SegmentationTool

# 创建一个空代理并逐步添加工具
agent = SPAgent(model=GPTModel())

# 根据需要添加工具
if need_depth:
    agent.add_tool(DepthEstimationTool(use_mock=True))

if need_detection:
    agent.add_tool(ObjectDetectionTool(use_mock=True))
    
if need_segmentation:
    agent.add_tool(SegmentationTool(use_mock=True))

# 使用配置好的代理
result = agent.solve_problem("image.jpg", "Analyze the image using available tools")
```

## 工具混合策略

SPAgent 提供了强大的策略来组合多个工具以解决复杂的视觉任务。

### 1. 并行工具执行

SPAgent 自动检测可以并行执行的工具，提高性能：

```python
# 这个问题将触发多个工具并行执行
result = agent.solve_problem(
    "image.jpg",
    "Perform depth estimation, object detection, and image segmentation simultaneously"  # 将并行执行 3 个工具
)
```

### 2. 条件工具选择

模型会根据问题描述自动选择所需的工具：

```python
# 只会使用与深度相关的工具
result1 = agent.solve_problem("image.jpg", "Analyze depth relationships")

# 只会使用与检测相关的工具  
result2 = agent.solve_problem("image.jpg", "Detect vehicles and pedestrians")

# 将使用多个工具
result3 = agent.solve_problem("image.jpg", "Comprehensively analyze the image")
```

### 3. 工具链组合

创建工具按顺序使用的复杂处理管道：

```python
# 复杂的工具链：检测 → 分割 → 深度分析
result = agent.solve_problem(
    "image.jpg",
    """
    First detect the main objects in the image,
    then perform precise segmentation on the detected objects,
    finally analyze the depth relationships of these objects
    """
)
```

## 视频分析测试

SPAgent 支持通过提取视频帧并使用 Pi3 等工具进行 3D 重建来分析视频。

### 基本视频帧分析

```python
# test/test_pi3_llm.py - 完整的视频分析示例
import cv2
from pathlib import Path
from spagent.core.spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import Pi3Tool

def extract_video_frames(video_path: str, num_frames: int = 10):
    """从视频中均匀提取帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames / num_frames
    
    frame_paths = []
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    
    for i in range(num_frames):
        frame_idx = int(i * frame_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = temp_dir / f"frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
    
    cap.release()
    return frame_paths

# 配置模型和 Pi3 工具
model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)
tools = [Pi3Tool(use_mock=False, server_url="http://localhost:20030")]

agent = SPAgent(model=model, tools=tools, max_workers=4)

# 从视频中提取帧
video_path = "path/to/video.mp4"
frame_paths = extract_video_frames(video_path, num_frames=10)

# 使用 Pi3 3D 重建分析视频帧
result = agent.solve_problem(
    frame_paths,
    "Based on these frames from a video, which direction did the object move?",
    video_path=video_path,  # 传递视频路径，Pi3 可以提取更多帧
    pi3_num_frames=50  # Pi3 分析使用的帧数
)

print(f"答案: {result['answer']}")
print(f"使用的工具: {result['used_tools']}")
```

## 强化学习训练

SPAgent 支持使用 [ms-swift](https://github.com/modelscope/ms-swift) 进行 GRPO（Group Relative Policy Optimization）强化学习训练。

### 训练脚本概览

| 脚本 | 描述 |
|------|------|
| `train/train_grpo.sh` | 带工具调用的标准 GRPO 训练 |
| `train/train_grpo_all_angles.sh` | 使用所有角度组合的 GRPO 训练 |
| `train/train_grpo_notool.sh` | 不使用工具调用的 GRPO 训练（基线） |
| `train/merge_lora.sh` | 将 LoRA 适配器合并到基础模型 |
| `train/compress_model.sh` | 压缩训练后的模型检查点 |

### 运行训练

```bash
# 带工具调用的标准 GRPO 训练
cd train
bash train_grpo.sh

# 无工具训练（用于基线对比）
bash train_grpo_notool.sh

# 使用所有角度组合训练（用于 Pi3）
bash train_grpo_all_angles.sh
```

### 关键训练配置

```bash
# GRPO 训练配置示例
swift rlhf \
    --rlhf_type grpo \
    --model path/to/Qwen3-VL-4B-Instruct \
    --external_plugins plugin/plugin.py \
    --multi_turn_scheduler spagent_tool_call_scheduler \
    --max_turns 3 \                              # 最大工具调用轮数
    --reward_funcs external_r1v_acc external_multiturn_format \
    --reward_weights 1.0 1.0 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset path/to/training_data.jsonl \
    --max_completion_length 1024 \
    --learning_rate 1e-6 \
    --num_generations 8 \                        # 每个样本的生成数量
    --temperature 0.6 \
    --deepspeed zero2 \
    --output_dir output/grpo_experiment
```

### 训练后操作

```bash
# 将 LoRA 权重合并到基础模型
swift export \
    --adapters output/grpo_xxx/checkpoint-xxx \
    --merge_lora true

# 压缩模型检查点用于部署
bash train/compress_model.sh
```

### 系统提示词

不同的训练模式使用位于 `train/system_prompt/` 的不同系统提示词：

- `system_prompt_grpo.txt` - 带工具调用的标准训练
- `system_prompt_grpo_all_angles.txt` - 使用所有角度组合的训练
- `system_prompt_grpo_wotool.txt` - 无工具训练

## 相关文档

- [快速入门指南](../../readme.md#-quick-start)
- [工具参考](../Tool/TOOL_USING.md)
- [评估指南](../Evaluation/EVALUATION.md)

---

更多信息或支持，请参考主 [README](../../readme.md) 或在 GitHub 上提交问题。

