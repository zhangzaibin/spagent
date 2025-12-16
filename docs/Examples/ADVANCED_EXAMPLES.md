# Advanced Examples

> **中文版本**: [中文文档](ADVANCED_EXAMPLES_ZH.md) | **English Version**: This document

This guide provides advanced usage examples and strategies for SPAgent, including specialized agents, tool mixing strategies, and complex workflows.

## 📋 Table of Contents

- [Command Line Examples](#command-line-examples)
- [Specialized Agent Examples](#specialized-agent-examples)
- [Tool Mixing Strategies](#tool-mixing-strategies)
- [Video Analysis Testing](#video-analysis-testing)
- [Reinforcement Learning Training](#reinforcement-learning-training)

## Command Line Examples

### Basic Command Line Usage

```bash
# Run evaluation on a dataset
python examples/evaluation/evaluate_img.py \
    --data_path dataset/your_data.jsonl \
    --model gpt-4o-mini \
    --max_samples 10 \
    --max_iterations 3

# Evaluation without tools (baseline)
python examples/evaluation/evaluate_img_wotools.py \
    --data_path dataset/your_data.jsonl \
    --model gpt-4o-mini \
    --max_samples 10

# Collect training data during evaluation
python examples/evaluation/evaluate_img_with_data_collection.py \
    --data_path dataset/your_data.jsonl \
    --model gpt-4o-mini \
    --max_samples 10 \
    --enable_data_collection
```

## Specialized Agent Examples

Create specialized agents tailored for specific tasks by selecting appropriate tool combinations.

### 1. Depth Analysis Specialized Agent

Build an agent focused on depth analysis tasks:

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool

# Agent specialized for depth analysis
model = GPTModel(model_name="gpt-4o-mini")
depth_tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True)  # Auxiliary segmentation
]

depth_agent = SPAgent(model=model, tools=depth_tools)
result = depth_agent.solve_problem(
    "image.jpg", 
    "Analyze the depth distribution of the image: which objects are close to the camera and which are far?"
)
```

### 2. Object Detection Specialized Agent

Create an agent optimized for object detection tasks:

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import ObjectDetectionTool, SupervisionTool, YOLOETool, SegmentationTool

# Agent specialized for object detection
model = GPTModel(model_name="gpt-4o-mini")
detection_tools = [
    ObjectDetectionTool(use_mock=True),
    SupervisionTool(use_mock=True),
    YOLOETool(use_mock=True),
    SegmentationTool(use_mock=True)  # Auxiliary segmentation
]

detection_agent = SPAgent(model=model, tools=detection_tools)
result = detection_agent.solve_problem(
    "image.jpg", 
    "Detect and identify all objects in the image, including their positions and types"
)
```

### 3. Custom Tool Combination

Dynamically build agents by conditionally adding tools based on requirements:

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, ObjectDetectionTool, SegmentationTool

# Create an empty agent and add tools step by step
agent = SPAgent(model=GPTModel())

# Add tools as needed
if need_depth:
    agent.add_tool(DepthEstimationTool(use_mock=True))

if need_detection:
    agent.add_tool(ObjectDetectionTool(use_mock=True))
    
if need_segmentation:
    agent.add_tool(SegmentationTool(use_mock=True))

# Use the configured agent
result = agent.solve_problem("image.jpg", "Analyze the image using available tools")
```

## Tool Mixing Strategies

SPAgent provides powerful strategies for combining multiple tools to solve complex vision tasks.

### 1. Parallel Tool Execution

SPAgent automatically detects tools that can be executed in parallel, improving performance:

```python
# This problem will trigger multiple tools to execute in parallel
result = agent.solve_problem(
    "image.jpg",
    "Perform depth estimation, object detection, and image segmentation simultaneously"  # Will execute 3 tools in parallel
)
```

### 2. Conditional Tool Selection

The model automatically selects the needed tools based on the problem description:

```python
# Will only use depth-related tools
result1 = agent.solve_problem("image.jpg", "Analyze depth relationships")

# Will only use detection-related tools  
result2 = agent.solve_problem("image.jpg", "Detect vehicles and pedestrians")

# Will use multiple tools
result3 = agent.solve_problem("image.jpg", "Comprehensively analyze the image")
```

### 3. Tool Chain Combination

Create complex processing pipelines where tools are used sequentially:

```python
# Complex tool chain: detection → segmentation → depth analysis
result = agent.solve_problem(
    "image.jpg",
    """
    First detect the main objects in the image,
    then perform precise segmentation on the detected objects,
    finally analyze the depth relationships of these objects
    """
)
```

## Video Analysis Testing

SPAgent supports video analysis by extracting frames and using tools like Pi3 for 3D reconstruction.

### Basic Video Frame Analysis

```python
# test/test_pi3_llm.py - Complete video analysis example
import cv2
from pathlib import Path
from spagent.core.spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import Pi3Tool

def extract_video_frames(video_path: str, num_frames: int = 10):
    """Extract frames uniformly from a video"""
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

# Configure model and Pi3 tool
model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)
tools = [Pi3Tool(use_mock=False, server_url="http://localhost:20030")]

agent = SPAgent(model=model, tools=tools, max_workers=4)

# Extract frames from video
video_path = "path/to/video.mp4"
frame_paths = extract_video_frames(video_path, num_frames=10)

# Analyze video frames with Pi3 3D reconstruction
result = agent.solve_problem(
    frame_paths,
    "Based on these frames from a video, which direction did the object move?",
    video_path=video_path,  # Pass video path for Pi3 to extract more frames if needed
    pi3_num_frames=50  # Number of frames for Pi3 analysis
)

print(f"Answer: {result['answer']}")
print(f"Used tools: {result['used_tools']}")
```

## Reinforcement Learning Training

SPAgent supports GRPO (Group Relative Policy Optimization) reinforcement learning training using [ms-swift](https://github.com/modelscope/ms-swift).

### Training Scripts Overview

| Script | Description |
|--------|-------------|
| `train/train_grpo.sh` | Standard GRPO training with tool calling |
| `train/train_grpo_all_angles.sh` | GRPO training with all angle combinations |
| `train/train_grpo_notool.sh` | GRPO training without tool calling (baseline) |
| `train/merge_lora.sh` | Merge LoRA adapters into base model |
| `train/compress_model.sh` | Compress trained model checkpoints |

### Running Training

```bash
# Standard GRPO training with tool calling
cd train
bash train_grpo.sh

# Training without tools (baseline comparison)
bash train_grpo_notool.sh

# Training with all angle combinations for Pi3
bash train_grpo_all_angles.sh
```

### Key Training Configuration

```bash
# Example GRPO training configuration
swift rlhf \
    --rlhf_type grpo \
    --model path/to/Qwen3-VL-4B-Instruct \
    --external_plugins plugin/plugin.py \
    --multi_turn_scheduler spagent_tool_call_scheduler \
    --max_turns 3 \                              # Maximum tool calling turns
    --reward_funcs external_r1v_acc external_multiturn_format \
    --reward_weights 1.0 1.0 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset path/to/training_data.jsonl \
    --max_completion_length 1024 \
    --learning_rate 1e-6 \
    --num_generations 8 \                        # Number of generations per sample
    --temperature 0.6 \
    --deepspeed zero2 \
    --output_dir output/grpo_experiment
```

### Post-Training Operations

```bash
# Merge LoRA weights into base model
swift export \
    --adapters output/grpo_xxx/checkpoint-xxx \
    --merge_lora true

# Compress model checkpoint for deployment
bash train/compress_model.sh
```

### System Prompts

Different training modes use different system prompts located in `train/system_prompt/`:

- `system_prompt_grpo.txt` - Standard training with tool calling
- `system_prompt_grpo_all_angles.txt` - Training with all angle combinations
- `system_prompt_grpo_wotool.txt` - Training without tools

## Related Documentation

- [Quick Start Guide](../../readme.md#-quick-start)
- [Tool Reference](../Tool/TOOL_USING.md)
- [Evaluation Guide](../Evaluation/EVALUATION.md)

---

For more information or support, please refer to the main [README](../../readme.md) or open an issue on GitHub.
