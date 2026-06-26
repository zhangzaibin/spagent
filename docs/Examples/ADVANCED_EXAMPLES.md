# Advanced Examples

> **中文版本**: [中文文档](ADVANCED_EXAMPLES_ZH.md) | **English Version**: This document

This guide provides advanced usage examples and strategies for SPAgent, including specialized agents, tool mixing strategies, and complex workflows.

## 📋 Table of Contents

- [Command Line Examples](#command-line-examples)
- [Specialized Agent Examples](#specialized-agent-examples)
- [Tool Mixing Strategies](#tool-mixing-strategies)
- [Dynamic Tool Management](#dynamic-tool-management)
- [Multi-Image Analysis](#multi-image-analysis)
- [Custom System Prompt](#custom-system-prompt)
- [Video Generation (Veo / Sora / VACE)](#video-generation-veo--sora--vace)
- [Image Generation (Sana)](#image-generation-sana)
- [step() API and AgentMemory](#step-api-and-agentmemory)
- [Testing & Development](#testing--development)
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

## Dynamic Tool Management

```python
# Start with a basic agent
agent = SPAgent(model=GPTModel())

# Dynamically add tools at runtime
agent.add_tool(DepthEstimationTool(use_mock=True))
agent.add_tool(SegmentationTool(use_mock=True))

# View current tools
print(f"Current tools: {agent.list_tools()}")

# Remove a tool
agent.remove_tool("depth_estimation_tool")

# Swap the underlying model
from spagent.models import QwenModel
agent.set_model(QwenModel(model_name="qwen2.5-vl-7b-instruct"))
```

## Multi-Image Analysis

```python
# Pass multiple images to the agent
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
result = agent.solve_problem(
    image_paths,
    "Compare the depth changes and object distribution across these images"
)
```

## Custom System Prompt

`SPAgent` accepts an optional `system_prompt` parameter. Use a built-in template or supply your own string. A `{tools_json}` placeholder is replaced automatically with the live tool schema; if omitted, the tools block is appended at the end.

```python
from spagent.core.prompts import GENERAL_VISION_SYSTEM_PROMPT, SPATIAL_3D_SYSTEM_PROMPT

# General vision agent (GroundingDINO + SAM2, no 3D instructions)
agent = SPAgent(
    model=GPTModel(model_name="gpt-4o"),
    tools=[ObjectDetectionTool(...), SegmentationTool(...)],
    system_prompt=GENERAL_VISION_SYSTEM_PROMPT,
)

# 3D spatial agent (default — same as omitting system_prompt)
agent = SPAgent(model=..., tools=[Pi3XTool(...)], system_prompt=SPATIAL_3D_SYSTEM_PROMPT)

# Fully custom prompt
agent = SPAgent(
    model=..., tools=tools,
    system_prompt="You are a specialist.\n<tools>\n{tools_json}\n</tools>\n..."
)
```

The same parameter is forwarded by `evaluate_tool_config`:

```python
evaluate_tool_config(..., system_prompt=GENERAL_VISION_SYSTEM_PROMPT)
```

## Video Generation (Veo / Sora / VACE)

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import VeoTool, SoraTool, VaceTool

model = GPTModel(model_name="gpt-4o")

# Text-to-video with Google Veo (requires GOOGLE_API_KEY)
agent = SPAgent(model=model, tools=[VeoTool()])
result = agent.solve_problem(
    "dummy",
    "Generate a video of a golden retriever running on a beach at sunset",
    video_num_frames=4
)

# Image-to-video with OpenAI Sora (requires OPENAI_API_KEY)
agent = SPAgent(model=model, tools=[SoraTool()])
result = agent.solve_problem(
    "assets/dog.jpeg",
    "Make the dog start running across the field",
    video_num_frames=4
)

# Local first-frame video with VACE (no cloud API, requires vace_server running)
agent = SPAgent(model=model, tools=[VaceTool(use_mock=False, server_url="http://localhost:20035")])
result = agent.solve_problem(
    "assets/example.png",
    "Generate a video showing the camera moving forward"
)
```

**Mock testing (no API key or GPU required):**

```bash
python test/test_tool.py --tool veo --image dummy --prompt "test" --use_mock
python test/test_tool.py --tool sora --image dummy --prompt "test" --use_mock
python test/test_tool.py --tool vace --image assets/example.png --prompt "move forward" --use_mock
```

## Image Generation (Sana)

Start the Sana service first:

```bash
bash scripts/run_sana_30000.sh
# Override model or GPU:
bash scripts/run_sana_30000.sh \
    --model-path Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
    --gpu-device 0 --port 30000
```

Then use `SanaTool` with SPAgent:

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import SanaTool

agent = SPAgent(
    model=GPTModel(model_name="gpt-4o"),
    tools=[SanaTool(use_mock=False, server_url="http://127.0.0.1:30000")],
    workflow_mode="auto"
)
result = agent.solve_problem(
    [],
    "Generate an image of a compact household robot organizing books on a wooden shelf."
)
print(result["answer"])
print(result["additional_images"])
```

**Evaluation:**

```bash
# Real service
python examples/evaluation/evaluate_sana.py \
    --config sana_real --data_path dataset/sana_sprint_cases_sample.jsonl \
    --max_samples 3 --max_workers 1 --max_iterations 1 --model gpt-4o

# Mock mode (no server needed)
python examples/evaluation/evaluate_sana.py \
    --config sana_mock --max_samples 3 --max_workers 1
```

## `step()` API and AgentMemory

`step()` is the general-purpose entry point for the agent. It replaces `solve_problem` (which remains as a backward-compatible wrapper) and returns a typed `StepResult`.

### One-shot use (stateless)

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool

agent = SPAgent(model=GPTModel(model_name="gpt-4o"), tools=[DepthEstimationTool(use_mock=True)])

result = agent.step("Analyze the depth relationships in this scene.", images="photo.jpg")
print(result.answer)             # final answer text
print(result.used_tools)         # e.g. ["depth_estimation_tool_iter1"]
print(result.additional_images)  # depth maps / vis images produced by tools
```

### Multi-turn stateful conversation

Pass a shared `AgentMemory` across calls — the agent accumulates context automatically.

```python
from spagent.core.memory import AgentMemory

memory = AgentMemory()

r1 = agent.step("Describe the scene.", images="photo.jpg", memory=memory)
print(r1.answer)

# Follow-up: memory carries the full history, no need to re-supply the image
r2 = agent.step("Now estimate the depth of the main object.", memory=memory)
print(r2.answer)

# Inspect memory
print(f"Total entries: {len(memory)}")
print(f"Tool images:   {memory.get_all_images()}")
print(f"Tool calls:    {[e.metadata['tool_name'] for e in memory.get_tool_calls()]}")
```

### Persist and restore a session

```python
memory.save("session.json")

memory = AgentMemory.load("session.json")
r3 = agent.step("What objects did we identify earlier?", memory=memory)
print(r3.answer)
```

### `StepResult` fields

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | Final answer text (may contain `<answer>` tags) |
| `memory` | `AgentMemory` | Updated memory after this step |
| `tool_calls` | `List[Dict]` | All tool-call dicts made this step |
| `tool_results` | `Dict[str, Any]` | Mapping of `tool_name_iterN → result` |
| `used_tools` | `List[str]` | Names of tools that succeeded |
| `additional_images` | `List[str]` | All image paths produced by tools |
| `iterations` | `int` | Number of tool-call iterations performed |
| `prompts` | `Dict[str, str]` | Key prompts used (system, user, workflow label) |

## Testing & Development

### Direct Tool Testing (without LLM)

Use `test/test_tool.py` to directly test any external expert tool without involving an LLM or Agent:

```bash
# Pi3 / Pi3X — 3D reconstruction
python test/test_tool.py --tool pi3  --image assets/dog.jpeg --azimuth 45 --elevation -30
python test/test_tool.py --tool pi3x --image assets/dog.jpeg --azimuth 45 --elevation -30

# Custom server URL
python test/test_tool.py --tool pi3 --image assets/dog.jpeg --server_url http://localhost:20030

# Multiple input images
python test/test_tool.py --tool pi3x --image img1.jpg img2.jpg --azimuth 45 --elevation -30

# Video generation (real API)
python test/test_tool.py --tool veo  --image dummy --prompt "A golden retriever running on a beach" --duration 8
python test/test_tool.py --tool sora --image assets/dog.jpeg --prompt "The dog starts running" --duration 5

# Mock mode (no API key or GPU required)
python test/test_tool.py --tool veo  --image dummy --prompt "test" --use_mock
python test/test_tool.py --tool sora --image dummy --prompt "test" --use_mock
python test/test_tool.py --tool vace --image assets/example.png --prompt "move forward" --use_mock
```

Call directly from Python:

```python
from test.test_tool import test_pi3

output_path = test_pi3(
    image_paths=["assets/dog.jpeg"],
    azimuth_angle=45,
    elevation_angle=-30,
    server_url="http://localhost:20030"
)
print(f"Rendered image saved to: {output_path}")
```

### Test Scripts Overview

| Script | Description |
|--------|-------------|
| `test/test_tool.py` | Direct tool testing without LLM (Pi3, Depth, Detection, Veo, Sora, WildDet3D, …) |
| `test/test_orient_anything_v2_tool.py` | OrientAnythingV2 — mock & real server modes |
| `test/test_pi3_llm.py` | Pi3 integration through Agent + LLM |
| `test/test_prompt.py` | Verify system prompt construction (no server or API key needed) |

```bash
python test/test_prompt.py                  # all cases
python test/test_prompt.py --case general   # general vision prompt
python test/test_prompt.py --case 3d        # 3D spatial prompt
```

### Real Service Mode

```python
tools = [
    DepthEstimationTool(use_mock=False, server_url="http://localhost:20019"),
    SegmentationTool(use_mock=False,    server_url="http://localhost:20020"),
    ZoomObjectTool(use_mock=False,      server_url="http://localhost:20022"),
    LocalizeObjectTool(use_mock=False,  server_url="http://localhost:20022"),
]
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
