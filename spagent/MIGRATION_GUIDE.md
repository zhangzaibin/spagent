# SPAgent Migration Guide

This guide helps you migrate from the old workflow-based architecture to the new SPAgent system.

## Overview of Changes

### Old Architecture (Workflows)
- Separate workflow classes for each tool combination
- Hard-coded tool orchestration
- Limited flexibility
- Difficult to extend

### New Architecture (SPAgent)
- Single SPAgent class with modular tools
- Dynamic tool composition
- Parallel tool execution
- Easy to extend and maintain

## Migration Examples

### 1. Depth QA Workflow → SPAgent

**Old Code:**
```python
from workflows.depth_qa_workflow import DepthQAWorkflow, infer

# Old workflow approach
workflow = DepthQAWorkflow(use_mock_depth=True)
result = workflow.run_workflow("image.jpg", "How is the depth distribution?")

# Or using the simple interface
result = infer("image.jpg", "How is the depth distribution?")
```

**New Code:**
```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool

# New SPAgent approach
model = GPTModel(model_name="gpt-4o-mini")
tools = [DepthEstimationTool(use_mock=True)]
agent = SPAgent(model=model, tools=tools)

result = agent.solve_problem("image.jpg", "How is the depth distribution?")
```

### 2. Mixed Expert Workflow → SPAgent

**Old Code:**
```python
from workflows.mix_workflow import MixedExpertWorkflow

workflow = MixedExpertWorkflow(
    ip="10.8.131.51", 
    port_depth=30750, 
    port_sam2=30646, 
    port_gdino=30969, 
    use_mock=True
)
result = workflow.run_workflow("image.jpg", "Analyze this image")
```

**New Code:**
```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import (
    DepthEstimationTool,
    SegmentationTool,
    ObjectDetectionTool
)

model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True),
    ObjectDetectionTool(use_mock=True)
]
agent = SPAgent(model=model, tools=tools)

result = agent.solve_problem("image.jpg", "Analyze this image")
```

### 3. SAM2 QA Workflow → SPAgent

**Old Code:**
```python
from workflows.sam2_qa_workflow import SAM2QAWorkflow

workflow = SAM2QAWorkflow(use_mock=True)
result = workflow.run_workflow("image.jpg", "Segment the objects")
```

**New Code:**
```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import SegmentationTool

model = GPTModel(model_name="gpt-4o-mini")
tools = [SegmentationTool(use_mock=True)]
agent = SPAgent(model=model, tools=tools)

result = agent.solve_problem("image.jpg", "Segment the objects")
```

## Key Benefits of Migration

### 1. Modular Design
- Tools are independent and reusable
- Easy to mix and match tools
- Clean separation of concerns

### 2. Dynamic Configuration
```python
# Add tools dynamically
agent.add_tool(DepthEstimationTool(use_mock=True))
agent.add_tool(SegmentationTool(use_mock=True))

# Remove tools when not needed
agent.remove_tool("depth_estimation_tool")

# Change models on the fly
agent.set_model(QwenModel(model_name="qwen2.5-vl-7b-instruct"))
```

### 3. Parallel Tool Execution
The new system automatically executes tools in parallel when possible, improving performance.

### 4. Better Error Handling
- Graceful degradation when tools fail
- Detailed error reporting
- Fallback strategies

### 5. Multi-Image Support
```python
# Analyze multiple images at once
result = agent.solve_problem(
    ["image1.jpg", "image2.jpg"], 
    "Compare these images"
)
```

## Available Tools

| Tool Class | Description | Replaces |
|------------|-------------|----------|
| `DepthEstimationTool` | Depth-AnythingV2 integration | `DepthQAWorkflow` |
| `SegmentationTool` | SAM2 integration | `SAM2QAWorkflow` |
| `ObjectDetectionTool` | GroundingDINO integration | `GdinoQAWorkflow` |
| `SupervisionTool` | Supervision integration | `SVQAWorkflow` |
| `YOLOETool` | YOLO-E integration | `YoloeQAWorkflow` |

## Available Models

| Model Class | Description | Original Function |
|-------------|-------------|-------------------|
| `GPTModel` | GPT integration | `gpt_single_image_inference` |
| `QwenModel` | Qwen API integration | `qwen_single_image_inference` |
| `QwenVLLMModel` | Qwen VLLM integration | `qwen_vllm_single_image_inference` |

## Complete Example

Here's a complete example showing how to create a fully-featured SPAgent:

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import (
    DepthEstimationTool,
    SegmentationTool,
    ObjectDetectionTool,
    SupervisionTool
)

# Create model
model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)

# Create tools
tools = [
    DepthEstimationTool(use_mock=True),
    SegmentationTool(use_mock=True),
    ObjectDetectionTool(use_mock=True),
    SupervisionTool(use_mock=True)
]

# Create agent
agent = SPAgent(model=model, tools=tools, max_workers=4)

# Solve a complex problem
result = agent.solve_problem(
    "image.jpg",
    "Analyze this image comprehensively: identify objects, their depth relationships, and segment important regions."
)

# Access results
print(f"Answer: {result['answer']}")
print(f"Tools used: {result['used_tools']}")
print(f"Additional images: {result['additional_images']}")
```

## Migration Checklist

- [ ] Replace workflow imports with SPAgent imports
- [ ] Create model and tool instances
- [ ] Initialize SPAgent with model and tools
- [ ] Replace `run_workflow()` calls with `solve_problem()`
- [ ] Update result access patterns
- [ ] Test with your specific use cases
- [ ] Remove old workflow code

## Backwards Compatibility

The old workflow system is still available but deprecated. You'll see warnings when importing workflows. Plan to migrate to SPAgent as the workflow system will be removed in a future version.

## Getting Help

- Check the example in `spagent/examples/spagent_example.py`
- Look at the tool and model class documentation
- The SPAgent class has comprehensive docstrings
- Old examples can be adapted using this migration guide 