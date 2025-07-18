# Workflows

This directory contains workflow implementations that orchestrate the interaction between VLLM models and external experts for spatial intelligence tasks.

## Depth QA Workflow

The `DepthQAWorkflow` implements a complete depth estimation question-answering pipeline that follows the workflow diagram structure:

### Workflow Components

1. **Initial VLLM Response**: VLLM first answers the question
2. **Tool Detection**: Check if VLLM's response indicates need for depth tool
3. **Depth Estimation**: Call depth estimation if needed
4. **Final VLLM Response**: VLLM provides final answer using both original image and depth map

### Usage

#### Simple Interface
```python
from workflows import infer

# Simple inference
result = infer(
    image_path="path/to/image.jpg",
    question="How is the depth distribution of objects in this image?"
)

print(result['answer'])  # Final answer
```

#### Advanced Interface
```python
from workflows.depth_qa_workflow import DepthQAWorkflow

# Create workflow instance (uses mock depth estimation by default)
workflow = DepthQAWorkflow(use_mock_depth=True)

# Run the workflow
result = workflow.run_workflow(
    image_path="path/to/image.jpg",
    question="How is the depth distribution of objects in this image?"
)

# Access results
print(result['answer'])  # Final answer
print(result['depth_used'])  # Whether depth tool was used
print(result['depth_result'])  # Depth estimation result (if used)
```

### Example

Run the example script to see the workflow in action:

```bash
cd spagent/workflows
python example_usage.py
```

### Configuration

- **Mock Mode**: Set `use_mock_depth=True` to use simulated depth estimation (default)
- **Real API Mode**: Set `use_mock_depth=False` to use the actual depth estimation API

### Output Structure

The workflow produces structured output with the following components:

- **action_instructions**: Commands for depth detection and analysis
- **visual_signals**: Depth map paths, keypoints, and bounding boxes
- **reasoning**: Analysis of depth information and spatial reasoning
- **answer**: Final textual response to the user's question
- **evaluation**: Quality metrics and recommendations

### Integration with External Experts

The workflow uses a simple approach to determine when to call external experts:

- VLLM first provides an initial answer to the question
- The system checks if the response contains depth-related keywords
- If depth information is needed, the depth estimation tool is called
- VLLM then provides a final answer using both the original image and the generated depth map

### Future Extensions

- Add support for SAM2 object detection
- Implement multi-modal reasoning workflows
- Add support for video input
- Integrate additional spatial intelligence experts 