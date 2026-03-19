from spagent import SPAgent
from spagent.models import GPTModel,QwenModel
from spagent.tools import DepthEstimationTool, SegmentationTool

# Create model and tools
model = GPTModel()
tools = [
    DepthEstimationTool(use_mock=False),    # Depth estimation
    SegmentationTool(use_mock=True)        # Image segmentation
]

# Create agent
agent = SPAgent(model=model, tools=tools)

# Solve problem
result = agent.solve_problem("spagent/assets/dog.jpeg", "Analyze the depth relationships and main objects in this image. Which object in the picture is closer to the camera?")
print(result['answer'])