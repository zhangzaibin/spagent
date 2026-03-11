import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spagent.core.spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import VGGTTool

# 1. Model
model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)

# 2. Tool (placeholder call, just returns success)
tools = [VGGTTool()]

# 3. Agent
agent = SPAgent(model=model, tools=tools)
print(f"Registered tools: {agent.list_tools()}")

# 4. Test: does the model call vggt_tool?
result = agent.solve_problem(
    "assets/dog.jpeg",
    "What is the 3D structure of this scene? Estimate the depth and camera parameters."
)

# 5. Check
print(f"Answer: {result['answer']}")
print(f"Used tools: {result['used_tools']}")
