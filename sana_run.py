from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import SanaTool

# Create model and tools
model = GPTModel(model_name="qwen3.5-35b-a3b")
tools = [
    SanaTool(
        use_mock=False,
        server_url="http://127.0.0.1:30000",
    )
]

# Create agent with automatic workflow routing
agent = SPAgent(
    model=model,
    tools=tools,
    workflow_mode="auto",
)

# Solve problem (text-only task, no input image needed)
result = agent.solve_problem(
    [],
    "Generate an image of a compact household robot organizing books on a wooden shelf in a warm study room."
)

print("Selected workflow:", result["prompts"].get("workflow"))
print("Answer:", result["answer"])
print("Tool calls:", result["tool_calls"])
print("Generated images:", result["additional_images"])
