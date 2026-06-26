"""
Autonomous 3-step agent run — Pi3X + GroundingDINO + SAM2.

Give one goal up front; the model decides what to do in each of 3 steps.
Uses 4 multi-view MindCube images of an office scene.

Run with:
    python test.py

Required servers (see docs/Tool/TOOL_USING.md):
  - GroundingDINO  localhost:20022
  - SAM2           localhost:20020
  - Pi3X           localhost:20031
"""

from pathlib import Path

from spagent.core import SPAgent, AgentMemory, GENERAL_VISION_CONTINUATION_HINT
from spagent.models import GPTModel
from spagent.tools import Pi3XTool, ObjectDetectionTool, SegmentationTool

NUM_STEPS = 3

model = GPTModel(model_name="gpt-4.1-mini", temperature=0.7)

tools = [
    ObjectDetectionTool(use_mock=False, server_url="http://localhost:20022"),
    SegmentationTool(use_mock=False, server_url="http://localhost:20020"),
    Pi3XTool(use_mock=False, server_url="http://localhost:20031"),
]

AGENT_ROLE = """\
You are a spatial reasoning agent. You have access to detection, segmentation,
and 3D reconstruction tools. Plan your own investigation — decide which tools
to call, in what order, and when you have enough evidence to answer.
Use tools when they help; skip them when they don't.\
"""

GOAL = """\
These images are multiple views of the same indoor scene.

Your goal: figure out the spatial layout of the room — especially the relative
positions of the office chair, white cabinet, and TV — and explain how the
scene looks from different viewpoints.

You have exactly 3 steps to work on this. In each step, decide for yourself
what to do next (observe, call tools, or conclude). On the last step, provide
your final answer in <answer></answer> tags.\
"""

agent = SPAgent(
    model=model,
    tools=tools,
    max_workers=3,
    system_prompt=AGENT_ROLE,
    continuation_hint=GENERAL_VISION_CONTINUATION_HINT,
    workflow_mode="auto",
)

IMAGE_DIR = Path(
    "dataset/mindcube/data/other_all_image/around/"
    "1be0927f8925c30aa632ee35e0513d8d4ae4a77092330fed20757d92568fe6e0"
)
IMAGES = sorted(str(p) for p in IMAGE_DIR.glob("*.png"))
if not IMAGES:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

print(f"Loaded {len(IMAGES)} views:")
for i, path in enumerate(IMAGES, 1):
    print(f"  [{i}] {path}")

memory = AgentMemory()
results = []

print("=" * 60)
print(f"GOAL:\n{GOAL.strip()}")
print("=" * 60)

for step_idx in range(1, NUM_STEPS + 1):
    print(f"\n[STEP {step_idx}/{NUM_STEPS}]")

    if step_idx == 1:
        content = GOAL
        images = IMAGES
    else:
        content = f"Step {step_idx}/{NUM_STEPS}. Continue toward the goal."
        images = None  # memory already holds the original views

    result = agent.step(
        content=content,
        images=images,
        memory=memory,
        max_tool_iterations=3,
    )
    results.append(result)

    print(result.answer)
    if result.used_tools:
        print(f"  → tools used this step: {result.used_tools}")
    if result.additional_images:
        print(f"  → new images: {result.additional_images}")

print("\n" + "=" * 60)
print("SESSION DEBRIEF")
print("=" * 60)
print(f"Total memory entries : {len(memory)}")
print(f"All tool calls       : {[e.metadata['tool_name'] for e in memory.get_tool_calls()]}")
print(f"All output images    : {memory.get_all_images()}")

memory.save("outputs/autonomous_session.json")
print("\nSession saved to outputs/autonomous_session.json")
