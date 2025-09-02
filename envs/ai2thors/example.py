# Initialize the agent
agent = GPTThorAgent(
    model_name="gpt-4o-mini",  # Your GPT model name
    scene="FloorPlan1"         # AI2THOR scene
)

# Define a goal
goal = "Find and pick up the apple on the kitchen counter"

# Take actions toward the goal
state = agent.step(goal)

# Check results
print(f"Action succeeded: {state['success']}")
if not state['success']:
    print(f"Error: {state['error']}")

# Get visible objects
visible_objects = [
    obj["objectType"] 
    for obj in state["metadata"]["objects"]
    if obj.get("visible", False)
]
print(f"Visible objects: {visible_objects}")

# Cleanup
agent.close()
