"""
Multi-Step Workflow Example for SPAgent

This example demonstrates how to use the multi-step workflow feature
that allows the model to call tools multiple times with different angles.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spagent.core.spagent import SPAgent
from spagent.core.model import Model
from spagent.external_experts.Pi3.pi3_tool import Pi3Tool, Pi3MultiimgTool


def example_single_iteration():
    """Example: Traditional single iteration (backward compatible)"""
    print("=" * 60)
    print("Example 1: Single Iteration (Traditional Mode)")
    print("=" * 60)
    
    # Initialize model and agent
    model = Model(model_name="Qwen2-VL-7B-Instruct")
    agent = SPAgent(model=model)
    
    # Add Pi3 tool
    pi3_tool = Pi3Tool(server_url="http://localhost:20021")
    agent.add_tool(pi3_tool)
    
    # Solve problem with default max_iterations=1
    result = agent.solve_problem(
        image_path="path/to/your/image.jpg",
        question="What is the 3D structure of this object?",
        max_iterations=1  # Default value, can be omitted
    )
    
    print(f"Iterations: {result['iterations']}")
    print(f"Tool calls: {len(result['tool_calls'])}")
    print(f"Answer: {result['answer'][:200]}...")


def example_multi_step_workflow():
    """Example: Multi-step workflow with angle adjustments"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Step Workflow (3 iterations)")
    print("=" * 60)
    
    # Initialize model and agent
    model = Model(model_name="Qwen2-VL-7B-Instruct")
    agent = SPAgent(model=model)
    
    # Add Pi3 tool
    pi3_tool = Pi3Tool(server_url="http://localhost:20021")
    agent.add_tool(pi3_tool)
    
    # Solve problem with multiple iterations
    result = agent.solve_problem(
        image_path="path/to/your/image.jpg",
        question="""Analyze this object's 3D structure from multiple angles. 
        First get a front view, then check from the left side, 
        and finally from the top.""",
        max_iterations=3  # Allow up to 3 iterations
    )
    
    print(f"Iterations completed: {result['iterations']}")
    print(f"Total tool calls: {len(result['tool_calls'])}")
    print(f"Used tools: {result['used_tools']}")
    print(f"Generated images: {len(result['additional_images'])}")
    print(f"\nAnswer: {result['answer'][:200]}...")
    
    # Show tool call details
    print("\n--- Tool Call Details ---")
    for i, call in enumerate(result['tool_calls'], 1):
        print(f"\nCall {i}:")
        print(f"  Tool: {call['name']}")
        print(f"  Arguments: {call['arguments']}")


def example_with_specific_angles():
    """Example: Using specific angles for Pi3 rendering"""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Step with Specific Angles")
    print("=" * 60)
    
    # Initialize model and agent
    model = Model(model_name="Qwen2-VL-7B-Instruct")
    agent = SPAgent(model=model)
    
    # Add Pi3 tool
    pi3_tool = Pi3Tool(server_url="http://localhost:20021")
    agent.add_tool(pi3_tool)
    
    # The model will be instructed to use specific angles
    result = agent.solve_problem(
        image_path="path/to/your/image.jpg",
        question="""Generate 3D reconstruction with these viewing angles:
        1. Front view (azimuth=0, elevation=0)
        2. Left view (azimuth=-45, elevation=0)  
        3. Top view (azimuth=0, elevation=45)
        
        Then analyze the structure from all these perspectives.""",
        max_iterations=3
    )
    
    print(f"Iterations: {result['iterations']}")
    print(f"Tool calls: {len(result['tool_calls'])}")
    
    # Extract angle information from tool calls
    print("\n--- Viewing Angles Used ---")
    for i, call in enumerate(result['tool_calls'], 1):
        if 'arguments' in call:
            args = call['arguments']
            azimuth = args.get('azimuth_angle', 'default')
            elevation = args.get('elevation_angle', 'default')
            print(f"Call {i}: azimuth={azimuth}°, elevation={elevation}°")


def example_early_termination():
    """Example: Early termination when model provides answer before max_iterations"""
    print("\n" + "=" * 60)
    print("Example 4: Early Termination")
    print("=" * 60)
    
    # Initialize model and agent
    model = Model(model_name="Qwen2-VL-7B-Instruct")
    agent = SPAgent(model=model)
    
    # Add Pi3 tool
    pi3_tool = Pi3Tool(server_url="http://localhost:20021")
    agent.add_tool(pi3_tool)
    
    # Even with max_iterations=5, the workflow may end earlier
    # if the model provides a final answer
    result = agent.solve_problem(
        image_path="path/to/your/image.jpg",
        question="What is this object?",
        max_iterations=5
    )
    
    print(f"Max iterations allowed: 5")
    print(f"Actual iterations used: {result['iterations']}")
    print(f"Reason: Model provided final answer earlier")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("SPAgent Multi-Step Workflow Examples")
    print("=" * 60)
    
    # Note: These are demonstration examples
    # In practice, you need to:
    # 1. Start the Pi3 server (python pi3_server.py)
    # 2. Have actual image files
    # 3. Initialize your model properly
    
    print("\nNote: These are code examples showing the API usage.")
    print("To run them, you need to:")
    print("1. Start Pi3 server: python spagent/external_experts/Pi3/pi3_server.py")
    print("2. Replace 'path/to/your/image.jpg' with actual image paths")
    print("3. Configure your model properly")
    
    # Uncomment to run examples (after setup):
    # example_single_iteration()
    # example_multi_step_workflow()
    # example_with_specific_angles()
    # example_early_termination()


if __name__ == "__main__":
    main()

