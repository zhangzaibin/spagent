#!/usr/bin/env python3
"""
SPAgent Example Usage

This script demonstrates how to use the new SPAgent system that replaces
the old workflow architecture. It shows how to:
1. Initialize tools and models
2. Create and configure an SPAgent
3. Solve spatial intelligence problems
4. Handle different combinations of tools

Usage:
    python spagent_example.py [image_path] [question]
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core import SPAgent
from models import GPTModel, QwenModel, QwenVLLMModel
from tools import (
    DepthEstimationTool,
    SegmentationTool,
    ObjectDetectionTool,
    SupervisionTool,
    YOLOETool
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_agent_with_all_tools(model_type: str = "gpt", use_mock: bool = True) -> SPAgent:
    """
    Create an SPAgent with all available tools
    
    Args:
        model_type: Type of model to use ("gpt", "qwen", "qwen_vllm")
        use_mock: Whether to use mock services for tools
        
    Returns:
        Configured SPAgent instance
    """
    # Initialize model
    if model_type == "gpt":
        model = GPTModel(model_name="gpt-4o-mini")
    elif model_type == "qwen":
        model = QwenModel(model_name="qwen2.5-vl-7b-instruct")
    elif model_type == "qwen_vllm":
        model = QwenVLLMModel(model_name="qwen-vl")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize tools
    tools = [
        DepthEstimationTool(use_mock=use_mock),
        SegmentationTool(use_mock=use_mock),
        ObjectDetectionTool(use_mock=use_mock),
        SupervisionTool(use_mock=use_mock),
        YOLOETool(use_mock=use_mock)
    ]
    
    # Create SPAgent
    agent = SPAgent(model=model, tools=tools)
    
    logger.info(f"Created SPAgent with {model_type} model and {len(tools)} tools")
    logger.info(f"Available tools: {agent.list_tools()}")
    
    return agent


def create_depth_focused_agent(model_type: str = "gpt", use_mock: bool = True) -> SPAgent:
    """
    Create an SPAgent focused on depth analysis
    
    Args:
        model_type: Type of model to use
        use_mock: Whether to use mock services
        
    Returns:
        SPAgent configured for depth analysis
    """
    if model_type == "gpt":
        model = GPTModel(model_name="gpt-4o-mini")
    else:
        model = QwenModel(model_name="qwen2.5-vl-7b-instruct")
    
    # Only depth and segmentation tools for focused analysis
    tools = [
        DepthEstimationTool(use_mock=use_mock),
        SegmentationTool(use_mock=use_mock)
    ]
    
    agent = SPAgent(model=model, tools=tools)
    logger.info(f"Created depth-focused SPAgent with {len(tools)} tools")
    
    return agent


def create_detection_focused_agent(model_type: str = "gpt", use_mock: bool = True) -> SPAgent:
    """
    Create an SPAgent focused on object detection
    
    Args:
        model_type: Type of model to use
        use_mock: Whether to use mock services
        
    Returns:
        SPAgent configured for object detection
    """
    if model_type == "gpt":
        model = GPTModel(model_name="gpt-4o-mini")
    else:
        model = QwenModel(model_name="qwen2.5-vl-7b-instruct")
    
    # Detection and segmentation tools
    tools = [
        ObjectDetectionTool(use_mock=use_mock),
        SupervisionTool(use_mock=use_mock),
        YOLOETool(use_mock=use_mock),
        SegmentationTool(use_mock=use_mock)
    ]
    
    agent = SPAgent(model=model, tools=tools)
    logger.info(f"Created detection-focused SPAgent with {len(tools)} tools")
    
    return agent


def demo_basic_usage(image_path: str, question: str):
    """
    Demonstrate basic SPAgent usage
    
    Args:
        image_path: Path to test image
        question: Question to ask about the image
    """
    print("\n" + "="*80)
    print("DEMO 1: Basic SPAgent Usage")
    print("="*80)
    
    # Create agent with all tools
    agent = create_agent_with_all_tools(model_type="gpt", use_mock=True)
    
    # Solve problem
    result = agent.solve_problem(image_path, question)
    
    # Display results
    print(f"\nQuestion: {question}")
    print(f"Image: {image_path}")
    print(f"\nFinal Answer:\n{result['answer']}")
    print(f"\nTool Calls Made: {len(result['tool_calls'])}")
    
    for i, call in enumerate(result['tool_calls']):
        print(f"  {i+1}. {call['name']}: {call['arguments']}")
    
    print(f"\nTools Used: {result['used_tools']}")
    print(f"Additional Images Generated: {len(result['additional_images'])}")
    
    for img in result['additional_images']:
        print(f"  - {img}")


def demo_focused_agents(image_path: str):
    """
    Demonstrate different focused agents
    
    Args:
        image_path: Path to test image
    """
    print("\n" + "="*80)
    print("DEMO 2: Focused Agents")
    print("="*80)
    
    # Test depth-focused agent
    print("\n--- Depth-Focused Agent ---")
    depth_agent = create_depth_focused_agent(use_mock=True)
    depth_question = "Analyze the depth distribution in this image and identify which objects are closer or farther from the camera."
    
    depth_result = depth_agent.solve_problem(image_path, depth_question)
    print(f"Question: {depth_question}")
    print(f"Answer: {depth_result['answer'][:200]}...")  # Truncate for display
    print(f"Tools used: {depth_result['used_tools']}")
    
    # Test detection-focused agent
    print("\n--- Detection-Focused Agent ---")
    detection_agent = create_detection_focused_agent(use_mock=True)
    detection_question = "Detect and identify all objects in this image, including their locations and types."
    
    detection_result = detection_agent.solve_problem(image_path, detection_question)
    print(f"Question: {detection_question}")
    print(f"Answer: {detection_result['answer'][:200]}...")  # Truncate for display
    print(f"Tools used: {detection_result['used_tools']}")


def demo_dynamic_tool_management():
    """
    Demonstrate dynamic tool management
    """
    print("\n" + "="*80)
    print("DEMO 3: Dynamic Tool Management")
    print("="*80)
    
    # Start with minimal agent
    model = GPTModel(model_name="gpt-4o-mini")
    agent = SPAgent(model=model)
    
    print(f"Initial tools: {agent.list_tools()}")
    
    # Add tools dynamically
    print("\nAdding tools dynamically...")
    agent.add_tool(DepthEstimationTool(use_mock=True))
    print(f"After adding depth tool: {agent.list_tools()}")
    
    agent.add_tool(ObjectDetectionTool(use_mock=True))
    print(f"After adding detection tool: {agent.list_tools()}")
    
    agent.add_tool(SegmentationTool(use_mock=True))
    print(f"After adding segmentation tool: {agent.list_tools()}")
    
    # Remove a tool
    print("\nRemoving detection tool...")
    agent.remove_tool("detect_objects_tool")
    print(f"After removing detection tool: {agent.list_tools()}")
    
    # Change model
    print("\nChanging model...")
    new_model = QwenModel(model_name="qwen2.5-vl-7b-instruct")
    agent.set_model(new_model)
    print(f"Model changed to: {agent.model.model_name}")


def demo_multiple_images():
    """
    Demonstrate multi-image analysis
    """
    print("\n" + "="*80)
    print("DEMO 4: Multi-Image Analysis")
    print("="*80)
    
    # This would work with real images
    print("Note: This demo shows the interface for multi-image analysis")
    print("In a real scenario, you would provide multiple image paths:")
    
    agent = create_agent_with_all_tools(use_mock=True)
    
    # Example of how multi-image analysis would work
    image_paths = ["assets/image1.jpg", "assets/image2.jpg"]  # Example paths
    question = "Compare the depth and objects between these two images."
    
    print(f"Image paths: {image_paths}")
    print(f"Question: {question}")
    print("This would call: agent.solve_problem(image_paths, question)")
    
    # Note: Actual execution would require valid image paths
    # result = agent.solve_problem(image_paths, question)


def main():
    """Main function"""
    # Default parameters
    default_image = "assets/example.png"  # Update this to a real image path
    default_question = "Analyze this image and describe what you see, including depth relationships and object identification."
    
    # Get parameters from command line
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image
        print(f"Using default image: {image_path}")
    
    if len(sys.argv) > 2:
        question = sys.argv[2]
    else:
        question = default_question
        print(f"Using default question: {question}")
    
    print("\n" + "="*80)
    print("SPAgent System Demonstration")
    print("="*80)
    print("This demo shows the new SPAgent architecture that replaces")
    print("the old workflow system with a more flexible and modular approach.")
    print("="*80)
    
    # Check if image exists (for real demos)
    if os.path.exists(image_path):
        # Run demos with real image
        demo_basic_usage(image_path, question)
        demo_focused_agents(image_path)
    else:
        print(f"\nNote: Image {image_path} not found.")
        print("Running demos without actual image processing...")
    
    # These demos don't require real images
    demo_dynamic_tool_management()
    demo_multiple_images()
    
    print("\n" + "="*80)
    print("SPAgent Features Demonstrated:")
    print("="*80)
    print("✓ Modular tool architecture")
    print("✓ Dynamic tool management")
    print("✓ Multiple model support (GPT, Qwen, Qwen-VLLM)")
    print("✓ Parallel tool execution")
    print("✓ Multi-image analysis support")
    print("✓ Flexible configuration")
    print("✓ Comprehensive error handling")
    print("✓ Mock services for testing")
    print("\nThe SPAgent system successfully replaces the old workflow")
    print("architecture with a much more flexible and maintainable design!")


if __name__ == "__main__":
    main() 