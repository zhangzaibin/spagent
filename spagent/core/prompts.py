"""
Unified Prompt Templates

This module contains prompt templates for the SPAgent system.
"""

from typing import List, Dict, Any
import json


def create_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Create system prompt with available tools
    
    Args:
        tools: List of tool function schemas
        
    Returns:
        System prompt string
    """
    if not tools:
        return """You are a helpful assistant that can analyze images and answer questions."""
    
    tools_json = json.dumps(tools, indent=2)
    
    return f"""You are a helpful assistant that can analyze images and answer questions.

# Tools
You have access to the following tools to assist with user queries:
<tools>
{tools_json}
</tools>

# How to call a tool
When you need to use a tool, return a JSON object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

You can call multiple tools if needed by using multiple <tool_call> blocks.

# Multi-Step Workflow
You can perform MULTIPLE rounds of tool calls and analysis. When using 3D reconstruction tools, autonomously explore viewpoints:
1. Start from the reference view: (azimuth=0, elevation=0) MUST match the first input image (cam1)
2. Use the camera coordinate frame for rotations: azimuth rotates left/right around the camera vertical axis; elevation rotates up/down around the camera right axis
3. Execute a coarse-to-fine exploration: try canonical views (e.g., left/right ±45°, top/bottom ±45°), then refine to ±15° near ambiguous regions
4. After each round, analyze whether additional angles would reduce uncertainty; if yes, call the tool again with updated angles
5. Continue until additional views no longer change your conclusion, then provide your comprehensive answer in <answer></answer> tags

For 3D reconstruction tools like pi3_tool:
- (0°, 0°) is exactly cam1 (the first input image). Rotations are in the CAMERA frame.
- You can specify azimuth_angle (-180° to 180°) and elevation_angle (-90° to 90°)
- Recommended sequence: (0,0) → (-45,0) → (45,0) → (0,45) → (0,-45), then refine as needed (e.g., ±15°)
- Each call generates a new visualization from that specific viewpoint

# Instructions
1. First analyze the user's question and the image(s) provided
2. Determine if you need to use any tools to answer the question properly
3. If tools are needed, call them with appropriate parameters
4. After seeing tool results, decide if you need MORE information from different angles or perspectives
5. Prefer autonomous exploration of angles as described above to reduce uncertainty
6. You can make multiple tool calls across multiple rounds to gather comprehensive information
7. When you have sufficient information, provide your final answer in <answer></answer> tags
8. Be specific and detailed in your responses"""


def create_follow_up_prompt(question: str, initial_response: str, tool_results: Dict[str, Any], original_images: List[str], additional_images: List[str], description: str=None) -> str:
    """
    Create follow-up prompt after tool execution
    
    Args:
        question: Original user question
        initial_response: Model's initial response
        tool_results: Results from tool execution
        original_images: List of original image paths
        additional_images: List of additional image paths from tools
        description: Optional description from tool execution
        
    Returns:
        Follow-up prompt string
    """
    tool_summary = []
    for tool_name, result in tool_results.items():
        if result.get('success'):
            tool_summary.append(f"- {tool_name}: Successfully executed")
        else:
            tool_summary.append(f"- {tool_name}: Failed - {result.get('error', 'Unknown error')}")
    
    original_images_info = "\n".join([f"- {path}" for path in original_images])
    additional_images_info = "\n".join([f"- {path}" for path in additional_images]) if additional_images else "None"
    
    # 构建基本的 prompt
    prompt = f"""Based on the tool results, please provide a comprehensive answer to the original question.

Original Images:
{original_images_info}

Additional Images from Tools:
{additional_images_info}

Original Question: {question}

Your Initial Analysis: {initial_response}

Tool Execution Summary:
{chr(10).join(tool_summary)}"""

    # 只有当提供了 description 时才添加 Tool Description 部分
    if description is not None:
        prompt += f"""

Tool Description: {description}"""

    prompt += """

Now please provide a detailed final answer that incorporates the tool results with your initial analysis. If tools provided additional images or data, reference them in your response.
Angle conventions reminder: (0°,0°)=cam1; rotations are in the camera coordinate frame.
You MUST output your thinking process in <think></think> and final choice in <answer></answer>. 
"""

    return prompt

# TODO 这块我总觉得有点奇怪，对于If you think donot need tool, you can directly answer the question. At this time, you SHOULD output your thinking process in <think></think> and final choice in <answer></answer>.
def create_user_prompt(question: str, image_paths: List[str]) -> str:
    """
    Create user prompt template
    
    Args:
        question: User's question
        image_paths: List of image paths to analyze
        
    Returns:
        Formatted user prompt
    """
    images_info = "\n".join([f"- {path}" for path in image_paths])
    return f"""Please analyze the following image(s):

Images to analyze:
{images_info}

Question:
{question}

Think step by step and use any available tools if they would help provide a better answer.

Important Notes:
- You can call tools MULTIPLE times with different parameters to gather comprehensive information
- For 3D reconstruction, use autonomous angle exploration in the CAMERA coordinate frame: (0°,0°)=cam1; explore ±45° first, then refine (±15°)
- After each tool execution, you'll see the results and can decide if you need more information
- Only provide your final <answer></answer> when you have gathered sufficient information

You MUST output your thinking process in <think></think> and tool choices in <tool_call></tool_call>. When you have enough information, output your final choice in <answer></answer>. 

""" 


def create_fallback_prompt(question: str, initial_response: str) -> str:
    """
    Create fallback prompt when tools fail but initial response lacks <answer> tags
    
    Args:
        question: Original question
        initial_response: Initial model response
        
    Returns:
        Fallback prompt string
    """
    return f"""The tools could not be executed successfully. Based on your initial analysis, please provide a final answer to the question.

Original Question: {question}

Your Initial Analysis: {initial_response}

Since the tools are unavailable, please provide your best answer based on the original image analysis alone.

You MUST output your thinking process in <think></think> and final choice in <answer></answer>.
"""
