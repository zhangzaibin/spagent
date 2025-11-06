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
You can perform MULTIPLE rounds of tool calls and analysis. When using 3D reconstruction tools (Pi3), autonomously explore viewpoints:

**IMPORTANT: The input image(s) already show the scene at (azimuth=0°, elevation=0°) viewpoint. DO NOT call Pi3 tools with (0°, 0°) as it will just return the same view you already have!
The camera is visualized as a pyramid frustum, where the apex represents the camera's position and viewing direction.**


# Recommended NEW viewing angles to explore:
- Left views: azimuth=-45° or -90° (see scenes from right view)
- Right views: azimuth=45° or 90° (see scenes from left view)
- Top views: elevation=30° to 60° (see scenes from top view, better capture the object relation and relatifve position of cam and objects.)
- Back views: azimuth=180° or ±135° (see scenes from back view)
- Diagonal views: combine azimuth and elevation (e.g., 45°, 30°)

Workflow:
1. Analyze the current view(s) you have
2. Decide which NEW angles (NOT 0°,0°!) would help answer the question
3. Call tools with specific angles that are DIFFERENT from (0°,0°)
4. After each round, analyze whether additional angles would reduce uncertainty
5. Continue until additional views no longer change your conclusion
6. Only put number (like 1,2,3) or Options in <answer></answer> tags, do not put any other text.

Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1 represents the first frame.
You can examine the image to understand what is around cam1.
The 3D reconstruction provides relative positional information, so you should reason interactively and complementarily between the 2D image and the 3D reconstruction to form a complete understanding.
You need to analyze deeply the camera, its orientation, and the content captured in the frame.

TIPS: For questions related to orientation or relative positioning, it is recommended to choose top view.
"""


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

**Reminder: The original input image(s) are at (0°,0°). When calling pi3_tool again, explore DIFFERENT angles (NOT 0°,0°!) such as ±45°, ±90°, 180° for azimuth, or ±30° to ±60° for elevation.**

You MUST output your thinking process in <think></think> and final choice in <answer></answer>. 
"""

    return prompt

# TODO 这块我总觉得有点奇怪，对于If you think donot need tool, you can directly answer the question. At this time, you SHOULD output your thinking process in <think></think> and final choice in <answer></answer>.
def create_user_prompt(question: str, image_paths: List[str], tool_schemas: List[Dict[str, Any]] = None) -> str:
    """
    Create user prompt template
    
    Args:
        question: User's question
        image_paths: List of image paths to analyze
        tool_schemas: List of tool function schemas, optional
    Returns:
        Formatted user prompt
    """
    images_info = "\n".join([f"- {path}" for path in image_paths])
    base_prompt = f"""Please analyze the following image(s):

Images to analyze:
{images_info}

Question:
{question}

Think step by step to analyze the question and provide a detailed answer."""

    if tool_schemas:
        base_prompt += """

Important Notes:
- You can call tools MULTIPLE times with different parameters to gather comprehensive information
- After each tool execution, you'll see the results and can decide if you need more information
- Only provide your final <answer></answer> when you have gathered sufficient information

You MUST output your thinking process in <think></think> and tool choices in <tool_call></tool_call>. When you have enough information, output your final choice in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text."""
    else:
        base_prompt += """

You MUST output your thinking process in <think></think> and your final answer in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text."""

    return base_prompt 


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

You MUST output your thinking process in <think></think> and final choice in <answer></answer>. Only put Options (A,B,C,D) in <answer></answer> tags, do not put any other text.
"""
