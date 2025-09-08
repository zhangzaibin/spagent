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
        return """You are a helpful assistant that can analyze images and answer questions. You MUST output your thinking process in <think></think> and final choice in <answer></answer>."""
    
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

# Instructions
1. First analyze the user's question and the image(s) provided
2. Determine if you need to use any tools to answer the question properly
3. If tools are needed, call them with appropriate parameters
4. Provide a comprehensive answer based on your analysis and any tool results
5. Be specific and detailed in your responses"""


def create_follow_up_prompt(question: str, initial_response: str, tool_results: Dict[str, Any], original_images: List[str], additional_images: List[str]) -> str:
    """
    Create follow-up prompt after tool execution
    
    Args:
        question: Original user question
        initial_response: Model's initial response
        tool_results: Results from tool execution
        original_images: List of original image paths
        additional_images: List of additional image paths from tools
        
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
    
    return f"""Based on the tool results, please provide a comprehensive answer to the original question.

Original Images:
{original_images_info}

Additional Images from Tools:
{additional_images_info}

Original Question: {question}

Your Initial Analysis: {initial_response}

Tool Execution Summary:
{chr(10).join(tool_summary)}

Now please provide a detailed final answer that incorporates the tool results with your initial analysis. If tools provided additional images or data, reference them in your response. 
You MUST output your thinking process in <think></think> and final choice in <answer></answer>. 
"""

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


If you think donot need tool, you can directly answer the question. At this time, you SHOULD output your thinking process in <think></think> and final choice in <answer></answer>. 
""" 