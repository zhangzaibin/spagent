"""
Unified Prompt Templates

This module contains prompt templates for the SPAgent system.
"""

from typing import List, Dict, Any, Optional
import json


# ─── Reusable workflow instruction blocks ────────────────────────────────────

SPATIAL_3D_WORKFLOW = """# Multi-Step Workflow
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
4. **If you have multiple input images**: Try different rotation_reference_camera values (1, 2, 3, etc.) to see the scene from different camera positions base on your analysis on the question.
5. **Consider using camera_view=true** to get first-person perspective from specific camera positions, especially useful for understanding spatial relationships and what each camera can actually see
6. After each round, analyze whether additional angles, camera positions, or perspective modes would reduce uncertainty
8. Continue until additional views no longer change your conclusion
9. Only put number (like 1,2,3) or Options in <answer></answer> tags, do not put any other text.


Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1 represents the first frame.
You can examine the image to understand what is around cam1.
The 3D reconstruction provides relative positional information, so you should reason interactively and complementarily between the 2D image and the 3D reconstruction to form a complete understanding.
You need to analyze deeply the camera, its orientation, and the content captured in the frame.

TIPS: For questions related to orientation or relative positioning, it is recommended to choose top view."""

# ─── Continuation hints (injected into the multi-step iteration prompt) ──────

SPATIAL_3D_CONTINUATION_HINT = """1. **Continue investigating** - Call tools with DIFFERENT parameters:
   - **IMPORTANT**: Your original input images are already at (azimuth=0°, elevation=0°). DO NOT call Pi3 tools with (0°, 0°) again!
   - For Pi3 tools: Try NEW viewing angles to understand the 3D structure better
   - Recommended NEW angles (NOT 0°,0°!):
     * Left: (-45°, 0°) or (-90°, 0°)
     * Right: (45°, 0°) or (90°, 0°)
     * Top: (0°, 45°) or (0°, 60°)
     * Bottom: (0°, -45°)
     * Back: (180°, 0°) or (±135°, 0°)
     * Diagonal: (45°, 30°) or (-45°, 30°)
   - Each NEW angle reveals different aspects of the 3D structure

   **Advanced Pi3 Parameters**:
   - **rotation_reference_camera** (integer, 1-based): When you have multiple input images, try DIFFERENT camera positions as rotation centers
     * Default is 1 (first camera), Set to 2, 3, etc. to rotate around different camera positions
   - **camera_view** (boolean): False = global bird's-eye view; True = first-person camera view

2. **Provide final answer** - If you have sufficient information, output your analysis in <think></think> and final answer in <answer></answer>.

Instructions:
- Think: Do you need to see the object from another NEW angle (NOT 0°,0°!) to answer the question better?
- If YES: Use <tool_call></tool_call> to request a DIFFERENT viewing angle (avoid 0°,0° as you already have it!)
- If NO: output your thinking process in <think></think> and your final answer in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text.

Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1 represents the first frame.
The 3D reconstruction provides relative positional information, so reason interactively and complementarily between 2D images and the 3D reconstruction."""

GENERAL_VISION_CONTINUATION_HINT = """1. **Continue investigating** - Call the available tools with different parameters or on different regions if needed.

2. **Provide final answer** - If you have gathered sufficient information:

Instructions:
- Think: Do you need more information from the tools to answer confidently?
- If YES: Use <tool_call></tool_call> to call a tool with appropriate parameters.
- If NO: output your thinking process in <think></think> and your final answer in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text.

Tool usage policy for image generation tools such as Sana:
- Use image generation only when you need to visualize a hypothetical scene, target state, plan outcome, or imagined world state.
- Do not use generated images as direct evidence about the original observation.
- For factual understanding of the provided input image(s), prefer analysis tools such as detection, segmentation, depth, or 3D reasoning tools first."""


GENERAL_VISION_WORKFLOW = """# Multi-Step Workflow
You can perform MULTIPLE rounds of tool calls and analysis to thoroughly understand the image.

Workflow:
1. Carefully analyze the image(s) and the question
2. Decide which tools would help gather additional information
3. Call tools with appropriate parameters — you may call the same tool multiple times with different inputs
4. After each tool result, assess whether you need more information before answering
5. Continue until you have sufficient evidence to answer confidently
6. Treat outputs from image generation tools such as Sana as synthetic visualizations of hypotheses or desired outcomes, not as direct observations
7. Only put number (like 1,2,3) or Options in <answer></answer> tags, do not put any other text.

Tool usage policy for image generation tools such as Sana:
- Use Sana when you need to visualize a hypothetical scene, target state, plan outcome, or imagined world state.
- Do not use Sana to replace factual perception of the original image.
- If the task is about what is actually present in the provided image(s), prefer observation tools before generation tools."""


GENERATION_CONTINUATION_HINT = """Instructions:
- If generation has not been attempted and is needed, call the generation tool directly.
- If generation succeeded, do not continue analyzing. Return a short final response in <answer></answer>.
- If generation failed, you may retry once with adjusted parameters or return a short failure summary in <answer></answer>.
- Keep reasoning minimal and avoid repeated self-reflection.
- Do not use generated images as factual evidence about real observations."""


GENERATION_WORKFLOW = """# Generation Workflow
You are an execution-oriented multimodal generation assistant.

Your task is to decide whether to call a generation tool and, if needed, call it with clean and specific parameters.

Rules:
1. Prefer action over long deliberation.
2. Do not produce long reasoning or repeated self-reflection.
3. If the request is clearly a generation task, directly call the relevant generation tool.
4. Use concise prompts and only include essential parameters.
5. After successful generation, provide a short final response in <answer></answer>.
6. Treat outputs from generation tools as synthetic visualizations, not as direct observations.

Tool usage policy for generation tools such as Sana:
- Use Sana when you need to visualize a hypothetical scene, target state, plan outcome, or imagined world state.
- Do not use Sana to replace factual perception.
- Keep <think></think> very short and focused on the action decision."""

# ─── Full system prompt templates (with {tools_json} placeholder) ─────────────

SPATIAL_3D_SYSTEM_PROMPT = (
    "You are a helpful assistant that can analyze images and answer questions.\n\n"
    "# Tools\n"
    "You have access to the following tools to assist with user queries:\n"
    "<tools>\n"
    "{tools_json}\n"
    "</tools>\n\n"
    "# How to call a tool\n"
    "When you need to use a tool, return a JSON object with the function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": "<function-name>", "arguments": {"param1": "value1", "param2": "value2"}}\n'
    "</tool_call>\n\n"
    "You can call multiple tools if needed by using multiple <tool_call> blocks.\n\n"
    + SPATIAL_3D_WORKFLOW
)

GENERAL_VISION_SYSTEM_PROMPT = (
    "You are a helpful visual assistant that can analyze images and answer questions.\n\n"
    "# Tools\n"
    "You have access to the following tools to assist with your analysis:\n"
    "<tools>\n"
    "{tools_json}\n"
    "</tools>\n\n"
    "# How to call a tool\n"
    "When you need to use a tool, return a JSON object with the function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": "<function-name>", "arguments": {"param1": "value1", "param2": "value2"}}\n'
    "</tool_call>\n\n"
    "You can call multiple tools if needed by using multiple <tool_call> blocks.\n\n"
    + GENERAL_VISION_WORKFLOW
)

GENERATION_SYSTEM_PROMPT = (
    "You are a helpful multimodal generation assistant.\n\n"
    "# Tools\n"
    "You have access to the following tools to assist with generation tasks:\n"
    "<tools>\n"
    "{tools_json}\n"
    "</tools>\n\n"
    "# How to call a tool\n"
    "When you need to use a tool, return a JSON object with the function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": "<function-name>", "arguments": {"param1": "value1", "param2": "value2"}}\n'
    "</tool_call>\n\n"
    "If the request is clearly a generation task, prefer calling the tool directly instead of long analysis.\n\n"
    + GENERATION_WORKFLOW
)


def create_system_prompt(tools: List[Dict[str, Any]], workflow: Optional[str] = None) -> str:
    """
    Create system prompt with available tools.

    Args:
        tools: List of tool function schemas
        workflow: Optional workflow instruction block to override the default
                  3D spatial workflow. Use one of the SPATIAL_3D_WORKFLOW or
                  GENERAL_VISION_WORKFLOW constants, or supply your own string.

    Returns:
        System prompt string
    """
    if not tools:
        return "You are a helpful assistant that can analyze images and answer questions."

    tools_json = json.dumps(tools, indent=2)
    chosen_workflow = workflow if workflow is not None else SPATIAL_3D_WORKFLOW

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

{chosen_workflow}
"""


def create_follow_up_prompt(
    question: str,
    initial_response: str,
    tool_results: Dict[str, Any],
    original_images: List[str],
    additional_images: List[str],
    description: str = None,
    continuation_hint: Optional[str] = None,
) -> str:
    """
    Create follow-up prompt after tool execution.

    Args:
        question: Original user question
        initial_response: Model's initial response
        tool_results: Results from tool execution
        original_images: List of original image paths
        additional_images: List of additional image paths from tools
        description: Optional description from tool execution
        continuation_hint: Optional next-step instructions injected at the end.
            Defaults to SPATIAL_3D_CONTINUATION_HINT when None.  Pass
            GENERAL_VISION_CONTINUATION_HINT (or a custom string) to avoid
            3D-specific instructions appearing in the prompt.

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

    hint = continuation_hint if continuation_hint is not None else SPATIAL_3D_CONTINUATION_HINT

    prompt += f"""

Now please provide a detailed final answer that incorporates the tool results with your initial analysis. If tools provided additional images or data, reference them in your response.

{hint}

You MUST output your thinking process in <think></think> and final choice in <answer></answer>.
"""

    return prompt

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
