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
- If NO: output your thinking process in <think></think> and your final answer in <answer></answer>. Put your concise final answer in <answer></answer> tags."""

SKILL_VISION_CONTINUATION_HINT = """1. **Continue investigating** - Call already-activated tools with <tool_call></tool_call>, or select new skills with <skill_select></skill_select> if needed.

2. **Provide final answer** - If you have gathered sufficient information:

Instructions:
- Think: Do you need more information from the tools to answer confidently?
- If YES and you already have the tool instructions: Use <tool_call></tool_call> to call a tool with appropriate parameters.
- If YES but you need a different tool: Use <skill_select>skill_name</skill_select> to select a new skill first.
- If NO: output your thinking process in <think></think> and your final answer in <answer></answer>. Put your concise final answer in <answer></answer> tags."""


GENERAL_VISION_WORKFLOW = """# Multi-Step Workflow
You can perform MULTIPLE rounds of tool calls and analysis to thoroughly understand the image.
"""

SKILL_VISION_WORKFLOW = """# Multi-Step Workflow
You can perform MULTIPLE rounds of skill selection and tool calls to thoroughly understand the image.

Workflow:
1. Carefully analyze the image(s) and the question
2. Decide which skill(s) would help gather additional information — select them with <skill_select></skill_select>
3. After receiving the full usage instructions for each selected skill, call the corresponding tool(s) with <tool_call></tool_call> following the provided instructions
4. After each tool result, assess whether you need more information before answering
5. You may select additional skills or call already-activated tools again with different parameters
6. Continue until you have sufficient evidence to answer confidently
7. Put your concise final answer in <answer></answer> tags."""

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


# ─── Skill-mode prompt helpers (progressive disclosure) ───────────────────────

def create_skill_system_prompt(skill_index: str, workflow: Optional[str] = None) -> str:
    """
    Create system prompt for skill mode (progressive disclosure).

    In skill mode the model initially sees only a compact skill index
    (name + summary).  It must output <skill_select>name</skill_select>
    to receive the full usage instructions before it can call the tool.

    Args:
        skill_index: XML string from SkillRegistry.get_skill_index()
        workflow: Optional workflow block.  Defaults to SKILL_VISION_WORKFLOW.

    Returns:
        System prompt string with skill index
    """
    chosen_workflow = workflow if workflow is not None else SKILL_VISION_WORKFLOW

    return f"""You are a helpful assistant that can analyze images and answer questions.

# Skills
You have access to the following skills to assist with your analysis.
Each skill wraps a specialized tool. You can see the skill name and a short summary below:

<available_skills>
{skill_index}
</available_skills>

# How to select a skill
When you need to use a skill, output its name inside <skill_select></skill_select> XML tags:
<skill_select>skill_name</skill_select>

You can select multiple skills at once by using multiple <skill_select> blocks.
After selection, you will receive the full usage instructions (parameters, call format, etc.) for each selected skill, and then you can call the tool with <tool_call></tool_call> tags.

{chosen_workflow}
"""


def create_skill_activation_prompt(
    activated_skills: list,
    question: str = "",
    image_paths: list = None,
    previous_analysis: str = "",
) -> str:
    """
    Build a prompt that injects full usage instructions for selected skills
    and asks the model to immediately call the tool(s).

    Args:
        activated_skills: List of (skill_title, skill_usage_prompt) tuples
        question: The original user question (for context)
        image_paths: List of image paths being analyzed
        previous_analysis: The model's Step-1 response (skill selection + thinking)

    Returns:
        Activation prompt string
    """
    blocks = []
    for title, usage in activated_skills:
        blocks.append(
            f"# Skill Activated: {title}\n\n"
            f"{usage}"
        )
    skill_instructions = "\n\n---\n\n".join(blocks)

    images_info = ""
    if image_paths:
        images_info = "Images: " + ", ".join(image_paths) + "\n"

    previous_block = ""
    if previous_analysis:
        previous_block = f"""# Your Previous Analysis
{previous_analysis}

---

"""

    return f"""{previous_block}The following skill(s) have been activated. You now have the full usage instructions.

{skill_instructions}

# How to call a tool
When you need to use a tool, return a JSON object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

You can call multiple tools by using multiple <tool_call> blocks.

---

{images_info}Question: {question}

Now call the tool(s) with concrete arguments based on the instructions above.
Output your thinking in <think></think> and your tool call(s) in <tool_call></tool_call>."""


# ─── Legacy full-schema system prompt (backward compatible) ───────────────────

def create_system_prompt(tools: List[Dict[str, Any]], workflow: Optional[str] = None) -> str:
    """
    Create system prompt with available tools (legacy full-schema mode).

    Args:
        tools: List of tool function schemas
        workflow: Optional workflow instruction block.
                  Defaults to GENERAL_VISION_WORKFLOW.

    Returns:
        System prompt string
    """
    if not tools:
        return "You are a helpful assistant that can analyze images and answer questions."

    tools_json = json.dumps(tools, indent=2)
    chosen_workflow = workflow if workflow is not None else GENERAL_VISION_WORKFLOW

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

    hint = continuation_hint if continuation_hint is not None else GENERAL_VISION_CONTINUATION_HINT

    prompt += f"""

Now please provide a detailed final answer that incorporates the tool results with your initial analysis. If tools provided additional images or data, reference them in your response.

{hint}

You MUST output your thinking process in <think></think> and final choice in <answer></answer>.
"""

    return prompt

def create_user_prompt(
    question: str,
    image_paths: List[str],
    tool_schemas: List[Dict[str, Any]] = None,
    use_skill_mode: bool = False,
) -> str:
    """
    Create user prompt template.

    Args:
        question: User's question
        image_paths: List of image paths to analyze
        tool_schemas: List of tool function schemas (legacy mode)
        use_skill_mode: Whether skill-based progressive disclosure is active
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

    if use_skill_mode:
        base_prompt += """

Important Notes:
- You can select skills with <skill_select></skill_select> to get detailed tool usage instructions
- After receiving instructions, you can call tools MULTIPLE times with different parameters to gather comprehensive information
- After each tool execution, you'll see the results and can decide if you need more information
- Only provide your final <answer></answer> when you have gathered sufficient information
Notice! You MUST output the following format. You MUST output thinking process in <think> thinking process here </think> and skill selections in <skill_select> skill name here </skill_select> tags."""
    elif tool_schemas:
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
