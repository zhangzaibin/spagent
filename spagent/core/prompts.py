"""
Unified Prompt Templates

This module contains prompt templates for the SPAgent system.

Three-layer prompt architecture
--------------------------------
1. Role prompt  (user-replaceable)  — what the agent is / what it should do.
                                      Use SPATIAL_3D_ROLE, GENERAL_VISION_ROLE,
                                      GENERATION_ROLE, or supply your own string.
2. Tool block   (auto-appended)     — tool list + <tool_call> wire format.
                                      Always added by build_system_prompt().
3. Workflow     (optional preset)   — multi-step iteration guidance.
                                      Use SPATIAL_3D_WORKFLOW,
                                      GENERAL_VISION_WORKFLOW, or GENERATION_WORKFLOW.

Quick usage::

    from spagent.core.prompts import build_system_prompt, GENERAL_VISION_WORKFLOW
    system_prompt = build_system_prompt(
        role_prompt="You are a drone navigation assistant.",
        tools_json=json.dumps(tool_schemas, indent=2),
        workflow=GENERAL_VISION_WORKFLOW,
    )
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

Detection tool guidance:
- **zoom_object_tool** → use for attribute questions (color, texture, material, text, pattern). Returns close-up crops.
- **localize_object_tool** → use for spatial/counting questions (where is X, how many, left/right of Y). Returns annotated full image.
- If either tool returns 0 results or "no region passed the confidence threshold", this does NOT mean the object is absent. The object may be small, partially occluded, or low-contrast. Always inspect the full original image directly before concluding the object is not there.
- Never answer "none of the options" solely because detection found nothing — the question guarantees a valid answer. Fall back to careful visual inspection of the original image.
- If a synonym may help (e.g. 'motorbike' instead of 'motorcycle', 'luggage' instead of 'suitcase'), retry with that synonym before giving up.

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

ALL_TOOLS_ROLE = (
    "You are a spatial intelligence agent with access to a broad toolkit spanning "
    "2D perception, vision-language reasoning, 3D reconstruction, orientation estimation, "
    "and image/video generation.\n\n"
    "Your job is to analyze the user's task, choose the most appropriate tool(s), "
    "and combine their outputs into an accurate final answer."
)

TOOL_SELECTION_GUIDE = """# Tool Selection Guide

Use this guide to pick the right tool before calling it. Only call tools that are listed in <tools>.

## 2D Perception
- **depth_estimation_tool**: Use when you need relative depth, near/far relationships, or occlusion ordering in a single image. Do not use for object labels or segmentation masks.
- **segment_image_tool**: Use when you need precise pixel masks for objects or regions (SAM2). Provide points/boxes when possible.
- **zoom_object_tool**: Detect an object and return cropped close-up image(s) for fine-grained attribute inspection (GroundingDINO). Use when the question is about COLOR, TEXTURE, MATERIAL, PATTERN, TEXT, or any detail that requires magnification. Example: "What color is the helmet?" → zoom into helmet.
- **localize_object_tool**: Detect objects and draw bounding boxes on the full image (GroundingDINO). Use when the question is about WHERE objects are, HOW MANY there are, or their SPATIAL LAYOUT. Returns annotated full image + text position summary. Example: "How many cars are there?" or "Is the dog to the left of the cat?"
- **supervision_tool**: Use for classic YOLO-style detection (`image_det`) or instance segmentation (`image_seg`) with visualization.
- **yoloe_detection_tool**: Use when you need custom class names with YOLO-E detection (bounding boxes only).
- **yolo26_tool**: Use for fast local detection with class labels and confidence scores when no text prompt is needed.
- **qwenvl_detection_tool**: Use for referring or reasoning-based detection via Qwen VL when a language-guided box is needed.

## Vision-Language (VLM)
- **moondream_tool**: Use for lightweight captioning, VQA, pointing, or simple visual reasoning on one image.
- **molmo2_tool**: Use for richer QA, captioning, or point grounding when you need structured language outputs or annotated points.

## 3D & Spatial
- **pi3_tool** / **pi3x_tool**: Use for camera motion, novel viewpoints, and 3D spatial reasoning from images. **Never** call with azimuth=0, elevation=0 (that repeats the input view). Prefer pi3x_tool when available.
- **vggt_tool**: Use for multi-view 3D reconstruction and camera pose estimation from several images or video frames.
- **mapanything_tool**: Use for dense multi-view 3D point clouds via depth + pose fusion.
- **orient_anything_v2_tool**: Use for object orientation (azimuth/elevation/rotation) or relative pose between two views.

## Generation
- **image_generation_sana_tool**: Use to visualize hypothetical scenes or planned outcomes from text. Output is synthetic, not factual evidence.
- **video_generation_veo_tool** / **video_generation_sora_tool** / **video_generation_wan_tool**: Use for cloud API text/image-to-video when motion synthesis is required.
- **video_generation_vace_tool**: Use for local first-frame video generation from one reference image + prompt. One call per turn only; slow and GPU-heavy.

## Selection Rules
1. Prefer perception tools (depth, detection, segmentation, 3D) before generation tools when answering factual questions about provided images.
2. Treat generated images/videos as hypotheses or visualizations, not as direct observations of the original scene.
3. For spatial/viewpoint questions, use 3D tools with **new** angles (not 0°,0°) and consider `camera_view=true` for first-person reasoning.
4. Call only tools present in <tools>; do not invent tool names.
5. You may call multiple tools across iterations, but avoid redundant calls with identical parameters."""

ALL_TOOLS_WORKFLOW = """# All-Tools Workflow

You can perform MULTIPLE rounds of tool calls and analysis.

Workflow:
1. Read the question and inspect available input image(s).
2. Decide whether you can answer directly or need tool evidence.
3. Pick the smallest useful set of tools using the Tool Selection Guide.
4. Call tools with precise parameters; reuse outputs in later reasoning.
5. After each round, decide whether another tool would materially reduce uncertainty.
6. For 3D tools, explore **new** viewpoints (never azimuth=0, elevation=0).
7. For generation tasks, call the generation tool directly with a concise prompt.
8. Only provide your final <answer></answer> when evidence is sufficient.
9. Only put number (like 1,2,3) or Options in <answer></answer> tags, do not put any other text."""

ALL_TOOLS_CONTINUATION_HINT = """1. **Continue investigating** - Call a different tool or the same tool with new parameters if more evidence is needed.

2. **Provide final answer** - If you have gathered sufficient information:

Instructions:
- Think: Which tool from the Tool Selection Guide would most reduce remaining uncertainty?
- If YES: Use <tool_call></tool_call> with appropriate parameters (avoid repeating failed or redundant calls).
- If NO: output your thinking process in <think></think> and your final answer in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text.

Reminder:
- Prefer perception before generation for factual image questions.
- Never call 3D reconstruction tools at (azimuth=0, elevation=0).
- Generated media is synthetic visualization, not direct evidence."""

# ─── Fixed tool-calling block (always appended, never user-modified) ──────────
#
# This block teaches the model the <tool_call> wire format.  It is intentionally
# kept separate from the role/task description so users can swap the latter
# freely without breaking tool invocation.

TOOL_CALLING_BLOCK = (
    "\n# Tools\n"
    "You have access to the following tools:\n"
    "<tools>\n"
    "{tools_json}\n"
    "</tools>\n\n"
    "# How to call a tool\n"
    "When you need to use a tool, return a JSON object with the function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": "<function-name>", "arguments": {"param1": "value1", "param2": "value2"}}\n'
    "</tool_call>\n\n"
    "You can call multiple tools if needed by using multiple <tool_call> blocks.\n"
)


def build_system_prompt(
    role_prompt: str,
    tools_json: str,
    workflow: Optional[str] = None,
) -> str:
    """
    Compose a complete system prompt from three independent layers.

    Args:
        role_prompt: What the agent is / what it should do.  This is the only
                     part users need to customise.  It must NOT include tool
                     schemas or calling instructions — those are added here.
        tools_json:  JSON-serialised list of tool schemas (from
                     ``ToolRegistry.get_function_schemas()``).
        workflow:    Optional multi-step iteration guidance appended at the end
                     (e.g. ``SPATIAL_3D_WORKFLOW``, ``GENERAL_VISION_WORKFLOW``).
                     Pass ``None`` to omit.

    Returns:
        A ready-to-use system prompt string.
    """
    tool_block = TOOL_CALLING_BLOCK.replace("{tools_json}", tools_json)
    parts: List[str] = [role_prompt.rstrip(), tool_block]
    if workflow:
        parts.append(workflow)
    return "\n".join(parts)


# ─── Built-in role prompts ─────────────────────────────────────────────────────
#
# These are *role-only* strings — no tool schemas, no <tool_call> syntax.
# They are the first argument to build_system_prompt().

SPATIAL_3D_ROLE = "You are a helpful assistant that can analyze images and answer questions."

GENERAL_VISION_ROLE = "You are a helpful visual assistant that can analyze images and answer questions."

GENERATION_ROLE = (
    "You are a helpful multimodal generation assistant.\n"
    "If the request is clearly a generation task, prefer calling the tool directly "
    "instead of long analysis."
)

# ─── Full system prompt templates (with {tools_json} placeholder) ─────────────
#
# These templates keep the old {tools_json} interface for backward compatibility.
# New code should use build_system_prompt() + the role/workflow constants above.

SPATIAL_3D_SYSTEM_PROMPT = (
    SPATIAL_3D_ROLE + "\n"
    + TOOL_CALLING_BLOCK
    + "\n" + SPATIAL_3D_WORKFLOW
)

GENERAL_VISION_SYSTEM_PROMPT = (
    GENERAL_VISION_ROLE + "\n"
    + TOOL_CALLING_BLOCK
    + "\n" + GENERAL_VISION_WORKFLOW
)

GENERATION_SYSTEM_PROMPT = (
    GENERATION_ROLE + "\n"
    + TOOL_CALLING_BLOCK
    + "\n" + GENERATION_WORKFLOW
)

ALL_TOOLS_SYSTEM_PROMPT = (
    ALL_TOOLS_ROLE + "\n"
    + TOOL_CALLING_BLOCK
    + "\n" + TOOL_SELECTION_GUIDE
    + "\n" + ALL_TOOLS_WORKFLOW
)


def create_all_tools_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Build the all-tools system prompt with full inline schemas and selection guide.

    Args:
        tools: List of tool function schemas from ToolRegistry.get_function_schemas().

    Returns:
        Complete system prompt string for workflow_mode='all_tools'.
    """
    if not tools:
        return ALL_TOOLS_ROLE + "\n" + TOOL_SELECTION_GUIDE

    tools_json = json.dumps(tools, indent=2)
    return ALL_TOOLS_SYSTEM_PROMPT.replace("{tools_json}", tools_json)


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
        return SPATIAL_3D_ROLE

    tools_json = json.dumps(tools, indent=2)
    chosen_workflow = workflow if workflow is not None else SPATIAL_3D_WORKFLOW
    return build_system_prompt(SPATIAL_3D_ROLE, tools_json, workflow=chosen_workflow)


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
