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

SPATIAL_3D_WORKFLOW = """# Spatial Reasoning Protocol
Use original images for object identity and Pi3X only for missing geometry. camN always
corresponds to image N. Never call Pi3X at (azimuth=0, elevation=0).


# 1. Route the question before reasoning
Choose exactly one solver below and follow its checks. Do not mix page coordinates,
world coordinates, and a camera's ego coordinates.

## A. Orbit views named front/left/back/right
These labels describe CAMERA POSITIONS around a common subject, not camera headings.
If the prompt explicitly calls views front/left/back/right around an object, ALWAYS use
this solver and NEVER use the fixed-heading image1=0/image2=90 rule. Every camera looks
inward toward the subject. Therefore never assign the position label "right view" to an
ego turn right. Determine ego directions while looking inward:
    front-view camera: ego-right leads around the subject toward the left-view camera;
    left-view camera: ego-right leads toward the back-view camera;
    back-view camera: ego-right leads toward the right-view camera;
    right-view camera: ego-right leads toward the front-view camera.
Ego-left follows the reverse cycle. Use the reconstructed camera centers to verify this
inward-looking orbit convention. For turn-and-move questions, move along that ego
direction and compare distance to the target; do not infer direction solely from the
name of the view containing the target.
Perspective sanity check: while an observer at the front-view camera looks inward,
their ego-right points toward the subject's LEFT side, not toward the camera position
called the right view. If your reasoning maps front-view + ego-right to right-view,
discard it and recompute in the observer's frame.

## B. Exact heading composition
Use this whenever the question states exact turns. Explicitly compute:
    query_heading = (start_heading + body_turn + queried_direction) mod 360
Use clockwise-positive headings:
    image1=0; each stated 90-degree right view adds 90; opposite image=180.
Use offsets:
    right=+90, left=-90, behind=+180, front=0.
The body turn and queried direction are TWO separate operations; never collapse them.
In <think>, fill all five worksheet fields before interpreting any object:
    start=__; body_turn=__; queried_relation=__; sum=__; matching_view=__.
If the question says "turn left, then what is left", -90 MUST appear twice. If a field
is missing, the calculation is invalid. If the text already
defines the view sequence, trust this arithmetic over noisy reconstructed camera poses.
After finding the query heading, use all original images and geometry to identify which
option lies on that camera's OPTICAL AXIS. Follow the ray to the far surface/object near
the image center. Never choose a large close foreground object intruding from an image
edge merely because it is visually dominant.
The computed query_heading ALREADY includes "what is left/right/behind." At the matching
view, identify what lies along its forward central ray; do NOT apply left/right/behind
a second time within that matching image. Once the sum is computed, do not reinterpret
or reverse it in prose.

## C. Camera translation from view 1 to view 2
Use cam1's ego frame. Let C1 and C2 be camera centers and let O be the common subject or
scene center. Approximate cam1-forward by O-C1, then assess displacement C2-C1:
    dot(C2-C1, cam1-forward) gives forward/backward;
    the signed lateral component gives left/right.
Reason from the frustum apex, forward axis, and scene center—not raw left/right position
on the rendered page. If the lateral sign is not unmistakable, request a second global
elevated view from a substantially different azimuth and verify the same ego-frame sign.
Changing render azimuth only rotates the display and is not independent evidence unless
you explicitly track cam1's forward/right axes. Cross-check with motion parallax in the
original pair: camera motion right makes a common foreground subject shift left relative
to the distant background; camera motion left makes it shift right. Do not answer
confidently from ambiguous page coordinates.

## D. Turn and move closer/farther
Form the post-turn movement vector in the starting camera's ego frame. Compare distance
to the target before and after a small conceptual step, or equivalently check whether
the movement has a positive projection toward the target. Do not infer target direction
merely from which named view contains it.

## E. Viewpoint-relative object relation
Use the named camera as origin. For left/right, compare lateral coordinates in that
camera frame. For behind/in-front, compare depth from that camera: target B is behind A
when B is farther along the viewing direction. Depth ordering is primary; do not require
perfect occlusion, exact collinearity, or visibility in the queried image. If B is absent
in that view but appears in a side view at greater depth, it may be occluded by A—this
supports rather than refutes "behind." For qualitative multiple-choice relations,
lateral offset does not cancel a clear positive depth difference. Reject "behind" only
when B has no meaningful greater depth.
Operational rule: never require or argue about being "directly behind." Compute only
whether depth(B) > depth(A) in the named camera frame.

# 2. Pi3X view policy
- Layout, camera motion, or depth: use camera_view=false, elevation=60, and set
  rotation_reference_camera to the named viewpoint (cam1 for view1->view2 motion).
- Ego-view confirmation: use camera_view=true with the named camera and a nonzero
  azimuth/elevation selected to test a specific uncertainty.
- The rendered page axes are display axes, not semantic left/right. A second rendering
  may rotate or mirror page appearance while leaving the 3D relation unchanged.

# 3. Final verification
Check the chosen option against the solver's computed direction/depth and the original
images. If prose intuition conflicts with the numeric heading calculation, redo the
calculation. Put only the option letter/number in <answer></answer>."""

# ─── Continuation hints (injected into the multi-step iteration prompt) ──────

SPATIAL_3D_CONTINUATION_HINT = """Before another call, state the unresolved geometric
quantity: heading, lateral displacement sign, target projection, or depth ordering.

- Exact stated rotations: stop using tools and show the modular heading sum.
- Named orbit views: remember labels are camera positions looking inward; ego-right
  follows front-position -> left-position -> back-position -> right-position.
- Ambiguous view1->view2 lateral sign: request a second GLOBAL view with
  camera_view=false, rotation_reference_camera=1, and a substantially different
  nonzero angle. Reproject C2-C1 into cam1's frame; never use page-left/page-right.
- Object depth: use camera_view=false with the named reference camera. Greater depth is
  sufficient for "behind" even with lateral offset; invisibility may be occlusion.
- After a modular heading sum, do not apply the queried relation again in the matched
  image; inspect the forward central ray and far scene boundary.
- Never use (0,0) and never repeat an equivalent projection.

Then verify the selected option against original-image object identity. Output analysis
in <think></think> and only the option letter/number in <answer></answer>."""

# ─── Spatial-2 workflow (second spatial preset, does not touch SPATIAL_3D_WORKFLOW) ──
#
# Deliberately generic and short: general reasoning principles that apply to any
# spatial/positional/motion/attribute question, instead of per-task-type branching.

SPATIAL_2_ROLE = (
    "You are a spatial-reasoning assistant for questions about pairs (or short "
    "sequences) of photos of the same scene, covering camera position/motion, "
    "object/region layout, or object attributes."
)

SPATIAL_2_WORKFLOW = """# Spatial Reasoning Protocol (v2)
Treat the original photos as primary evidence for object identity, layout, motion, and
attributes. Pi3X 3D reconstruction is a secondary cross-check for geometry — it never
overrides what is plainly visible in the photos, and its raw XYZ axes/colors carry NO
compass or absolute-direction meaning by themselves. camN corresponds to image N.
Never call Pi3X at (azimuth=0, elevation=0).

# General reasoning principles
1. State the reference frame explicitly before answering: who/what is the observer or
   anchor, which way it faces (or what absolute direction is given), and which frame
   the final answer must be expressed in. Do not mix frames mid-answer.
2. Compute the answer as an explicit offset from a known anchor or observation (an
   angle, a visible position, a stated fact) rather than guessing a shortcut like "it
   must be the opposite side" or "it must be the far corner" — always point to the
   specific evidence that gives you the offset.
3. If the observer/camera turns or moves partway through the question, update the
   reference frame for that step first, then recompute the full answer in the new
   frame — do not reuse a direction/sign computed before the move.
4. For "which way is it turning/moving" questions, compare at least two independent
   shared reference points across the photos (not a single centered object) before
   deciding a direction, and consider both possible axes of change rather than
   assuming one by default.
5. For counting or size/measurement comparisons, prefer the original photos and a
   shared visual reference over Pi3X's point cloud, which has no guaranteed
   consistent scale across two different photos.
6. Before finalizing, re-derive the answer a second, independent way (re-reading the
   photos, or a different Pi3X view) and check it agrees; if a computed answer
   conflicts with a gut-feeling shortcut, trust the computation.

# Pi3X usage
Call Pi3X only when the photos alone leave the geometry ambiguous. A top-down render
(elevation 45-90, camera_view=false) is useful for layout questions; camera_view=true
on a named camera shows what that camera can/cannot see. A render from a different
azimuth may mirror the page layout without changing the real 3D relation — always
reproject back into the observer's/anchor's own frame rather than reading
page-left/page-right directly.

# Final answer
Confirm your answer is grounded in a specific stated observation, not intuition. Put
only the option letter/number in <answer></answer>."""

SPATIAL_2_CONTINUATION_HINT = """Before another tool call, restate the reference frame
(observer + facing, or the given anchor/direction) and which specific observation is
still missing.

- Never read compass or absolute-direction meaning off Pi3X's raw axes/colors — ground
  every direction in an explicit angle or position read from one consistent view.
- After any turn/move described in the question, recompute the full answer in the new
  frame instead of reusing a pre-move direction or sign.
- For turning/moving-direction questions, compare at least two independent shared
  reference points and check both possible axes of change before deciding.
- For counting/size questions, re-examine the original photos with a shared scale
  reference rather than trusting Pi3X point-cloud density/height alone.
- Reject "opposite direction" / "far corner" style shortcuts — always compute an
  explicit offset from a stated anchor or observation.
- Never call Pi3X at (0,0) and never repeat an equivalent view you already have.

Then verify your answer against the original photos one more time. Output analysis in
<think></think> and only the option letter/number in <answer></answer>."""


# ─── Per-tool guidance entries for the general-vision continuation hint ───────
#
# Each entry is keyed by the tool's runtime name.  build_general_vision_continuation_hint()
# filters this dict to only include tools that are actually registered.

_GV_TOOL_ENTRIES: Dict[str, str] = {
    "zoom_object_tool": (
        "zoom_object_tool — for inspecting a specific named object's attribute:\n"
        "  • Exact color, texture, material, text/label, pattern of a PARTICULAR object\n"
        '  • text_prompt = specific object name (e.g. "red car", "tissue box", "woman with backpack")'
    ),
    "localize_object_tool": (
        "localize_object_tool — for finding or counting objects in ONE image:\n"
        '  • "How many X are there?", "Is X to the left/right of Y?"\n'
        "  • text_prompt = object category to detect"
    ),
    "pi3x_tool": (
        "pi3x_tool — for 3D spatial questions with multiple-viewpoint images:\n"
        '  • "From camera N / image N viewpoint, what is to the left/right/behind/in-front?"\n'
        '  • "Which direction did the camera move between views?"\n'
        "  • ⚠ NEVER call pi3x with azimuth=0, elevation=0 — it repeats the input view and gives no new info!\n"
        "  • Best angles:\n"
        "      - Top-down layout view: elevation=45, azimuth=0   ← best for left/right/front/behind reasoning\n"
        "      - Left side:   azimuth=-90, elevation=0\n"
        "      - Right side:  azimuth=90, elevation=0\n"
        "      - Back view:   azimuth=180, elevation=0\n"
        "  • For \"from camera X / image X viewpoint\" questions:\n"
        "    rotation_reference_camera=X, camera_view=true, elevation=30~45"
    ),
    "pi3_tool": (
        "pi3_tool — for 3D spatial questions with multiple-viewpoint images:\n"
        '  • "From camera N / image N viewpoint, what is to the left/right/behind/in-front?"\n'
        "  • ⚠ NEVER call pi3 with azimuth=0, elevation=0 — it repeats the input view!\n"
        "  • Best angles: elevation=45 (top-down), azimuth=±90 (sides), azimuth=180 (back)"
    ),
}

_GV_DETECTION_FALLBACK = (
    "Detection fallback:\n"
    "  • If zoom/localize returns 0 results: object may be small or occluded — inspect original image\n"
    "  • Never conclude absent solely from detection failure\n"
    "  • Try synonyms ('motorbike'→'motorcycle', 'sofa'→'couch') before giving up"
)

_GV_PREAMBLE = """Decide your next action:

**Option A — Provide final answer now** if you have enough information:
Output <think>...</think> then <answer>option letter or text only</answer>.

**Option B — Call another tool** if it will materially reduce uncertainty:
Use <tool_call>...</tool_call> with appropriate parameters."""


def build_general_vision_continuation_hint(tool_names: Optional[set] = None) -> str:
    """
    Build a continuation hint that only mentions tools present in *tool_names*.

    Args:
        tool_names: Set of runtime tool name strings (e.g. ``{"pi3x_tool"}``).
                    Pass ``None`` or an empty set to get the preamble-only hint
                    (no tool guide section).  Pass the full set of all known
                    tools to reproduce the original static constant.

    Returns:
        A continuation-hint string ready to be injected into the multi-step prompt.
    """
    if not tool_names:
        return _GV_PREAMBLE

    entries = [text for name, text in _GV_TOOL_ENTRIES.items() if name in tool_names]
    if not entries:
        return _GV_PREAMBLE

    detection_tools = {"zoom_object_tool", "localize_object_tool"}
    include_fallback = bool(detection_tools & tool_names)

    guide_lines = [
        "",
        "────────────────────────────────────────────────",
        "TOOL SELECTION GUIDE",
        "────────────────────────────────────────────────",
    ]
    guide_lines.extend(entries)
    if include_fallback:
        guide_lines.append("")
        guide_lines.append(_GV_DETECTION_FALLBACK)

    return _GV_PREAMBLE + "\n" + "\n".join(guide_lines)


# Backward-compatible constant — includes all known tools.
GENERAL_VISION_CONTINUATION_HINT = build_general_vision_continuation_hint(
    set(_GV_TOOL_ENTRIES.keys())
)


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

# ─── Per-tool entries for the all-tools selection guide ──────────────────────
#
# Organised by category.  build_tool_selection_guide() filters entries to only
# include tools that are actually registered, and omits empty category headers.

_TSG_CATEGORIES: List[Dict[str, Any]] = [
    {
        "header": "## 2D Perception",
        "tools": {
            "depth_estimation_tool": (
                "- **depth_estimation_tool**: Use when you need relative depth, near/far "
                "relationships, or occlusion ordering in a single image. Do not use for "
                "object labels or segmentation masks."
            ),
            "segment_image_tool": (
                "- **segment_image_tool**: Use when you need precise pixel masks for objects "
                "or regions (SAM2). Provide points/boxes when possible."
            ),
            "zoom_object_tool": (
                "- **zoom_object_tool**: Detect an object and return cropped close-up image(s) "
                "for fine-grained attribute inspection (GroundingDINO). Use when the question "
                "is about COLOR, TEXTURE, MATERIAL, PATTERN, TEXT, or any detail that requires "
                'magnification. Example: "What color is the helmet?" → zoom into helmet.'
            ),
            "localize_object_tool": (
                "- **localize_object_tool**: Detect objects and draw bounding boxes on the full "
                "image (GroundingDINO). Use when the question is about WHERE objects are, HOW "
                "MANY there are, or their SPATIAL LAYOUT. Returns annotated full image + text "
                'position summary. Example: "How many cars are there?" or "Is the dog to the '
                'left of the cat?"'
            ),
            "supervision_tool": (
                "- **supervision_tool**: Use for classic YOLO-style detection (`image_det`) or "
                "instance segmentation (`image_seg`) with visualization."
            ),
            "yoloe_detection_tool": (
                "- **yoloe_detection_tool**: Use when you need custom class names with YOLO-E "
                "detection (bounding boxes only)."
            ),
            "yolo26_tool": (
                "- **yolo26_tool**: Use for fast local detection with class labels and confidence "
                "scores when no text prompt is needed."
            ),
            "qwenvl_detection_tool": (
                "- **qwenvl_detection_tool**: Use for referring or reasoning-based detection via "
                "Qwen VL when a language-guided box is needed."
            ),
        },
    },
    {
        "header": "## Vision-Language (VLM)",
        "tools": {
            "moondream_tool": (
                "- **moondream_tool**: Use for lightweight captioning, VQA, pointing, or simple "
                "visual reasoning on one image."
            ),
            "molmo2_tool": (
                "- **molmo2_tool**: Point-grounding tool — locates a specific object or region "
                "and returns an annotated image showing its exact position. Use whenever you "
                "need to visually locate something before answering. Always pass a short "
                'reasoning sentence as `prompt`, e.g. "Point to the object the robot should '
                'grasp next." or "Point to the item that is out of place." Never pass a bare '
                "object name."
            ),
        },
    },
    {
        "header": "## 3D & Spatial",
        "tools": {
            "pi3_tool": (
                "- **pi3_tool** / **pi3x_tool**: Use for camera motion, novel viewpoints, and "
                "3D spatial reasoning from images. **Never** call with azimuth=0, elevation=0 "
                "(that repeats the input view). Prefer pi3x_tool when available."
            ),
            "pi3x_tool": (
                "- **pi3x_tool**: Use for camera motion, novel viewpoints, and 3D spatial "
                "reasoning from images. **Never** call with azimuth=0, elevation=0 (that "
                "repeats the input view)."
            ),
            "vggt_tool": (
                "- **vggt_tool**: Use for multi-view 3D reconstruction and camera pose "
                "estimation from several images or video frames."
            ),
            "mapanything_tool": (
                "- **mapanything_tool**: Use for dense multi-view 3D point clouds via depth + "
                "pose fusion."
            ),
            "orient_anything_v2_tool": (
                "- **orient_anything_v2_tool**: Use for object orientation "
                "(azimuth/elevation/rotation) or relative pose between two views."
            ),
        },
    },
    {
        "header": "## Generation",
        "tools": {
            "image_generation_sana_tool": (
                "- **image_generation_sana_tool**: Use to visualize hypothetical scenes or "
                "planned outcomes from text. Output is synthetic, not factual evidence."
            ),
            "video_generation_veo_tool": (
                "- **video_generation_veo_tool** / **video_generation_sora_tool** / "
                "**video_generation_wan_tool**: Use for cloud API text/image-to-video when "
                "motion synthesis is required."
            ),
            "video_generation_sora_tool": (
                "- **video_generation_sora_tool**: Use for cloud API text/image-to-video when "
                "motion synthesis is required."
            ),
            "video_generation_wan_tool": (
                "- **video_generation_wan_tool**: Use for cloud API text/image-to-video when "
                "motion synthesis is required."
            ),
            "video_generation_vace_tool": (
                "- **video_generation_vace_tool**: Use for local first-frame video generation "
                "from one reference image + prompt. One call per turn only; slow and GPU-heavy."
            ),
        },
    },
]

_TSG_SELECTION_RULES = """## Selection Rules
1. Prefer perception tools (depth, detection, segmentation, 3D) before generation tools when answering factual questions about provided images.
2. Treat generated images/videos as hypotheses or visualizations, not as direct observations of the original scene.
3. For spatial/viewpoint questions, use 3D tools with **new** angles (not 0°,0°) and consider `camera_view=true` for first-person reasoning.
4. Call only tools present in <tools>; do not invent tool names.
5. You may call multiple tools across iterations, but avoid redundant calls with identical parameters."""


def build_tool_selection_guide(tool_names: Optional[set] = None) -> str:
    """
    Build the all-tools selection guide filtered to *tool_names*.

    Args:
        tool_names: Set of runtime tool name strings.  Only categories that
                    contain at least one registered tool are included.  Pass
                    ``None`` to include every known tool (reproduces the
                    original static constant).

    Returns:
        A "# Tool Selection Guide" string ready to be embedded in the system prompt.
    """
    parts = [
        "# Tool Selection Guide\n",
        "Use this guide to pick the right tool before calling it. "
        "Only call tools that are listed in <tools>.\n",
    ]

    for category in _TSG_CATEGORIES:
        if tool_names is None:
            matching = list(category["tools"].values())
        else:
            matching = [
                text
                for name, text in category["tools"].items()
                if name in tool_names
            ]
        if not matching:
            continue
        parts.append(category["header"])
        parts.extend(matching)

    parts.append("")
    parts.append(_TSG_SELECTION_RULES)
    return "\n".join(parts)


# Backward-compatible constant — includes all known tools.
TOOL_SELECTION_GUIDE = build_tool_selection_guide(tool_names=None)

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

    Only the tools that are actually registered appear in both the JSON ``<tools>``
    block and the prose ``Tool Selection Guide``.

    Args:
        tools: List of tool function schemas from ToolRegistry.get_function_schemas().

    Returns:
        Complete system prompt string for workflow_mode='all_tools'.
    """
    registered_names: Optional[set] = (
        {s["function"]["name"] for s in tools} if tools else None
    )
    selection_guide = build_tool_selection_guide(registered_names)

    if not tools:
        return ALL_TOOLS_ROLE + "\n" + selection_guide

    tools_json = json.dumps(tools, indent=2)
    tool_block = TOOL_CALLING_BLOCK.replace("{tools_json}", tools_json)
    return "\n".join([ALL_TOOLS_ROLE, tool_block, selection_guide, ALL_TOOLS_WORKFLOW])


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


def create_spatial2_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Create the "spatial2" system prompt (SPATIAL_2_ROLE + tools + SPATIAL_2_WORKFLOW).

    This is a standalone counterpart to ``create_system_prompt()`` (which builds the
    MindCube-tuned SPATIAL_3D_WORKFLOW) — use it for the ego-quadrant / compass-bearing
    / motion-direction / counting question structure instead of the ``spatial`` prompt,
    so the original ``spatial`` prompt's named-orbit-view solvers stay untouched.

    Args:
        tools: List of tool function schemas

    Returns:
        System prompt string
    """
    if not tools:
        return SPATIAL_2_ROLE

    tools_json = json.dumps(tools, indent=2)
    return build_system_prompt(SPATIAL_2_ROLE, tools_json, workflow=SPATIAL_2_WORKFLOW)


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
