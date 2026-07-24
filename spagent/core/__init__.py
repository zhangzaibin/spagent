"""
SPAgent Core Module

This module contains the core components of the SPAgent architecture:
- SPAgent: Main agent class for problem solving
- Tool: Base class for external expert tools
- Model: Base class for VLLM model wrappers
- DataCollector: Training data collection for multimodal models
- AgentMemory: Multimodal typed memory for multi-turn conversations
- StepResult: Structured return value from SPAgent.step()

Prompt utilities (three-layer architecture):
- build_system_prompt: Compose role + tool block + workflow into a system prompt
- TOOL_CALLING_BLOCK: Fixed <tool_call> wire-format block (auto-appended)
- SPATIAL_3D_ROLE / GENERAL_VISION_ROLE / GENERATION_ROLE: Built-in role strings
- SPATIAL_3D_WORKFLOW / GENERAL_VISION_WORKFLOW / GENERATION_WORKFLOW: Workflow presets
"""

from .spagent import SPAgent
from .tool import Tool, ToolRegistry
from .tool_result import (
    ToolResult,
    CategoryContract,
    CATEGORY_CONTRACTS,
    ALL_CATEGORIES,
    validate_payload,
    visualization_paths,
    DetectionPayload,
    SegmentationPayload,
    PointsPayload,
    DepthPayload,
    FlowPayload,
    OrientationPayload,
    PointCloudPayload,
    MediaPayload,
    TextPayload,
)
from .render import render, RenderedOutput, resolve_projection, legacy_projection
from .model import Model
from .data_collector import DataCollector, InferenceSample, SessionData
from .memory import AgentMemory, MemoryEntry, StepResult
from .prompts import (
    build_system_prompt,
    build_general_vision_continuation_hint,
    build_tool_selection_guide,
    TOOL_CALLING_BLOCK,
    SPATIAL_3D_ROLE,
    GENERAL_VISION_ROLE,
    GENERATION_ROLE,
    SPATIAL_3D_WORKFLOW,
    GENERAL_VISION_WORKFLOW,
    GENERATION_WORKFLOW,
    SPATIAL_3D_CONTINUATION_HINT,
    GENERAL_VISION_CONTINUATION_HINT,
    GENERATION_CONTINUATION_HINT,
    ALL_TOOLS_ROLE,
    ALL_TOOLS_WORKFLOW,
    ALL_TOOLS_CONTINUATION_HINT,
    TOOL_SELECTION_GUIDE,
    create_all_tools_system_prompt,
    SPATIAL_2_ROLE,
    SPATIAL_2_WORKFLOW,
    SPATIAL_2_CONTINUATION_HINT,
    create_spatial2_system_prompt,
)

__all__ = [
    'SPAgent',
    'Tool',
    'ToolRegistry',
    # Standardized tool output
    'ToolResult',
    'CategoryContract',
    'CATEGORY_CONTRACTS',
    'ALL_CATEGORIES',
    'validate_payload',
    'visualization_paths',
    'DetectionPayload',
    'SegmentationPayload',
    'PointsPayload',
    'DepthPayload',
    'FlowPayload',
    'OrientationPayload',
    'PointCloudPayload',
    'MediaPayload',
    'TextPayload',
    # Render module
    'render',
    'RenderedOutput',
    'resolve_projection',
    'legacy_projection',
    'Model',
    'DataCollector',
    'InferenceSample',
    'SessionData',
    'AgentMemory',
    'MemoryEntry',
    'StepResult',
    # Prompt utilities
    'build_system_prompt',
    'build_general_vision_continuation_hint',
    'build_tool_selection_guide',
    'TOOL_CALLING_BLOCK',
    'SPATIAL_3D_ROLE',
    'GENERAL_VISION_ROLE',
    'GENERATION_ROLE',
    'SPATIAL_3D_WORKFLOW',
    'GENERAL_VISION_WORKFLOW',
    'GENERATION_WORKFLOW',
    'SPATIAL_3D_CONTINUATION_HINT',
    'GENERAL_VISION_CONTINUATION_HINT',
    'GENERATION_CONTINUATION_HINT',
    'ALL_TOOLS_ROLE',
    'ALL_TOOLS_WORKFLOW',
    'ALL_TOOLS_CONTINUATION_HINT',
    'TOOL_SELECTION_GUIDE',
    'create_all_tools_system_prompt',
    'SPATIAL_2_ROLE',
    'SPATIAL_2_WORKFLOW',
    'SPATIAL_2_CONTINUATION_HINT',
    'create_spatial2_system_prompt',
]
