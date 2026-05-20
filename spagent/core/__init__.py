"""
SPAgent Core Module

This module contains the core components of the SPAgent architecture:
- SPAgent: Main agent class for problem solving
- Tool: Base class for external expert tools
- Model: Base class for VLLM model wrappers
- DataCollector: Training data collection for multimodal models
- AgentMemory: Multimodal typed memory for multi-turn conversations
- StepResult: Structured return value from SPAgent.step()
"""

from .spagent import SPAgent
from .tool import Tool, ToolRegistry
from .model import Model
from .data_collector import DataCollector, InferenceSample, SessionData
from .memory import AgentMemory, MemoryEntry, StepResult

__all__ = [
    'SPAgent',
    'Tool',
    'ToolRegistry',
    'Model',
    'DataCollector',
    'InferenceSample',
    'SessionData',
    'AgentMemory',
    'MemoryEntry',
    'StepResult',
]
