"""
Tool Base Class and Registry

This module defines the base Tool interface and ToolRegistry for managing
external expert tools in the SPAgent system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for external expert tools"""
    
    def __init__(self, name: str, description: str):
        """
        Initialize a tool
        
        Args:
            name: Tool identifier name
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        Get the tool's parameter schema in OpenAI function format
        
        Returns:
            Parameter schema dictionary
        """
        pass
    
    @abstractmethod
    def call(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution result dictionary with at least:
            - success: bool
            - result: Any (tool-specific result)
            - error: str (if success=False)
        """
        pass
    
    # ── Skill integration ────────────────────────────────────────────────

    def to_skill(self):
        """Build a Skill for this tool.

        Resolution order:
        1. Look for a matching SKILL.md file in ``spagent/skills/`` (by tool_name).
        2. Fall back to auto-generated metadata from code attributes.
        """
        from .skill import Skill, load_skills_from_directory

        file_skills = load_skills_from_directory()
        for skill in file_skills.values():
            if skill.tool_name == self.name:
                logger.info(f"Loaded skill from file for tool {self.name}: {skill.source_path}")
                return skill

        logger.debug(f"No SKILL.md found for tool {self.name}, using auto-generated fallback")
        return self._build_fallback_skill()

    def _build_fallback_skill(self):
        """Auto-generate a Skill from the tool's code-level metadata."""
        from .skill import Skill

        title = self.name.replace("_tool", "").replace("_", " ").title()

        params = self.parameters
        props = params.get("properties", {})
        required = params.get("required", [])
        param_lines = []
        for pname, pschema in props.items():
            req_tag = " (required)" if pname in required else " (optional)"
            desc = pschema.get("description", "")
            ptype = pschema.get("type", "any")
            param_lines.append(f"- {pname} ({ptype}{req_tag}): {desc}")
        params_block = "\n".join(param_lines) if param_lines else "  (none)"
        example_args = {p: f"<{p}>" for p in props}

        usage_prompt = (
            f"## {title}\n\n"
            f"{self.description}\n\n"
            f"### Parameters\n{params_block}\n\n"
            f"### Call Format\n"
            f"<tool_call>\n"
            f'{json.dumps({"name": self.name, "arguments": example_args}, indent=2)}\n'
            f"</tool_call>"
        )

        return Skill(
            name=self.name.replace("_tool", ""),
            title=title,
            summary=self.description,
            usage_prompt=usage_prompt,
            tool_name=self.name,
        )

    # ── Legacy schema export ──────────────────────────────────────────

    def to_function_schema(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI function calling format
        
        Returns:
            Function schema dictionary
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """
        Register a tool
        
        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, tool_name: str):
        """
        Unregister a tool
        
        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
        else:
            logger.warning(f"Tool not found for unregistration: {tool_name}")
    
    def get(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name
        
        Args:
            tool_name: Name of tool to retrieve
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        Get list of registered tool names
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_all_tools(self) -> Dict[str, Tool]:
        """
        Get all registered tools
        
        Returns:
            Dictionary of tool_name -> Tool
        """
        return self._tools.copy()
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function schemas for all registered tools
        
        Returns:
            List of function schema dictionaries
        """
        return [tool.to_function_schema() for tool in self._tools.values()] 