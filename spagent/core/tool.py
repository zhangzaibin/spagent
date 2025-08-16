"""
Tool Base Class and Registry

This module defines the base Tool interface and ToolRegistry for managing
external expert tools in the SPAgent system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
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