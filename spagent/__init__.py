"""
SPAgent - Spatial Intelligence Agent

A flexible and modular agent system for spatial intelligence tasks
that combines VLLM models with external expert tools.

New Architecture Features:
- Modular tool system
- Dynamic tool management  
- Multiple model support
- Parallel tool execution
- Multi-image analysis
- Comprehensive error handling

Usage:
    from spagent import SPAgent
    from spagent.models import GPTModel
    from spagent.tools import DepthEstimationTool, SegmentationTool
    
    # Create model and tools
    model = GPTModel(model_name="gpt-4o-mini")
    tools = [
        DepthEstimationTool(use_mock=True),
        SegmentationTool(use_mock=True)
    ]
    
    # Create agent
    agent = SPAgent(model=model, tools=tools)
    
    # Solve problem
    result = agent.solve_problem("image.jpg", "What do you see?")
    print(result['answer'])
"""

# Core components
from .core import SPAgent, Tool, ToolRegistry, Model

# Import submodules for convenience
from . import models
from . import tools
from . import utils

__version__ = '2.0.0'
__all__ = ['SPAgent', 'Tool', 'ToolRegistry', 'Model', 'models', 'tools', 'utils']

# Legacy workflow imports for backwards compatibility (deprecated)
# These will be removed in future versions
try:
    from . import workflows
    import warnings
    warnings.warn(
        "The workflows module is deprecated and will be removed in a future version. "
        "Please migrate to the new SPAgent architecture.",
        DeprecationWarning,
        stacklevel=2
    )
except ImportError:
    # workflows module not available (expected after removal)
    pass 