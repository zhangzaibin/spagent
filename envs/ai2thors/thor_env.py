"""
AI2THOR Environment Wrapper

This module provides a wrapper around AI2THOR environment for use with SPAgent.
It handles environment initialization, action execution, and state management.
"""

import logging
from typing import Dict, Any, Optional, List
from ai2thor.controller import Controller

logger = logging.getLogger(__name__)

class ThorEnvironment:
    """Wrapper for AI2THOR environment"""
    
    def __init__(
        self,
        scene: str = "FloorPlan1",
        agent_mode: str = "arm",
        width: int = 300,
        height: int = 300,
        visibility_distance: float = 1.5,
        grid_size: float = 0.25,
        render_depth: bool = False,
        render_instance_seg: bool = False,
        field_of_view: int = 60
    ):
        """
        Initialize AI2THOR environment
        
        Args:
            scene: Scene name (e.g. "FloorPlan1") 
            agent_mode: Agent mode ("arm" for manipulation tasks)
            width: Frame width
            height: Frame height
            visibility_distance: Maximum visibility distance
            grid_size: Grid size for movement
            render_depth: Whether to render depth frames
            render_instance_seg: Whether to render instance segmentation
            field_of_view: Camera field of view in degrees
        """
        self.controller = Controller(
            agentMode=agent_mode,
            scene=scene,
            width=width,
            height=height,
            visibilityDistance=visibility_distance,
            gridSize=grid_size,
            renderDepthImage=render_depth,
            renderInstanceSegmentation=render_instance_seg,
            fieldOfView=field_of_view
        )
        self.last_event = None
        logger.info(f"Initialized ThorEnvironment with scene: {scene}")

    def reset(self, scene: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Reset environment to initial state
        
        Args:
            scene: New scene to load (optional)
            **kwargs: Additional reset parameters
            
        Returns:
            Initial environment state
        """
        reset_args = kwargs
        if scene:
            reset_args["scene"] = scene
            
        self.last_event = self.controller.reset(**reset_args)
        return self._get_state()

    def step(self, action: str, **action_args) -> Dict[str, Any]:
        """
        Execute action in environment
        
        Args:
            action: Action name (e.g. "MoveAhead", "RotateRight")
            **action_args: Additional action parameters
            
        Returns:
            New environment state
        """
        self.last_event = self.controller.step(
            action=action,
            **action_args
        )
        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        """
        Get current environment state
        
        Returns:
            Dictionary containing environment state information
        """
        if not self.last_event:
            return {}
            
        return {
            "frame": self.last_event.frame,
            "metadata": self.last_event.metadata,
            "success": self.last_event.metadata.get("lastActionSuccess", False),
            "error": self.last_event.metadata.get("errorMessage", "")
        }

    def get_reachable_positions(self) -> List[Dict[str, float]]:
        """Get all reachable positions in current scene"""
        return self.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]

    def get_objects(self) -> List[Dict[str, Any]]:
        """Get all objects in current scene"""
        return self.controller.last_event.metadata["objects"]

    def close(self):
        """Close environment and free resources"""
        if self.controller:
            self.controller.stop() 