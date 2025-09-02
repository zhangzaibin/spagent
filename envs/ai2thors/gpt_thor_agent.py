"""
GPT-powered AI2THOR Agent

This module implements an agent that uses GPT to control actions in the AI2THOR environment.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import tempfile
import cv2
import numpy as np

from models.gpt_model import GPTModel
from envs.ai2thors.thor_env import ThorEnvironment

logger = logging.getLogger(__name__)

class GPTThorAgent:
    """Agent that uses GPT to control AI2THOR environment"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        scene: str = "FloorPlan1",
        temperature: float = 0.7
    ):
        """
        Initialize GPT-powered AI2THOR agent
        
        Args:
            model_name: GPT model name
            scene: AI2THOR scene name
            temperature: GPT temperature parameter
        """
        self.gpt = GPTModel(model_name=model_name, temperature=temperature)
        self.env = ThorEnvironment(scene=scene)
        self.reset()
        
    def reset(self, scene: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment and agent state"""
        return self.env.reset(scene=scene)
        
    def _format_observation(self, state: Dict[str, Any]) -> str:
        """
        Format environment state as text for GPT
        
        Args:
            state: Environment state dictionary
            
        Returns:
            Formatted observation text
        """
        obs_parts = []
        
        # Add basic state info
        obs_parts.append("Current environment state:")
        obs_parts.append(f"- Last action {'succeeded' if state['success'] else 'failed'}")
        if state['error']:
            obs_parts.append(f"- Error: {state['error']}")
            
        # Add visible objects
        visible_objects = [
            obj for obj in state['metadata']['objects']
            if obj.get('visible', False)
        ]
        if visible_objects:
            obs_parts.append("\nVisible objects:")
            for obj in visible_objects:
                obj_desc = f"- {obj['objectType']}"
                if 'distance' in obj:
                    obj_desc += f" (distance: {obj['distance']:.2f}m)"
                obs_parts.append(obj_desc)
                
        return "\n".join(obs_parts)
        
    def _save_observation_image(self, frame: np.ndarray) -> str:
        """
        Save observation image to temporary file
        
        Args:
            frame: RGB image array
            
        Returns:
            Path to saved image
        """
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Save to temp file
        temp_dir = Path(tempfile.gettempdir())
        image_path = temp_dir / "thor_observation.jpg"
        cv2.imwrite(str(image_path), frame)
        
        return str(image_path)
        
    def get_action(self, goal: str) -> Dict[str, Any]:
        """
        Get next action from GPT based on current state and goal
        
        Args:
            goal: Text description of goal
            
        Returns:
            Action dictionary with name and parameters
        """
        # Get current state
        state = self.env._get_state()
        
        # Format observation
        obs_text = self._format_observation(state)
        
        # Save observation image
        image_path = self._save_observation_image(state['frame'])
        
        # Construct prompt
        prompt = f"""You are an AI agent in the AI2THOR environment. Your goal is: {goal}

Current observation:
{obs_text}

Based on this observation, what action should you take next? Respond with a JSON object containing:
- action: The action name (e.g. "MoveAhead", "RotateRight", "PickupObject")
- args: Dictionary of action parameters

Example response:
{{"action": "MoveAhead", "args": {{"magnitude": 0.25}}}}"""

        # Get GPT response
        try:
            response = self.gpt.single_image_inference(
                image_path=image_path,
                prompt=prompt
            )
            
            # Parse action from response
            action_dict = json.loads(response)
            return action_dict
            
        except Exception as e:
            logger.error(f"Failed to get action from GPT: {e}")
            # Return no-op action on failure
            return {"action": "Pass", "args": {}}
            
    def step(self, goal: str) -> Dict[str, Any]:
        """
        Take one step toward goal
        
        Args:
            goal: Text description of goal
            
        Returns:
            New environment state
        """
        # Get action from GPT
        action_dict = self.get_action(goal)
        
        # Execute action
        return self.env.step(
            action=action_dict["action"],
            **action_dict.get("args", {})
        )
        
    def close(self):
        """Cleanup resources"""
        self.env.close()


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = GPTThorAgent(
        model_name="gpt-4o-mini",
        scene="FloorPlan1"
    )
    
    # Example goal
    goal = "Find and pick up the apple on the kitchen counter"
    
    try:
        # Take a few steps
        for _ in range(5):
            state = agent.step(goal)
            
            # Check if goal achieved
            if not state["success"]:
                print(f"Action failed: {state['error']}")
                break
                
            # Print visible objects
            visible_objects = [
                obj["objectType"] 
                for obj in state["metadata"]["objects"]
                if obj.get("visible", False)
            ]
            print(f"Visible objects: {visible_objects}")
            
    finally:
        agent.close() 