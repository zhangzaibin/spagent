"""
Robotracer

Predict 3D spatial trace waypoints for robotic manipulation; 
Use when: planning a robot arm movement path from A to B in a scene.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class MyCustomTool(Tool):
    """Tool for [Ref_3dkeypoints]"""

    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:8000"):
        super().__init__(
            name="robotracerl",
            description="Predict 3D spatial trace waypoints for robotic manipulation; use when: planning a robot arm movement path from A to B in a scene."
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the robotracer client (mock or real)"""
        if self.use_mock:
             try:
                from external_experts.robotracer.mock_rt_service import MockRoboTracer
                self._client = MockRoboTracer()
                logger.info("Using mock RoboTracer service")
            except ImportError:

                class SimpleMockRoboTracer:
                    def trace(self, image_path, instruction, depth_path=""):
                        return {
                            "success": True,
                            "waypoints": [
                                {"x": 0.12, "y": 0.34, "z": 0.56},
                                {"x": 0.23, "y": 0.45, "z": 0.43},
                                {"x": 0.67, "y": 0.29, "z": 0.11}
                            ],
                            "num_steps": 3,
                            "output_path": f"outputs/trace_{Path(image_path).stem}.json"
                        }

                self._client = SimpleMockRoboTracer()
                logger.info("Using simple mock RoboTracer service")
        else:
            try:
                from external_experts.robotracer.rt_client import RoboTracerClient
                self._client = RoboTracerClient(server_url=self.server_url)
                logger.info(f"Using real RoboTracer service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import RoboTracer client: {e}")
                raise

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the RGB image of the scene."
                },
                "instruction": {
                    "type": "string",
                    "description": "Text instruction for the manipulation task, e.g. 'pick up the red cup and place it on the shelf'."
                },
                "depth_path": {
                    "type": "string",
                    "description": "Optional path to a depth map for improved metric accuracy.",
                    "default": ""
                }
            },
            "required": ["image_path", "instruction"]
        }

    def call(
        self,
        image_path: str,
        instruction: str,
        depth_path: str = ""
    ) -> Dict[str, Any]:
        """
        Generate 3D spatial trace waypoints

        Args:
            image_path: Path to the RGB image
            instruction: Natural language manipulation instruction
            depth_path: Optional path to depth map

        Returns:
            Dictionary with waypoints or error
        """
        try:
            logger.info(f"Running RoboTracer trace on: {image_path}")
            logger.info(f"Instruction: {instruction}")

            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            if not instruction.strip():
                return {
                    "success": False,
                    "error": "Instruction cannot be empty"
                }

            if depth_path and not Path(depth_path).exists():
                return {
                    "success": False,
                    "error": f"Depth map not found: {depth_path}"
                }

            result = self._client.trace(image_path, instruction, depth_path)

            if result and result.get("success"):
                logger.info(
                    f"RoboTracer completed: {result.get('num_steps', 0)} waypoints generated"
                )
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path")
                }
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"RoboTracer failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"RoboTracer trace failed: {error_msg}"
                }

        except Exception as e:
            logger.error(f"RoboTracer tool error: {e}")
            return {"success": False, "error": str(e)}
