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
            name="my_custom_tool",
            description="Clear description of what this tool does and when to use it."
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize client (mock or real)"""
        if self.use_mock:
            self._client = MockMyService()
        else:
            self._client = RealMyClient(server_url=self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image."
                },
                "option": {
                    "type": "string",
                    "enum": ["mode_a", "mode_b"],
                    "description": "Processing mode.",
                    "default": "mode_a"
                }
            },
            "required": ["image_path"]
        }

    def call(
        self,
        image_path: str,
        option: str = "mode_a"
    ) -> Dict[str, Any]:
        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image not found: {image_path}"}

            result = self._client.process(image_path, option=option)

            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error") if result else "No result"
                }
        except Exception as e:
            logger.error(f"MyCustomTool error: {e}")
            return {"success": False, "error": str(e)}
