"""
SoFar Tool

This module contains the SoFarTool that wraps SoFar (Spatial Orientation
from Affordance Reasoning) functionality for the SPAgent system.

SoFar (NeurIPS 2025 Spotlight) bridges spatial reasoning and robot manipulation
through language-grounded orientation understanding.

Paper: https://github.com/qizekun/SoFar
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class SoFarTool(Tool):
    """Tool for language-grounded 6-DoF pose estimation using SoFar."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://localhost:20036",
    ):
        super().__init__(
            name="sofar_tool",
            description=(
                "Estimates 6-DoF object pose and grasp affordance for robot manipulation "
                "using SoFar (NeurIPS 2025 Spotlight). Input: an RGB image and a natural "
                "language instruction (e.g. 'pick up the mug by the handle'). "
                "Output: bounding box of the target object, its 6-DoF pose relative to the "
                "camera, recommended approach vector for gripper, and a human-readable "
                "spatial description. Ideal for embodied AI and robotic manipulation tasks."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the client (mock or real)."""
        if self.use_mock:
            from external_experts.SoFar.mock_sofar_service import MockSoFarService
            self._client = MockSoFarService()
            logger.info("SoFarTool initialized with mock service")
        else:
            from external_experts.SoFar.sofar_client import SoFarClient
            self._client = SoFarClient(server_url=self.server_url)
            logger.info(f"SoFarTool initialized with server at {self.server_url}")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the RGB scene image.",
                },
                "instruction": {
                    "type": "string",
                    "description": (
                        "Natural language manipulation instruction specifying the target "
                        "object and desired action. "
                        "Examples: 'pick up the blue mug by the handle', "
                        "'grasp the bottle from the top', 'open the drawer on the left'."
                    ),
                },
                "camera_intrinsics": {
                    "type": "object",
                    "description": (
                        "Optional camera intrinsics for metric-scale output. "
                        "Keys: fx, fy, cx, cy (pixels). If omitted, relative pose is returned."
                    ),
                    "properties": {
                        "fx": {"type": "number"},
                        "fy": {"type": "number"},
                        "cx": {"type": "number"},
                        "cy": {"type": "number"},
                    },
                },
            },
            "required": ["image_path", "instruction"],
        }

    def call(
        self,
        image_path: str,
        instruction: str,
        camera_intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Execute SoFar pose estimation.

        Args:
            image_path: Path to the RGB scene image.
            instruction: Natural language manipulation instruction.
            camera_intrinsics: Optional dict with fx, fy, cx, cy.

        Returns:
            Dict with 'success', 'result', and optionally 'error' keys.
        """
        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image not found: {image_path}"}

            logger.info(f"Running SoFar: instruction='{instruction}'")

            result = self._client.infer(
                image_path=image_path,
                instruction=instruction,
                camera_intrinsics=camera_intrinsics,
            )

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"SoFarTool error: {e}")
            return {"success": False, "error": str(e)}
