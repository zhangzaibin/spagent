"""
Orient Anything V2 Tool

This module contains the OrientAnythingV2Tool that wraps
Orient Anything V2 functionality for the SPAgent system.

Orient Anything V2 (NeurIPS 2025 Spotlight) is a unified spatial vision model
for object orientation estimation, symmetry detection, and relative rotation.

Paper: https://orient-anythingv2.github.io/
Repo:  https://github.com/SpatialVision/Orient-Anything-V2
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class OrientAnythingV2Tool(Tool):
    """Tool for object orientation estimation using Orient Anything V2."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://localhost:20034",
    ):
        super().__init__(
            name="orient_anything_v2_tool",
            description=(
                "Estimates the 3D orientation of objects in images using Orient Anything V2 "
                "Given a single image, returns azimuth (0-360°), elevation (-90~90°), "
                "in-plane rotation (-180~180°), and symmetry_alpha (0/1/2/4 indicating "
                "rotational symmetry order). "
                "Given two images of the same object from different viewpoints, also returns "
                "rel_azimuth, rel_elevation, rel_rotation — the relative pose of the second "
                "image with respect to the first. "
                "Useful for robotic grasping, AR/VR scene understanding, and spatial reasoning."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the client (mock or real)."""
        if self.use_mock:
            from external_experts.OrientAnythingV2.mock_oa_v2_service import MockOrientAnythingV2Service
            self._client = MockOrientAnythingV2Service()
            logger.info("OrientAnythingV2Tool initialized with mock service")
        else:
            from external_experts.OrientAnythingV2.oa_v2_client import OrientAnythingV2Client
            self._client = OrientAnythingV2Client(server_url=self.server_url)
            logger.info(f"OrientAnythingV2Tool initialized with server at {self.server_url}")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the input image.",
                },
                "object_category": {
                    "type": "string",
                    "description": (
                        "The semantic category of the object to estimate orientation for "
                        "(e.g. 'chair', 'car', 'bottle', 'laptop'). Providing an accurate "
                        "category significantly improves estimation quality."
                    ),
                },
                "task": {
                    "type": "string",
                    "enum": ["orientation", "symmetry", "relative_rotation"],
                    "description": (
                        "'orientation': return azimuth/elevation/rotation of the object "
                        "and its symmetry_alpha. "
                        "'symmetry': same as orientation — symmetry_alpha is always returned. "
                        "'relative_rotation': estimate relative pose between two views "
                        "(requires image_path2); also returns the absolute pose of the first."
                    ),
                    "default": "orientation",
                },
                "image_path2": {
                    "type": "string",
                    "description": (
                        "Path to the second image. Required only when task='relative_rotation'."
                    ),
                },
            },
            "required": ["image_path", "object_category"],
        }

    def call(
        self,
        image_path: str,
        object_category: str,
        task: str = "orientation",
        image_path2: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Orient Anything V2 inference.

        Args:
            image_path: Path to the input image.
            object_category: Semantic category of the target object.
            task: One of 'orientation', 'symmetry', 'relative_rotation'.
            image_path2: Path to second image (required for relative_rotation).

        Returns:
            Dict with 'success', 'result', and optionally 'error' keys.
        """
        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image not found: {image_path}"}

            if task == "relative_rotation" and image_path2 is None:
                return {
                    "success": False,
                    "error": "'relative_rotation' mode requires 'image_path2'.",
                }

            if image_path2 is not None and not Path(image_path2).exists():
                return {"success": False, "error": f"Image not found: {image_path2}"}

            logger.info(f"Running Orient Anything V2: task={task}, category={object_category}")

            result = self._client.infer(
                image_path=image_path,
                object_category=object_category,
                task=task,
                image_path2=image_path2,
            )

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"OrientAnythingV2Tool error: {e}")
            return {"success": False, "error": str(e)}
