"""
VGGT Tool

Reconstruct 3D scene from RGB images.
Use when: estimating depth, recovering 3D geometry, or computing camera parameters.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent))
from core.tool import Tool

logger = logging.getLogger(__name__)


class VGGTTool(Tool):
    """Tool for 3D reconstruction from multiple views using VGGT."""

    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:8000"):
        super().__init__(
            name="vggt_tool",
            description=(
                "Perform 3D scene reconstruction from one or multiple images using VGGT "
                "(Visual Geometry Grounded Transformer). "
                "Use this tool for estimating camera parameters, depth maps, point maps, "
                "or full 3D reconstruction from RGB images."
            )
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the underlying client."""
        if self.use_mock:
            self._client = MockMyService()
        else:
            self._client = RealMyClient(server_url=self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of input image paths for 3D reconstruction. "
                        "Supports one or multiple views."
                    ),
                },
                "output_type": {
                    "type": "string",
                    "enum": [
                        "camera_params",
                        "depth_map",
                        "point_map",
                        "full_reconstruction",
                    ],
                    "description": (
                        "Type of output to produce: camera parameters, depth map, "
                        "point map, or full reconstruction."
                    ),
                    "default": "full_reconstruction",
                },
            },
            "required": ["image_paths"],
        }

    def call(
        self,
        image_paths: List[str],
        output_type: str = "full_reconstruction"
    ) -> Dict[str, Any]:
        try:
            missing = [p for p in image_paths if not Path(p).exists()]
            if missing:
                return {
                    "success": False,
                    "error": f"Image(s) not found: {missing}"
                }

            result = self._client.process(
                image_paths=image_paths,
                output_type=output_type
            )

            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                }

            return {
                "success": False,
                "error": result.get("error", "Unknown error") if result else "No result"
            }

        except Exception as e:
            logger.error(f"VGGTTool error: {e}")
            return {"success": False, "error": str(e)}
