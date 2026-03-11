"""
VGGT

Reconstruct 3D scene from RGB images; 
Use when: estimating depth, recovering 3D geometry, or computing camera parameters.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class VGGTTool(Tool):
    """Tool for 3D reconstruction from multiple views using VGGT (Visual Geometry Grounded Transformer)"""

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
        """Initialize VGGT tool
           Args:
                      use_mock: Whether to use mock client for testing
                      server_url: URL of the VGGT server
                  """
                  super().__init__(
                      name="vggt_tool",
                      description=(
                          "Perform 3D scene reconstruction from one or multiple images using VGGT "
                          "(Visual Geometry Grounded Transformer, CVPR 2025 Best Paper). "
                          "VGGT directly predicts all key 3D attributes in a single forward pass within seconds, "
                          "including:\n"
                          "- **Camera parameters**: Extrinsic and intrinsic matrices for each input view\n"
                          "- **Depth maps**: Per-pixel depth estimation for each view\n"
                          "- **Point maps**: Dense 3D point clouds from each view\n"
                          "- **3D point tracks**: Track corresponding points across multiple views\n\n"
                          "**When to use this tool**:\n"
                          "- Estimating camera poses and relative positions between views\n"
                          "- Getting depth information from single or multiple images\n"
                          "- Dense 3D point cloud reconstruction of a scene\n"
                          "- Understanding spatial relationships and 3D structure\n"
                          "- Tracking points across multiple views of the same scene\n\n"
                          "**Input**: One or more image paths (supports up to hundreds of views)\n"
                          "**Output types**: camera_params, depth_map, point_map, full_reconstruction\n\n"
                          "Unlike optimization-based methods (DUSt3R, COLMAP), VGGT is feed-forward and "
                          "produces results in under one second without post-processing."
                      )
                  )
      
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
                    "type": "list",
              "description": (
                        "List of paths to input images for 3D reconstruction. "
                        "VGGT supports single-view (monocular depth/structure), "
                        "few-view (2-10 images), and many-view (up to hundreds) reconstruction."
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
                        "Type of 3D output to produce. "
                        "'camera_params': estimate extrinsic/intrinsic camera parameters for each view. "
                        "'depth_map': per-pixel depth estimation for each view. "
                        "'point_map': dense 3D point cloud from each view. "
                        "'full_reconstruction': all of the above (cameras + depth + point cloud). "
                        "Default is 'full_reconstruction'."
                    ),
                    "default": "full_reconstruction",
                },
            },
            "required": ["image_path"],
        }


    def call(self, image_path: str, option: str = "full_reconstruction") -> Dict[str, Any]:
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
