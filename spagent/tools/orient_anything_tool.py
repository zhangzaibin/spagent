"""
Orient Anything Tool

Estimate object orientation from a single-object image using Orient Anything.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class OrientAnythingTool(Tool):
    """Tool for image-based object orientation estimation."""

    def __init__(
        self,
        use_mock: bool = True,
        repo_root: str = "./../Orient-Anything",
        device: str = "cuda:0",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            name="orient_anything_tool",
            description=(
                "Estimate the 3D orientation of a single prominent object in an image. "
                "Returns azimuth, polar angle, in-plane rotation, and confidence. "
                "Best used when the image contains one main object."
            )
        )
        self.use_mock = use_mock
        self.repo_root = repo_root
        self.device = device
        self.cache_dir = cache_dir
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            from external_experts.orient_anything.mock_orient_anything import MockOrientAnything
            self._client = MockOrientAnything()
        else:
            from external_experts.orient_anything.orient_anything_client import OrientAnythingClient
            self._client = OrientAnythingClient(
                repo_root=self.repo_root,
                device=self.device,
                cache_dir=self.cache_dir,
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image containing a single main object."
                },
                "model_size": {
                    "type": "string",
                    "enum": ["small", "base", "large"],
                    "description": "Which Orient Anything checkpoint to use.",
                    "default": "large"
                },
                "use_tta": {
                    "type": "boolean",
                    "description": "Whether to use test-time augmentation for more robust prediction.",
                    "default": False
                },
                "remove_background": {
                    "type": "boolean",
                    "description": "Whether to remove background before orientation estimation.",
                    "default": True
                },
                "device": {
                    "type": "string",
                    "description": "Device string such as 'cpu' or 'cuda:0'.",
                    "default": "cuda:0"
                }
            },
            "required": ["image_path"]
        }

    def call(
        self,
        image_path: str,
        model_size: str = "large",
        use_tta: bool = False,
        remove_background: bool = True,
        device: str = "cuda:0",
    ) -> Dict[str, Any]:
        try:
            p = Path(image_path)
            if not p.exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            result = self._client.predict(
                image_path=str(p),
                model_size=model_size,
                use_tta=use_tta,
                remove_background=remove_background,
                device=device,
            )

            if not result or not result.get("success", False):
                return {
                    "success": False,
                    "error": result.get("error", "Unknown Orient Anything error") if result else "No result"
                }

            azimuth = result["azimuth"]
            polar = result["polar"]
            rotation = result["rotation"]
            confidence = result["confidence"]

            summary = (
                f"Estimated orientation: azimuth={azimuth:.2f}, polar={polar:.2f}, "
                f"rotation={rotation:.2f}, confidence={confidence:.3f}."
            )

            return {
                "success": True,
                "result": {
                    "azimuth": azimuth,
                    "polar": polar,
                    "rotation": rotation,
                    "confidence": confidence,
                    "model_size": result.get("model_size", model_size),
                    "use_tta": result.get("use_tta", use_tta),
                    "remove_background": result.get("remove_background", remove_background),
                },
                "output_path": result.get("visualization_path"),
                "summary": summary,
            }
        except Exception as e:
            logger.exception("OrientAnythingTool error")
            return {
                "success": False,
                "error": str(e)
            }