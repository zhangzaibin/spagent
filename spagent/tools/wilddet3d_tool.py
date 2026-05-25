"""
WildDet3D Tool — promptable 3D object detection from single RGB images.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class WildDet3DTool(Tool):
    """Promptable 3D object detection using WildDet3D."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        score_threshold: float = 0.3,
        score_3d_threshold: float = 0.1,
        device: str = "cuda",
        use_mock: bool = False,
    ):
        super().__init__(
            name="wilddet3d_tool",
            description=(
                "WildDet3D: promptable 3D object detection from a single RGB image. "
                "Given a text prompt (e.g. 'chair', 'car', 'object'), detects and localizes "
                "objects in both 2D and 3D. Returns an annotated image with bounding boxes "
                "and 3D location estimates. Useful for object localization, spatial understanding, "
                "and scene analysis tasks."
            ),
        )
        self.use_mock = use_mock
        self._client = None
        self._client_kwargs = dict(
            checkpoint=checkpoint,
            score_threshold=score_threshold,
            score_3d_threshold=score_3d_threshold,
            device=device,
        )

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.use_mock:
            self._client = _MockWildDet3DClient()
        else:
            from external_experts.WildDet3D.wilddet3d_local import WildDet3DLocalClient
            self._client = WildDet3DLocalClient(**self._client_kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input RGB image.",
                },
                "prompt_text": {
                    "type": "string",
                    "description": (
                        "Text prompt describing the object(s) to detect. "
                        "Examples: 'chair', 'person', 'car', 'object' (detects all objects). "
                        "Ignored when input_boxes or input_points are provided. Default: 'object'."
                    ),
                },
                "input_boxes": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Optional 2D bounding box prompt [x1, y1, x2, y2] in pixel coordinates. "
                        "Use when you already know the approximate region of interest. "
                        "Takes priority over prompt_text."
                    ),
                },
                "input_points": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": (
                        "Optional point prompts [[x, y, label], ...] in pixel coordinates. "
                        "label=1 for foreground (object of interest), label=0 for background. "
                        "Takes priority over prompt_text."
                    ),
                },
            },
            "required": ["image_path"],
        }

    def call(
        self,
        image_path: Union[str, List[str]],
        prompt_text: str = "object",
        input_boxes: Optional[List[float]] = None,
        input_points: Optional[List[List]] = None,
    ) -> Dict[str, Any]:
        if isinstance(image_path, list):
            image_path = image_path[0]

        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        try:
            self._ensure_client()
            return self._client.detect(
                image_path=image_path,
                prompt_text=prompt_text,
                input_boxes=input_boxes,
                input_points=input_points,
            )
        except Exception as e:
            logger.exception("WildDet3DTool error")
            return {"success": False, "error": str(e)}


class _MockWildDet3DClient:
    def detect(self, image_path: str, prompt_text: str = "object",
               input_boxes=None, input_points=None, **kwargs) -> Dict[str, Any]:
        mode = "box" if input_boxes else "point" if input_points else f"text:'{prompt_text}'"
        return {
            "success": True,
            "boxes2d": [[100, 100, 300, 300]],
            "boxes3d": [],
            "scores": [0.95],
            "num_detections": 1,
            "output_path": image_path,
            "description": f"[mock] WildDet3D detected 1 object via {mode}.",
        }
