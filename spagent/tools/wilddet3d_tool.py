"""
WildDet3D Tool

Wraps promptable monocular 3D object detection for SPAgent.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class WildDet3DTool(Tool):
    """Tool for open-vocabulary and promptable 3D object detection."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://127.0.0.1:20036",
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="wilddet3d_3d_detection_tool",
            description=(
                "Detect and localize objects in 3D from a single image using WildDet3D. "
                "Supports text prompts, 2D box prompts, and point prompts for promptable "
                "open-vocabulary 3D object detection."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.output_dir = output_dir
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.use_mock:
            from external_experts.WildDet3D.mock_wilddet3d_service import MockWildDet3DService

            self._client = MockWildDet3DService(output_dir=self.output_dir)
            logger.info("Using mock WildDet3D service")
        else:
            from external_experts.WildDet3D.wilddet3d_client import WildDet3DClient

            self._client = WildDet3DClient(server_url=self.server_url, output_dir=self.output_dir)
            logger.info("Using real WildDet3D service at %s", self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image.",
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Object category or comma-separated categories, such as 'chair' or 'car, person'.",
                },
                "boxes": {
                    "type": "array",
                    "description": "Optional 2D box prompts in pixel xyxy format: [[x1, y1, x2, y2], ...].",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "points": {
                    "type": "array",
                    "description": "Optional point prompts in pixel format: [[x, y, label], ...], where label is 1 or 0.",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                "score_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum score for returned detections.",
                    "default": 0.3,
                },
                "save_visualization": {
                    "type": "boolean",
                    "description": "Whether to save a visualization image with 3D detection results.",
                    "default": True,
                },
            },
            "required": ["image_path"],
            "anyOf": [
                {"required": ["text_prompt"]},
                {"required": ["boxes"]},
                {"required": ["points"]},
            ],
        }

    def call(
        self,
        image_path: str,
        text_prompt: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
        score_threshold: float = 0.3,
        save_visualization: bool = True,
    ) -> Dict[str, Any]:
        try:
            path = Path(image_path)
            if not path.exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            clean_prompt = text_prompt.strip() if isinstance(text_prompt, str) else ""
            clean_boxes = self._validate_boxes(boxes)
            clean_points = self._validate_points(points)

            if not clean_prompt and not clean_boxes and not clean_points:
                return {
                    "success": False,
                    "error": "Provide at least one prompt: text_prompt, boxes, or points.",
                }

            result = self._client.infer(
                image_path=str(path),
                text_prompt=clean_prompt or None,
                boxes=clean_boxes or None,
                points=clean_points or None,
                score_threshold=float(score_threshold),
                save_visualization=bool(save_visualization),
            )

            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "boxes_2d": result.get("boxes_2d", []),
                    "boxes_3d": result.get("boxes_3d", []),
                    "scores": result.get("scores", []),
                    "class_names": result.get("class_names", []),
                    "depth_path": result.get("depth_path"),
                    "output_path": result.get("output_path"),
                }

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return {"success": False, "error": f"WildDet3D detection failed: {error_msg}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error("WildDet3D tool error: %s", e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _validate_boxes(boxes: Optional[List[List[float]]]) -> List[List[float]]:
        if boxes is None:
            return []
        if not isinstance(boxes, list):
            raise ValueError("boxes must be a list of [x1, y1, x2, y2] boxes.")
        normalized = []
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("Each box must have four values: [x1, y1, x2, y2].")
            normalized.append([float(v) for v in box])
        return normalized

    @staticmethod
    def _validate_points(points: Optional[List[List[float]]]) -> List[List[float]]:
        if points is None:
            return []
        if not isinstance(points, list):
            raise ValueError("points must be a list of [x, y, label] points.")
        normalized = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError("Each point must have three values: [x, y, label].")
            normalized.append([float(point[0]), float(point[1]), int(point[2])])
        return normalized
