"""
WildDet3D Tool

Wraps promptable monocular 3D object detection for SPAgent.
"""

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import BOX_XYXY_PIXEL, DETECTION, DetectionPayload, ToolResult

logger = logging.getLogger(__name__)

_ENVELOPE_KEYS = {
    "success", "description", "error", "category", "payload",
    "output_path", "vis_path", "overlay_path", "crop_paths",
}


class WildDet3DTool(Tool):
    """Tool for open-vocabulary and promptable 3D object detection."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: str = "cuda",
        checkpoint: Optional[str] = None,
        score_threshold: float = 0.3,
        score_3d_threshold: float = 0.1,
    ):
        super().__init__(
            name="wilddet3d_tool",
            description=(
                "Detect and localize objects in 3D from a single image using WildDet3D. "
                "Supports text prompts, 2D box prompts, and point prompts for promptable "
                "open-vocabulary 3D object detection."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url or os.environ.get("WILDDET3D_SERVER_URL")
        self.output_dir = output_dir
        self.device = device
        self.checkpoint = checkpoint
        self.default_score_threshold = self._validate_threshold(score_threshold)
        self.score_3d_threshold = self._validate_threshold(
            score_3d_threshold, name="score_3d_threshold", maximum=1.0
        )
        self._client = None
        self._backend_kind = ""
        self._init_client()

    def _init_client(self) -> None:
        if self.use_mock:
            from external_experts.WildDet3D.mock_wilddet3d_service import MockWildDet3DService

            self._client = MockWildDet3DService(output_dir=self.output_dir)
            self._backend_kind = "mock"
            logger.info("Using mock WildDet3D service")
        elif self.server_url:
            from external_experts.WildDet3D.wilddet3d_client import WildDet3DClient

            self._client = WildDet3DClient(server_url=self.server_url, output_dir=self.output_dir)
            self._backend_kind = "server"
            logger.info("Using real WildDet3D service at %s", self.server_url)
        else:
            self._backend_kind = "local"
            logger.info("Using lazy local WildDet3D checkpoint backend")

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
                    "minItems": 1,
                    "maxItems": 20,
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
                    "minItems": 1,
                    "maxItems": 100,
                },
                "score_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "description": "Minimum WildDet3D combined ranking score for returned detections.",
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
        image_path: Union[str, List[str]],
        text_prompt: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
        score_threshold: Optional[float] = None,
        save_visualization: bool = True,
        prompt_text: Optional[str] = None,
        input_boxes: Optional[List[float]] = None,
        input_points: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        try:
            if isinstance(image_path, list):
                if not image_path:
                    return ToolResult.fail("image_path list must not be empty.", category=DETECTION)
                image_path = image_path[0]
            path = Path(image_path)
            if not path.is_file():
                return ToolResult.fail(f"Image file not found: {image_path}", category=DETECTION)

            with Image.open(path) as image:
                image.verify()
                image_size = image.size

            if text_prompt is None:
                text_prompt = prompt_text
            if boxes is None and input_boxes is not None:
                boxes = [input_boxes] if self._is_single_box(input_boxes) else input_boxes
            if points is None:
                points = input_points

            clean_prompt = text_prompt.strip() if isinstance(text_prompt, str) else ""
            if len(clean_prompt) > 1000:
                raise ValueError("text_prompt must contain at most 1000 characters.")
            prompt_categories = [
                part.strip()
                for part in clean_prompt.replace(".", ",").split(",")
                if part.strip()
            ]
            if len(prompt_categories) > 50:
                raise ValueError("text_prompt must contain at most 50 categories.")
            clean_boxes = self._validate_boxes(boxes, image_size)
            clean_points = self._validate_points(points, image_size)

            # Match the pre-existing local API's precedence while ensuring the
            # HTTP server receives one unambiguous prompt mode.
            if clean_boxes:
                clean_prompt, clean_points = "", []
            elif clean_points:
                clean_prompt = ""

            if not clean_prompt and not clean_boxes and not clean_points:
                if self._backend_kind == "local":
                    clean_prompt = "object"
                else:
                    return ToolResult.fail(
                        "Provide at least one prompt: text_prompt, boxes, or points.",
                        category=DETECTION,
                    )

            threshold = self._validate_threshold(
                self.default_score_threshold if score_threshold is None else score_threshold
            )
            if self._backend_kind == "local" and len(clean_boxes) > 1:
                return ToolResult.fail(
                    "The local WildDet3D backend accepts one box per call; use the server backend for multiple boxes.",
                    category=DETECTION,
                )

            if self._backend_kind == "local":
                if self._client is None:
                    from external_experts.WildDet3D.wilddet3d_local import WildDet3DLocalClient

                    self._client = WildDet3DLocalClient(
                        checkpoint=self.checkpoint,
                        score_threshold=0.0,
                        score_3d_threshold=self.score_3d_threshold,
                        device=self.device,
                    )
                result = self._client.detect(
                    image_path=str(path),
                    prompt_text=clean_prompt or "object",
                    input_boxes=clean_boxes[0] if clean_boxes else None,
                    input_points=clean_points or None,
                    score_threshold=threshold,
                )
            else:
                result = self._client.infer(
                    image_path=str(path),
                    text_prompt=clean_prompt or None,
                    boxes=clean_boxes or None,
                    points=clean_points or None,
                    score_threshold=threshold,
                    save_visualization=bool(save_visualization),
                )

            if result and result.get("success"):
                boxes_2d = result.get("boxes_2d", result.get("boxes2d", [])) or []
                boxes_3d = result.get("boxes_3d", result.get("boxes3d", [])) or []
                scores = result.get("scores", []) or []
                scores_2d = result.get("scores_2d", []) or []
                class_names = result.get("class_names", []) or []
                if not class_names:
                    class_names = [clean_prompt or "object"] * len(boxes_2d)
                labels = [
                    class_names[index] if index < len(class_names) else (clean_prompt or "object")
                    for index in range(len(boxes_2d))
                ]
                payload = DetectionPayload(
                    boxes=boxes_2d,
                    labels=labels,
                    box_format=BOX_XYXY_PIXEL,
                    confidence=(
                        scores_2d
                        if len(scores_2d) == len(boxes_2d)
                        else scores if len(scores) == len(boxes_2d) else None
                    ),
                )
                raw = dict(result)
                raw.setdefault("boxes_2d", boxes_2d)
                raw.setdefault("boxes_3d", boxes_3d)
                raw.setdefault("scores", scores)
                raw.setdefault("class_names", class_names)
                # Preserve the result keys already released on main while
                # exposing the clearer snake_case aliases used by this API.
                raw.setdefault("boxes2d", boxes_2d)
                raw.setdefault("boxes3d", boxes_3d)
                raw.setdefault("num_detections", len(boxes_2d))
                raw["result"] = {
                    "boxes2d": boxes_2d,
                    "boxes3d": boxes_3d,
                    "scores": scores,
                    "num_detections": len(boxes_2d),
                }
                description = result.get("description") or (
                    f"WildDet3D detected {len(boxes_2d)} object(s)."
                )
                extras = {key: value for key, value in raw.items() if key not in _ENVELOPE_KEYS}
                tool_result = ToolResult(
                    success=True,
                    payload=payload,
                    description=description,
                    output_path=result.get("output_path"),
                    **extras,
                )
                # Preserve the original WildDet3D response shape when
                # visualization is disabled; ToolResult omits None paths.
                tool_result.setdefault("output_path", result.get("output_path"))
                return tool_result

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return ToolResult.fail(f"WildDet3D detection failed: {error_msg}", category=DETECTION)
        except ValueError as e:
            return ToolResult.fail(str(e), category=DETECTION)
        except Exception as e:
            logger.error("WildDet3D tool error: %s", e)
            return ToolResult.fail(str(e), category=DETECTION)

    @staticmethod
    def _is_single_box(boxes: List) -> bool:
        return len(boxes) == 4 and all(isinstance(value, (int, float)) for value in boxes)

    @staticmethod
    def _validate_boxes(
        boxes: Optional[List[List[float]]], image_size
    ) -> List[List[float]]:
        if boxes is None:
            return []
        if not isinstance(boxes, list):
            raise ValueError("boxes must be a list of [x1, y1, x2, y2] boxes.")
        if len(boxes) > 20:
            raise ValueError("boxes must contain at most 20 prompts.")
        normalized = []
        width, height = image_size
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("Each box must have four values: [x1, y1, x2, y2].")
            values = [float(v) for v in box]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Box coordinates must be finite numbers.")
            x1, y1, x2, y2 = values
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(
                    f"Each box must be a non-empty pixel xyxy region inside the {width}x{height} image."
                )
            normalized.append(values)
        return normalized

    @staticmethod
    def _validate_points(
        points: Optional[List[List[float]]], image_size
    ) -> List[List[float]]:
        if points is None:
            return []
        if not isinstance(points, list):
            raise ValueError("points must be a list of [x, y, label] points.")
        if len(points) > 100:
            raise ValueError("points must contain at most 100 prompts.")
        normalized = []
        width, height = image_size
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError("Each point must have three values: [x, y, label].")
            x, y, raw_label = float(point[0]), float(point[1]), float(point[2])
            if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(raw_label):
                raise ValueError("Point coordinates and labels must be finite numbers.")
            if raw_label not in (0.0, 1.0):
                raise ValueError("Point labels must be 0 (background) or 1 (foreground).")
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Each point must lie inside the {width}x{height} image.")
            normalized.append([x, y, int(raw_label)])
        if normalized and not any(point[2] == 1 for point in normalized):
            raise ValueError("At least one point must have foreground label 1.")
        return normalized

    @staticmethod
    def _validate_threshold(
        value, name: str = "score_threshold", maximum: Optional[float] = None
    ) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or (
            maximum is not None and value > maximum
        ):
            suffix = (
                f" in [0, {maximum:g}]"
                if maximum is not None
                else " greater than or equal to 0"
            )
            raise ValueError(f"{name} must be a finite number{suffix}.")
        return value
