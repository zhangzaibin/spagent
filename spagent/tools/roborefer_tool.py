from __future__ import annotations

import sys
import re
import ast
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from external_experts.RoboRefer.roborefer_client import RoboReferClient
from external_experts.RoboRefer.mock_roborefer_service import MockRoboReferService

logger = logging.getLogger(__name__)


def _ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def _normalize_points_structure(obj: Any) -> List[Tuple[float, float]]:
    if isinstance(obj, tuple) and len(obj) == 2:
        obj = [obj]

    if not isinstance(obj, list):
        raise ValueError(f"Expected list of points, got: {type(obj)}")

    points: List[Tuple[float, float]] = []
    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid point format: {item}")
        x, y = float(item[0]), float(item[1])
        points.append((x, y))

    return points


def _safe_literal_eval_points(text: str) -> List[Tuple[float, float]]:
    if not text:
        raise ValueError("Empty answer from RoboRefer")

    candidate = text.strip()

    try:
        parsed = ast.literal_eval(candidate)
        return _normalize_points_structure(parsed)
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", candidate)
    if match:
        try:
            parsed = ast.literal_eval(match.group(0))
            return _normalize_points_structure(parsed)
        except Exception:
            pass

    tuple_matches = re.findall(
        r"\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)",
        candidate
    )
    if tuple_matches:
        return [(float(x), float(y)) for x, y in tuple_matches]

    raise ValueError(f"Failed to parse normalized points from answer: {text}")


def _normalized_to_pixel_points(
    points: List[Tuple[float, float]],
    width: int,
    height: int
) -> List[Tuple[int, int]]:
    pixel_points = []
    for x, y in points:
        px = int(round(x * width))
        py = int(round(y * height))
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        pixel_points.append((px, py))
    return pixel_points


def _draw_points_on_image(
    image_path: str,
    pixel_points: List[Tuple[int, int]],
    output_path: str,
    radius: int = 12,
    border_thickness: int = 2,
    color: Tuple[int, int, int] = (244, 133, 66),
    border_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    for x, y in pixel_points:
        cv2.circle(image, (x, y), radius + border_thickness, border_color, thickness=-1)
        cv2.circle(image, (x, y), radius, color, thickness=-1)

    _ensure_parent_dir(output_path)
    cv2.imwrite(str(output_path), image)


class RoboReferTool(Tool):
    """Tool for spatial referring with RoboRefer."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://127.0.0.1:25547",
        timeout: int = 120,
        default_enable_depth: int = 1,
        default_output_dir: Optional[str] = None,
        append_output_format_hint: bool = True,
    ):
        super().__init__(
            name="roborefer_tool",
            description=(
                "Use RoboRefer for spatial referring in an image. "
                "Given an image and a natural-language prompt, it returns normalized point coordinates "
                "for the referred target or interaction location."
            )
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.timeout = timeout
        self.default_enable_depth = int(default_enable_depth)
        self.default_output_dir = default_output_dir
        self.append_output_format_hint = append_output_format_hint
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.use_mock:
            self._client = MockRoboReferService()
            logger.info("Initialized MockRoboReferService")
        else:
            self._client = RoboReferClient(
                server_url=self.server_url,
                timeout=self.timeout,
            )
            logger.info("Initialized RoboReferClient with server_url=%s", self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the RGB image for RoboRefer."
                },
                "prompt": {
                    "type": "string",
                    "description": "Natural-language spatial referring instruction."
                },
                "depth_path": {
                    "type": "string",
                    "description": "Optional path to a depth image aligned with the RGB image."
                },
                "enable_depth": {
                    "type": "integer",
                    "enum": [0, 1],
                    "description": "Whether to use depth input or depth-assisted inference.",
                    "default": 1
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to save the visualization image."
                },
                "return_visualization": {
                    "type": "boolean",
                    "description": "Whether to save an image with predicted points drawn on it.",
                    "default": True
                }
            },
            "required": ["image_path", "prompt"]
        }

    def _build_prompt(self, prompt: str) -> str:
        if not self.append_output_format_hint:
            return prompt

        suffix = (
            " Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], "
            "where each tuple contains the x and y coordinates of a point satisfying the "
            "conditions above. The coordinates should be between 0 and 1, indicating the "
            "normalized pixel locations of the points in the image."
        )
        return prompt.strip() + suffix

    def _resolve_output_path(self, image_path: str, output_path: Optional[str]) -> str:
        if output_path:
            return output_path

        image_p = Path(image_path)
        out_dir = Path(self.default_output_dir) if self.default_output_dir else image_p.parent
        out_name = f"{image_p.stem}_roborefer_result{image_p.suffix}"
        return str(out_dir / out_name)

    def call(
        self,
        image_path: str,
        prompt: str,
        depth_path: Optional[str] = None,
        enable_depth: int = 1,
        output_path: Optional[str] = None,
        return_visualization: bool = True,
    ) -> Dict[str, Any]:
        try:
            image_p = Path(image_path)
            if not image_p.exists():
                return {"success": False, "error": f"Image not found: {image_path}"}

            if depth_path is not None and not Path(depth_path).exists():
                return {"success": False, "error": f"Depth image not found: {depth_path}"}

            final_enable_depth = int(enable_depth if enable_depth is not None else self.default_enable_depth)
            final_prompt = self._build_prompt(prompt)

            response = self._client.query(
                image_path=image_path,
                prompt=final_prompt,
                enable_depth=final_enable_depth,
                depth_path=depth_path,
            )

            raw_answer = response.get("answer")
            if raw_answer is None:
                return {
                    "success": False,
                    "error": f"RoboRefer response missing 'answer': {response}"
                }

            normalized_points = _safe_literal_eval_points(raw_answer)

            import cv2
            img = cv2.imread(str(image_p))
            if img is None:
                return {"success": False, "error": f"Failed to read image: {image_path}"}

            height, width = img.shape[:2]
            pixel_points = _normalized_to_pixel_points(normalized_points, width, height)

            saved_output_path = None
            if return_visualization:
                saved_output_path = self._resolve_output_path(image_path, output_path)
                _draw_points_on_image(
                    image_path=image_path,
                    pixel_points=pixel_points,
                    output_path=saved_output_path,
                )

            result = {
                "normalized_points": normalized_points,
                "pixel_points": pixel_points,
                "image_size": {"width": width, "height": height},
                "raw_answer": raw_answer,
                "server_response": response,
                "used_depth": bool(final_enable_depth),
                "depth_path": depth_path,
            }

            if saved_output_path:
                result["output_path"] = saved_output_path

            summary = (
                f"RoboRefer found {len(pixel_points)} point(s): "
                + ", ".join([f"({x}, {y})" for x, y in pixel_points])
            )

            return {
                "success": True,
                "result": result,
                "output_path": saved_output_path,
                "summary": summary,
            }

        except Exception as e:
            logger.exception("RoboReferTool error")
            return {
                "success": False,
                "error": str(e),
            }