"""
Molmo2 Tool — visual point grounding via the Molmo2 HTTP expert (or mock).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import PointsPayload, ToolResult

logger = logging.getLogger(__name__)


class Molmo2Tool(Tool):
    """Point-grounding tool: locates objects in an image and returns annotated overlays."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://localhost:20025",
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="molmo2_tool",
            description=(
                "Molmo2 point-grounding tool. Given a natural-language instruction, it locates the "
                "described object or region in the image and returns an annotated overlay image "
                "showing the exact position with a marked point. "
                "Always use a short reasoning sentence as the prompt, e.g. "
                "'Point to the object the robot should grasp next.' or "
                "'Point to the item that does not belong with the others.'"
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.output_dir = output_dir or os.environ.get("MOLMO2_OUTPUT_DIR")
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.use_mock:
            self._client = None
            logger.info("Using mock Molmo2 (no HTTP)")
            return
        try:
            from external_experts.Molmo2.molmo2_client import Molmo2Client

            self._client = Molmo2Client(server_url=self.server_url)
            logger.info("Using Molmo2 service at %s", self.server_url)
        except ImportError as e:
            logger.error("Failed to import Molmo2 client: %s", e)
            raise

    @staticmethod
    def _normalize_paths(image_path: Union[str, List[str]]) -> List[str]:
        if isinstance(image_path, (list, tuple)):
            return [str(Path(p)) for p in image_path]
        return [str(Path(image_path))]

    def _generate_text(self, primary_path: str, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        if self.use_mock:
            return {
                "success": True,
                "text": f"[mock] {prompt[:120]} on {Path(primary_path).name}",
            }
        assert self._client is not None
        return self._client.infer_path(primary_path, prompt=prompt, max_new_tokens=max_new_tokens)

    def _run_point(
        self,
        paths: List[str],
        prompt: str,
        save_annotated: bool,
        max_new_tokens: int,
        out_base: Optional[Path],
    ) -> Dict[str, Any]:
        from PIL import Image

        from external_experts.Molmo2.point_utils import (
            annotate_images_as_base64,
            default_output_dir,
            extract_points_from_text,
            group_points_by_image,
            save_annotated_images,
        )

        if self.use_mock:
            raw_text = "Click(50.0, 50.0)"
        else:
            gen = self._generate_text(paths[0], prompt, max_new_tokens)
            if not gen.get("success"):
                return {"success": False, "error": gen.get("error", "generation failed")}
            raw_text = gen.get("text", "")

        sizes: List[tuple] = []
        for p in paths:
            with Image.open(p) as im:
                sizes.append(im.size)
        points = extract_points_from_text(raw_text, sizes)
        grouped = group_points_by_image(points, paths)

        result: Dict[str, Any] = {
            "task": "point",
            "raw_text": raw_text,
            "num_points": len(points),
            "points_by_image": grouped,
        }

        output_path: Optional[str] = None
        if save_annotated and points:
            annotated = annotate_images_as_base64(paths, grouped)
            dest = out_base if out_base is not None else default_output_dir()
            saved = save_annotated_images(annotated, output_dir=dest)
            if saved:
                output_path = saved[0]
            result["saved_paths"] = saved
        elif save_annotated and not points:
            logger.warning("point task: no parseable points in model output")

        saved = result.get("saved_paths", [])
        description = (
            f"Molmo2 pointed to {len(points)} location(s). Annotated image attached."
            if points
            else "Molmo2 found no parseable point coordinates in the model output."
        )
        # Standardized output: point_utils scales the model's 0-1000/0-100
        # output by image width/height, so points are PIXEL coordinates.
        # The payload surfaces the FIRST image's dims (single-image is the
        # common case); `points_by_image` stays as an extra for multi-image.
        flat_points = [pt for group in grouped for pt in group["points"]]
        first_w, first_h = sizes[0] if sizes else (None, None)
        payload = PointsPayload(
            points=flat_points,
            normalized=False,
            image_width=first_w,
            image_height=first_h,
        )
        return ToolResult(
            success=True,
            payload=payload,
            description=description,
            output_path=output_path,
            vis_path=output_path,
            crop_paths=saved[1:] if len(saved) > 1 else [],
            result=result,
            response_text=raw_text[:500],
            raw_text=raw_text,
            points_by_image=grouped,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image.",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "A short reasoning sentence describing what to point to. "
                        "Do NOT just name an object — phrase it as a task, e.g. "
                        "'Point to the object the robot should grasp next.' or "
                        "'Point to the ripest fruit on the table.' or "
                        "'Point to the item that is out of place.' "
                        "This lets the model apply scene understanding before pointing."
                    ),
                },
                "save_annotated": {
                    "type": "boolean",
                    "description": "Save a JPEG overlay with the marked point(s). Default true.",
                    "default": True,
                },
                "max_new_tokens": {"type": "integer", "default": 200},
            },
            "required": ["image_path", "prompt"],
        }

    def call(
        self,
        image_path: Union[str, List[str]],
        prompt: Optional[str] = None,
        save_annotated: bool = True,
        max_new_tokens: int = 200,
        # keep task kwarg for backward compat but ignore it — always point
        task: str = "point",
    ) -> Dict[str, Any]:
        paths = self._normalize_paths(image_path)
        for p in paths:
            if not Path(p).exists():
                return {"success": False, "error": f"Image file not found: {p}"}

        out_base = Path(self.output_dir) if self.output_dir else None
        use_prompt = prompt or "Point to the most salient object in the image."

        try:
            return self._run_point(paths, use_prompt, save_annotated, max_new_tokens, out_base)
        except Exception as e:
            logger.exception("Molmo2 tool error")
            return {"success": False, "error": str(e)}
