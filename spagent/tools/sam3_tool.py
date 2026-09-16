"""
SAM3 Segmentation Tool

Wraps SAM3 image/video text-prompt segmentation for SPAgent.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import SEGMENTATION, SegmentationPayload, ToolResult

logger = logging.getLogger(__name__)

_ENVELOPE_KEYS = {
    "success", "description", "error", "category", "payload",
    "output_path", "vis_path", "overlay_path", "crop_paths", "boxes",
}


class SAM3Tool(Tool):
    """Tool for image and video concept segmentation using SAM3."""

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, use_mock: bool = True, server_url: str = "http://127.0.0.1:20035"):
        super().__init__(
            name="sam3_concept_segmentation_tool",
            description=(
                "Segment objects in an image or video using SAM3 from a natural-language text prompt. "
                "Use this for concept-based image/video segmentation such as finding all people, chairs, "
                "bottles, or other described objects."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            from external_experts.SAM3.mock_sam3_service import MockSAM3Service

            self._client = MockSAM3Service()
            logger.info("Using mock SAM3 service")
        else:
            from external_experts.SAM3.sam3_client import SAM3Client

            self._client = SAM3Client(server_url=self.server_url)
            logger.info("Using real SAM3 service at %s", self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image, video file, or JPEG frame directory.",
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Natural-language concept to segment, such as 'person', 'red bottle', or 'office chair'.",
                },
                "task": {
                    "type": "string",
                    "enum": ["auto", "image", "video"],
                    "description": "Segmentation mode. Use 'auto' to infer image vs video from the path.",
                    "default": "auto",
                },
                "frame_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Video frame index where the text prompt is added. Ignored for image inputs.",
                    "default": 0,
                },
                "score_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum score for returned SAM3 instances.",
                    "default": 0.5,
                },
                "max_instances": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of segmented instances to return.",
                    "default": 20,
                },
                "save_overlay": {
                    "type": "boolean",
                    "description": "Whether to save an overlay visualization image or video.",
                    "default": True,
                },
            },
            "required": ["image_path", "text_prompt"],
        }

    def call(
        self,
        image_path: str,
        text_prompt: str,
        task: str = "auto",
        frame_index: int = 0,
        score_threshold: float = 0.5,
        max_instances: int = 20,
        save_overlay: bool = True,
    ) -> Dict[str, Any]:
        try:
            if not text_prompt or not text_prompt.strip():
                return ToolResult.fail("text_prompt must be a non-empty string.", category=SEGMENTATION)

            path = Path(image_path)
            if not path.exists():
                return ToolResult.fail(f"Input file not found: {image_path}", category=SEGMENTATION)

            if not 0.0 <= float(score_threshold) <= 1.0:
                return ToolResult.fail("score_threshold must be between 0 and 1.", category=SEGMENTATION)
            if int(max_instances) < 1:
                return ToolResult.fail("max_instances must be at least 1.", category=SEGMENTATION)
            if int(frame_index) < 0:
                return ToolResult.fail("frame_index must be non-negative.", category=SEGMENTATION)

            resolved_task = self._resolve_task(path, task)
            if resolved_task == "image" and path.is_dir():
                return ToolResult.fail(
                    "Image task requires an image file, not a directory.",
                    category=SEGMENTATION,
                )
            common_args = {
                "text_prompt": text_prompt.strip(),
                "score_threshold": float(score_threshold),
                "max_instances": int(max_instances),
                "save_overlay": bool(save_overlay),
            }

            if resolved_task == "image":
                result = self._client.infer(image_path=str(path), **common_args)
            elif resolved_task == "video":
                result = self._client.infer_video(
                    video_path=str(path),
                    frame_index=int(frame_index),
                    **common_args,
                )
            else:
                return ToolResult.fail(f"Unsupported task: {task}", category=SEGMENTATION)

            if result and result.get("success"):
                raw = dict(result)
                raw["result"] = result
                raw["task"] = result.get("task", resolved_task)
                raw.setdefault("masks", [])
                raw.setdefault("boxes", [])
                raw.setdefault("scores", [])

                mask_path = result.get("mask_path")
                masks = result.get("masks") or []
                payload = None
                if mask_path:
                    payload = SegmentationPayload(mask_path=mask_path)
                elif masks:
                    payload = SegmentationPayload(masks=masks)

                if resolved_task == "video":
                    description = result.get("description") or (
                        f"SAM3 segmented '{text_prompt.strip()}' across "
                        f"{result.get('frames', 0)} video frame(s). The overlay MP4 is "
                        "in output_path and masks contains frame-indexed mask paths."
                    )
                else:
                    description = result.get("description") or (
                        f"SAM3 segmented '{text_prompt.strip()}' in the image."
                    )
                extras = {key: value for key, value in raw.items() if key not in _ENVELOPE_KEYS}
                return ToolResult(
                    success=True,
                    payload=payload,
                    category=SEGMENTATION,
                    description=description,
                    output_path=result.get("output_path") or result.get("video_path"),
                    overlay_path=result.get("overlay_path"),
                    **extras,
                )

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return ToolResult.fail(f"SAM3 segmentation failed: {error_msg}", category=SEGMENTATION)
        except Exception as e:
            logger.error("SAM3 tool error: %s", e)
            return ToolResult.fail(str(e), category=SEGMENTATION)

    def _resolve_task(self, path: Path, task: str) -> str:
        if task not in {"auto", "image", "video"}:
            raise ValueError("task must be one of: auto, image, video")
        if task != "auto":
            return task
        if path.is_dir():
            return "video"
        suffix = path.suffix.lower()
        if suffix in self.VIDEO_EXTENSIONS:
            return "video"
        if suffix in self.IMAGE_EXTENSIONS:
            return "image"
        return "image"
