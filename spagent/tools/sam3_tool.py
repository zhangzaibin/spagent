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

logger = logging.getLogger(__name__)


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
                return {"success": False, "error": "text_prompt must be a non-empty string."}

            path = Path(image_path)
            if not path.exists():
                return {"success": False, "error": f"Input file not found: {image_path}"}

            resolved_task = self._resolve_task(path, task)
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
                return {"success": False, "error": f"Unsupported task: {task}"}

            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "task": result.get("task", resolved_task),
                    "output_path": result.get("output_path"),
                    "overlay_path": result.get("overlay_path"),
                    "mask_path": result.get("mask_path"),
                    "video_path": result.get("video_path"),
                    "shape": result.get("shape"),
                    "frames": result.get("frames"),
                    "fps": result.get("fps"),
                    "size": result.get("size"),
                    "masks": result.get("masks", []),
                    "boxes": result.get("boxes", []),
                    "scores": result.get("scores", []),
                }

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return {"success": False, "error": f"SAM3 segmentation failed: {error_msg}"}
        except Exception as e:
            logger.error("SAM3 tool error: %s", e)
            return {"success": False, "error": str(e)}

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

