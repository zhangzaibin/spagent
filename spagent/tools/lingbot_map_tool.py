"""
LingBot-Map Tool

Wraps LingBot-Map long-sequence 3D scene mapping for SPAgent.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LingBotMapTool(Tool):
    """Tool for building an interactive 3D map from an image sequence using LingBot-Map."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://127.0.0.1:20038",
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="lingbot_map_tool",
            description=(
                "Build an interactive 3D scene map from an ordered image sequence using LingBot-Map. "
                "Input can be either an image folder or an explicit list of image paths."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.output_dir = output_dir
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.use_mock:
            from external_experts.LingBotMap.mock_lingbot_map_service import MockLingBotMapService

            self._client = MockLingBotMapService(output_dir=self.output_dir)
            logger.info("Using mock LingBot-Map service")
        else:
            from external_experts.LingBotMap.lingbot_map_client import LingBotMapClient

            self._client = LingBotMapClient(server_url=self.server_url, output_dir=self.output_dir)
            logger.info("Using real LingBot-Map service at %s", self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_folder": {
                    "type": "string",
                    "description": "Path to a folder containing ordered input images.",
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of image paths. Use this instead of image_folder.",
                },
                "mask_sky": {
                    "type": "boolean",
                    "description": "Whether to enable LingBot-Map sky masking.",
                    "default": False,
                },
                "keyframe_interval": {
                    "type": "integer",
                    "description": "Use every Nth image as a keyframe.",
                    "default": 1,
                    "minimum": 1,
                },
                "max_frames": {
                    "type": "integer",
                    "description": "Maximum number of frames to send to the backend.",
                    "default": 128,
                    "minimum": 1,
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory for preview, trajectory, and point cloud files.",
                },
                "wait_for_completion": {
                    "type": "boolean",
                    "description": "Wait for the backend command to finish and collect files. Use false for interactive viewer mode.",
                    "default": False,
                },
            },
            "oneOf": [{"required": ["image_folder"]}, {"required": ["image_paths"]}],
        }

    def call(
        self,
        image_folder: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        mask_sky: bool = False,
        keyframe_interval: int = 1,
        max_frames: int = 128,
        output_dir: Optional[str] = None,
        wait_for_completion: bool = False,
    ) -> Dict[str, Any]:
        try:
            valid, error, normalized_paths = self._validate_inputs(
                image_folder=image_folder,
                image_paths=image_paths,
                keyframe_interval=keyframe_interval,
                max_frames=max_frames,
            )
            if not valid:
                return {"success": False, "error": error}

            result = self._client.infer(
                image_folder=image_folder,
                image_paths=normalized_paths,
                mask_sky=bool(mask_sky),
                keyframe_interval=int(keyframe_interval),
                max_frames=int(max_frames),
                output_dir=output_dir,
                wait_for_completion=bool(wait_for_completion),
            )

            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "output_dir": result.get("output_dir"),
                    "viewer_url": result.get("viewer_url"),
                    "preview_path": result.get("preview_path"),
                    "trajectory_path": result.get("trajectory_path"),
                    "point_cloud_path": result.get("point_cloud_path"),
                    "video_path": result.get("video_path"),
                    "num_frames": result.get("num_frames"),
                    "process_id": result.get("process_id"),
                    "log_path": result.get("log_path"),
                    "command": result.get("command"),
                }

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return {"success": False, "error": f"LingBot-Map failed: {error_msg}"}
        except Exception as e:
            logger.error("LingBot-Map tool error: %s", e)
            return {"success": False, "error": str(e)}

    def _validate_inputs(
        self,
        image_folder: Optional[str],
        image_paths: Optional[List[str]],
        keyframe_interval: int,
        max_frames: int,
    ) -> tuple[bool, Optional[str], Optional[List[str]]]:
        if bool(image_folder) == bool(image_paths):
            return False, "Provide exactly one of image_folder or image_paths.", None
        try:
            if int(keyframe_interval) < 1:
                return False, "keyframe_interval must be >= 1.", None
            if int(max_frames) < 1:
                return False, "max_frames must be >= 1.", None
        except (TypeError, ValueError):
            return False, "keyframe_interval and max_frames must be integers.", None

        if image_folder:
            folder = Path(image_folder)
            if not folder.exists() or not folder.is_dir():
                return False, f"Image folder not found: {image_folder}", None
            if not [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]:
                return False, f"No supported image files found in: {image_folder}", None
            return True, None, None

        if not isinstance(image_paths, list) or len(image_paths) == 0:
            return False, "image_paths must be a non-empty list.", None
        normalized_paths = []
        for image_path in image_paths:
            path = Path(image_path)
            if not path.exists() or not path.is_file():
                return False, f"Image file not found: {image_path}", None
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                return False, f"Unsupported image file extension: {image_path}", None
            normalized_paths.append(str(path))
        return True, None, normalized_paths
