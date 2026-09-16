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
from core.tool_result import RECONSTRUCTION_3D, PointCloudPayload, ToolResult

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_REAL_FRAMES = 8


class LingBotMapTool(Tool):
    """Tool for reconstructing a 3D point cloud and camera trajectory with LingBot-Map."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://127.0.0.1:20040",
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="lingbot_map_tool",
            description=(
                "Reconstruct a 3D scene from at least eight ordered images using LingBot-Map. "
                "Returns point-cloud, camera-trajectory, and preview artifacts."
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
                    "description": (
                        "Path to a folder containing at least eight ordered input images. "
                        "Provide exactly one of image_folder or image_paths."
                    ),
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Ordered list of at least eight image paths. Provide exactly one "
                        "of image_paths or image_folder."
                    ),
                },
                "mask_sky": {
                    "type": "boolean",
                    "description": "Whether to enable LingBot-Map sky masking.",
                    "default": False,
                },
                "keyframe_interval": {
                    "type": "integer",
                    "description": "Use every Nth image as a keyframe; use 1 for every frame.",
                    "default": 1,
                },
                "max_frames": {
                    "type": "integer",
                    "description": "Maximum number of frames to send to the backend; must be at least 8.",
                    "default": 128,
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory for preview, trajectory, and point cloud files.",
                },
            },
        }

    def call(
        self,
        image_folder: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        mask_sky: bool = False,
        keyframe_interval: int = 1,
        max_frames: int = 128,
        output_dir: Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> Dict[str, Any]:
        try:
            if not wait_for_completion:
                return ToolResult.fail(
                    "LingBotMapTool requires wait_for_completion=True so it can return reconstruction artifacts.",
                    category=RECONSTRUCTION_3D,
                )
            valid, error, normalized_paths = self._validate_inputs(
                image_folder=image_folder,
                image_paths=image_paths,
                keyframe_interval=keyframe_interval,
                max_frames=max_frames,
            )
            if not valid:
                return ToolResult.fail(error or "Invalid LingBot-Map input.", category=RECONSTRUCTION_3D)

            result = self._client.infer(
                image_folder=None,
                image_paths=normalized_paths,
                mask_sky=bool(mask_sky),
                keyframe_interval=int(keyframe_interval),
                max_frames=int(max_frames),
                output_dir=output_dir,
                wait_for_completion=bool(wait_for_completion),
            )

            if result and result.get("success"):
                point_cloud_path = result.get("point_cloud_path")
                trajectory_path = result.get("trajectory_path")
                points_count = result.get("points_count")
                if not point_cloud_path or not Path(point_cloud_path).is_file():
                    return ToolResult.fail(
                        "LingBot-Map completed without a readable point cloud artifact.",
                        category=RECONSTRUCTION_3D,
                    )
                if not trajectory_path or not Path(trajectory_path).is_file():
                    return ToolResult.fail(
                        "LingBot-Map completed without a readable camera trajectory artifact.",
                        category=RECONSTRUCTION_3D,
                    )
                if isinstance(points_count, bool) or not isinstance(points_count, int) or points_count <= 0:
                    return ToolResult.fail(
                        "LingBot-Map completed without a positive points_count.",
                        category=RECONSTRUCTION_3D,
                    )
                common = {
                    "result": result,
                    "output_dir": result.get("output_dir"),
                    "viewer_url": result.get("viewer_url"),
                    "preview_path": result.get("preview_path"),
                    "trajectory_path": trajectory_path,
                    "point_cloud_path": point_cloud_path,
                    "video_path": result.get("video_path"),
                    "num_frames": result.get("num_frames"),
                    "process_id": result.get("process_id"),
                    "log_path": result.get("log_path"),
                    "command": result.get("command"),
                    "metadata_path": result.get("metadata_path"),
                    "points_count": points_count,
                }
                description = (
                    f"LingBot-Map reconstructed {result.get('num_frames', 0)} frame(s) "
                    f"into {result.get('points_count', 0)} 3D point(s)."
                )
                return ToolResult(
                    success=True,
                    payload=PointCloudPayload(
                        ply_filename=point_cloud_path,
                        points_count=points_count,
                    ),
                    description=description,
                    output_path=result.get("preview_path"),
                    **common,
                )

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return ToolResult.fail(f"LingBot-Map failed: {error_msg}", category=RECONSTRUCTION_3D)
        except Exception as e:
            logger.error("LingBot-Map tool error: %s", e)
            return ToolResult.fail(str(e), category=RECONSTRUCTION_3D)

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
            if int(max_frames) < MIN_REAL_FRAMES:
                return False, f"max_frames must be >= {MIN_REAL_FRAMES} for LingBot-Map.", None
        except (TypeError, ValueError):
            return False, "keyframe_interval and max_frames must be integers.", None

        if image_folder:
            folder = Path(image_folder)
            if not folder.exists() or not folder.is_dir():
                return False, f"Image folder not found: {image_folder}", None
            frames = sorted(
                (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                key=_frame_sort_key,
            )
            if not frames:
                return False, f"No supported image files found in: {image_folder}", None
            sampled_count = len(frames[:: int(keyframe_interval)][: int(max_frames)])
            if sampled_count < MIN_REAL_FRAMES:
                return False, f"LingBot-Map requires at least {MIN_REAL_FRAMES} sampled frames.", None
            return True, None, [str(path) for path in frames]

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
        sampled_count = len(normalized_paths[:: int(keyframe_interval)][: int(max_frames)])
        if sampled_count < MIN_REAL_FRAMES:
            return False, f"LingBot-Map requires at least {MIN_REAL_FRAMES} sampled frames.", None
        return True, None, normalized_paths


def _frame_sort_key(path: Path):
    if path.stem.isdigit():
        return 0, int(path.stem)
    return 1, path.name.lower()
