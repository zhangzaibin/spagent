from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class D4RTTool(Tool):
    """
    SPAgent tool wrapper for D4RT-style dynamic 4D reconstruction & tracking.

    This tool is designed to call an external HTTP server.
    The server can be a mock first, then later replaced by a real D4RT inference backend.
    """

    def __init__(
        self,
        use_mock: bool = False,
        server_url: str = "http://127.0.0.1:20034",
        timeout: int = 1800,
    ):
        super().__init__()
        self.name = "d4rt_tool"
        self.description = (
            "Use this tool when the task involves a video or multiple frames and requires "
            "dynamic scene understanding over time, 4D reconstruction, temporal tracking, "
            "camera motion estimation, sparse or dense point tracking, or scene change analysis."
        )
        self.use_mock = use_mock
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Absolute or relative path to an input video file."
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of image frame paths. Use this if video_path is unavailable."
                },
                "query_mode": {
                    "type": "string",
                    "enum": ["reconstruct", "track", "both"],
                    "description": "Select reconstruction, tracking, or both.",
                    "default": "both"
                },
                "num_frames": {
                    "type": "integer",
                    "description": "Maximum number of frames to process.",
                    "default": 32
                },
                "save_visualization": {
                    "type": "boolean",
                    "description": "Whether to save visualization outputs.",
                    "default": True
                },
                "output_dir": {
                    "type": "string",
                    "description": "Output directory for saved artifacts."
                },
                "query_points": {
                    "type": "array",
                    "description": "Optional sparse query points for tracking. Each point is [frame_idx, x, y].",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                }
            },
            "required": []
        }

    def _validate_inputs(
        self,
        video_path: Optional[str],
        image_paths: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        if not video_path and not image_paths:
            return {
                "success": False,
                "error": "Either video_path or image_paths must be provided."
            }

        if video_path:
            vp = Path(video_path)
            if not vp.exists():
                return {
                    "success": False,
                    "error": f"Video file does not exist: {video_path}"
                }

        if image_paths:
            missing = [p for p in image_paths if not Path(p).exists()]
            if missing:
                return {
                    "success": False,
                    "error": f"Some image files do not exist: {missing[:5]}"
                }

        return None

    def call(
        self,
        video_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        query_mode: str = "both",
        num_frames: int = 32,
        save_visualization: bool = True,
        output_dir: Optional[str] = None,
        query_points: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Tool execution entry.
        """

        try:
            validation_error = self._validate_inputs(video_path, image_paths)
            if validation_error is not None:
                return validation_error

            if self.use_mock:
                out_dir = output_dir or "outputs/d4rt_mock"
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                return {
                    "success": True,
                    "summary": (
                        f"[MOCK] D4RT completed. mode={query_mode}, "
                        f"num_frames={num_frames}, visualization={save_visualization}"
                    ),
                    "output_dir": out_dir,
                    "visualization_path": str(Path(out_dir) / "mock_vis.mp4"),
                    "pointcloud_path": str(Path(out_dir) / "mock_scene.ply"),
                    "camera_json": str(Path(out_dir) / "mock_cameras.json"),
                    "raw_result": {
                        "mode": query_mode,
                        "num_frames": num_frames,
                        "query_points": query_points or [],
                    }
                }

            payload = {
                "video_path": video_path,
                "image_paths": image_paths,
                "query_mode": query_mode,
                "num_frames": num_frames,
                "save_visualization": save_visualization,
                "output_dir": output_dir,
                "query_points": query_points,
            }

            resp = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            if not result.get("success", False):
                return {
                    "success": False,
                    "error": result.get("error", "Unknown D4RT server error."),
                    "raw_result": result
                }

            return {
                "success": True,
                "summary": result.get("summary", "D4RT inference finished."),
                "output_dir": result.get("output_dir"),
                "visualization_path": result.get("visualization_path"),
                "pointcloud_path": result.get("pointcloud_path"),
                "camera_json": result.get("camera_json"),
                "tracks_json": result.get("tracks_json"),
                "raw_result": result,
            }

        except requests.exceptions.Timeout:
            logger.exception("D4RT request timed out")
            return {
                "success": False,
                "error": f"D4RT server timeout after {self.timeout} seconds."
            }
        except requests.exceptions.RequestException as e:
            logger.exception("D4RT request failed")
            return {
                "success": False,
                "error": f"D4RT HTTP request failed: {e}"
            }
        except Exception as e:
            logger.exception("D4RTTool call crashed")
            return {
                "success": False,
                "error": f"D4RTTool internal error: {e}"
            }


if __name__ == "__main__":
    tool = D4RTTool(use_mock=True)
    print(json.dumps(
        tool.call(
            video_path=__file__,   # just for quick local test; replace with a real video
            query_mode="both",
            num_frames=8,
        ),
        indent=2,
        ensure_ascii=False
    ))