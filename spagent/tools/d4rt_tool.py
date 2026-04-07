"""
D4RT Tool

This module contains the D4RTTool that wraps D4RT (Dynamic 4D Reconstruction
and Tracking) functionality for the SPAgent system.

D4RT (Google DeepMind) jointly estimates depth, spatio-temporal correspondence,
and camera parameters from monocular video - up to 300x faster than prior methods.

Paper: https://d4rt-paper.github.io/
Blog:  https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class D4RTTool(Tool):
    """Tool for dynamic 4D scene reconstruction from video using D4RT."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://localhost:20035",
    ):
        super().__init__(
            name="d4rt_tool",
            description=(
                "Reconstructs dynamic 4D scenes (geometry + motion) from video using D4RT "
                "(Google DeepMind). Input: a directory of ordered video frame images. "
                "Modes: (1) 'depth_and_camera' - returns per-frame depth maps and camera poses; "
                "(2) 'tracking' - tracks specified 2D query points across all frames and returns "
                "3D trajectories; (3) 'full_4d' - returns all outputs simultaneously. "
                "Best suited for dynamic scenes with moving objects or camera."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the client (mock or real)."""
        if self.use_mock:
            from external_experts.D4RT.mock_d4rt_service import MockD4RTService
            self._client = MockD4RTService()
            logger.info("D4RTTool initialized with mock service")
        else:
            from external_experts.D4RT.d4rt_client import D4RTClient
            self._client = D4RTClient(server_url=self.server_url)
            logger.info(f"D4RTTool initialized with server at {self.server_url}")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "frame_dir": {
                    "type": "string",
                    "description": (
                        "Path to a directory containing ordered video frame images "
                        "(e.g. frame_0001.jpg, frame_0002.jpg). Frames are sorted by filename."
                    ),
                },
                "task": {
                    "type": "string",
                    "enum": ["depth_and_camera", "tracking", "full_4d"],
                    "description": (
                        "'depth_and_camera': estimate per-frame depth and camera poses. "
                        "'tracking': estimate 3D motion trajectories for query points. "
                        "'full_4d': run all tasks in one pass."
                    ),
                    "default": "full_4d",
                },
                "query_points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "description": (
                        "List of [x, y] pixel coordinates in the first frame to track. "
                        "Required only for task='tracking'. Example: [[100, 200], [300, 400]]."
                    ),
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory to save output files. Defaults to outputs/d4rt/.",
                    "default": "outputs/d4rt",
                },
                "max_frames": {
                    "type": "integer",
                    "description": "Maximum number of frames to process. -1 for all frames.",
                    "default": -1,
                },
            },
            "required": ["frame_dir"],
        }

    def call(
        self,
        frame_dir: str,
        task: str = "full_4d",
        query_points: Optional[List[List[int]]] = None,
        output_dir: str = "outputs/d4rt",
        max_frames: int = -1,
    ) -> Dict[str, Any]:
        """Execute D4RT 4D reconstruction.

        Args:
            frame_dir: Path to directory containing video frames.
            task: One of 'depth_and_camera', 'tracking', 'full_4d'.
            query_points: List of [x, y] pixel coordinates to track.
            output_dir: Directory to save output files.
            max_frames: Maximum frames to process (-1 for all).

        Returns:
            Dict with 'success', 'result', and optionally 'error' keys.
        """
        try:
            if not Path(frame_dir).is_dir():
                return {"success": False, "error": f"Frame directory not found: {frame_dir}"}

            if task == "tracking" and not query_points:
                return {
                    "success": False,
                    "error": "task='tracking' requires 'query_points'.",
                }

            logger.info(f"Running D4RT: task={task}, frame_dir={frame_dir}")

            result = self._client.reconstruct(
                frame_dir=frame_dir,
                task=task,
                query_points=query_points,
                output_dir=output_dir,
                max_frames=max_frames,
            )

            return {
                "success": True,
                "result": result,
                "output_dir": result.get("output_dir", output_dir),
            }

        except Exception as e:
            logger.error(f"D4RTTool error: {e}")
            return {"success": False, "error": str(e)}
