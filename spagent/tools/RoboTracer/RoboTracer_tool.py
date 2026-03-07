import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Union

sys.path.append(str(Path(__file__).parent.parent))
from core.tool import Tool

logger = logging.getLogger(__name__)


class RoboTracerTool(Tool):
    """
    Tool for tracing approximate robot/camera trajectory from one or more frames.
    """

    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:20040"):
        super().__init__(
            name="robotracer_tool",
            description=(
                "Estimate an approximate robot or camera trajectory from one or more "
                "ordered image frames. Use this tool when you need motion path, "
                "waypoints, movement direction, or a short trajectory summary."
            )
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            try:
                from external_experts.RoboTracer.mock_robotracer_service import MockRoboTracerService
                self._client = MockRoboTracerService()
                logger.info("Using mock RoboTracer service")
            except ImportError as e:
                logger.error(f"Failed to import mock RoboTracer service: {e}")
                raise
        else:
            try:
                from external_experts.RoboTracer.robotracer_client import RoboTracerClient
                self._client = RoboTracerClient(server_url=self.server_url)
                logger.info(f"Using real RoboTracer service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import RoboTracer client: {e}")
                raise

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Ordered list of image/frame paths. The order should reflect time order "
                        "(earliest to latest)."
                    )
                },
                "coordinate_mode": {
                    "type": "string",
                    "enum": ["relative_2d", "normalized_2d"],
                    "description": (
                        "Trajectory coordinate convention. "
                        "'relative_2d' returns relative motion coordinates. "
                        "'normalized_2d' returns coordinates normalized to [0,1]."
                    ),
                    "default": "relative_2d"
                },
                "return_summary_only": {
                    "type": "boolean",
                    "description": "If true, return a compact result without detailed waypoint metadata.",
                    "default": False
                }
            },
            "required": ["image_paths"]
        }

    def call(
        self,
        image_paths: List[str],
        coordinate_mode: str = "relative_2d",
        return_summary_only: bool = False
    ) -> Dict[str, Any]:
        try:
            if not image_paths or not isinstance(image_paths, list):
                return {
                    "success": False,
                    "error": "image_paths must be a non-empty list of image file paths."
                }

            missing = [p for p in image_paths if not Path(p).exists()]
            if missing:
                return {
                    "success": False,
                    "error": f"Some image files were not found: {missing}"
                }

            if hasattr(self._client, "infer"):
                result = self._client.infer(
                    image_paths=image_paths,
                    coordinate_mode=coordinate_mode,
                    return_summary_only=return_summary_only
                )
            elif hasattr(self._client, "trace"):
                result = self._client.trace(
                    image_paths=image_paths,
                    coordinate_mode=coordinate_mode,
                    return_summary_only=return_summary_only
                )
            else:
                return {
                    "success": False,
                    "error": "RoboTracer client does not implement infer() or trace()."
                }

            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "trajectory_points": result.get("trajectory_points", []),
                    "total_distance": result.get("total_distance", 0.0),
                    "num_frames": result.get("num_frames", len(image_paths)),
                    "summary": result.get("summary", "Trajectory estimation completed."),
                    "output_path": result.get("output_path")
                }

            return {
                "success": False,
                "error": result.get("error", "Unknown RoboTracer error") if result else "No result returned"
            }

        except Exception as e:
            logger.error(f"RoboTracerTool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
