"""
InfiniDepth Tool

Wraps InfiniDepth relative depth estimation for SPAgent.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import DEPTH, DepthPayload, ToolResult

logger = logging.getLogger(__name__)


class InfiniDepthTool(Tool):
    """Tool for high-resolution monocular depth estimation using InfiniDepth."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://127.0.0.1:20039",
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="infinidepth_tool",
            description=(
                "Estimate high-resolution relative depth from a single RGB image using InfiniDepth. "
                "Can optionally export a point cloud when the backend supports it."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.output_dir = output_dir
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.use_mock:
            from external_experts.InfiniDepth.mock_infinidepth_service import MockInfiniDepthService

            self._client = MockInfiniDepthService(output_dir=self.output_dir)
            logger.info("Using mock InfiniDepth service")
        else:
            from external_experts.InfiniDepth.infinidepth_client import InfiniDepthClient

            self._client = InfiniDepthClient(server_url=self.server_url, output_dir=self.output_dir)
            logger.info("Using real InfiniDepth service at %s", self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to the input RGB image."},
                "task": {
                    "type": "string",
                    "enum": ["depth"],
                    "description": "InfiniDepth task. v1 supports single-image relative depth.",
                    "default": "depth",
                },
                "save_pcd": {
                    "type": "boolean",
                    "description": "Whether to export a point cloud when supported by the backend.",
                    "default": False,
                },
                "upsample_ratio": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4,
                    "description": "Depth output upsample ratio passed to the backend.",
                    "default": 2,
                },
                "output_resolution_mode": {
                    "type": "string",
                    "enum": ["original", "upsample"],
                    "description": "Return depth at the source resolution or at the model input resolution multiplied by upsample_ratio.",
                    "default": "original",
                },
                "output_dir": {"type": "string", "description": "Optional output directory."},
            },
            "required": ["image_path"],
        }

    def call(
        self,
        image_path: str,
        task: str = "depth",
        save_pcd: bool = False,
        upsample_ratio: int = 2,
        output_resolution_mode: str = "original",
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            path = Path(image_path)
            if not path.is_file():
                return ToolResult.fail(f"Image file not found: {image_path}", category=DEPTH)
            if task != "depth":
                return ToolResult.fail("InfiniDepthTool v1 only supports task='depth'.", category=DEPTH)
            upsample_value = float(upsample_ratio)
            if not 1 <= upsample_value <= 4 or not upsample_value.is_integer():
                return ToolResult.fail(
                    "upsample_ratio must be an integer in [1, 4].",
                    category=DEPTH,
                )
            upsample_ratio = int(upsample_value)
            if output_resolution_mode not in {"original", "upsample"}:
                return ToolResult.fail(
                    "output_resolution_mode must be 'original' or 'upsample'.",
                    category=DEPTH,
                )

            result = self._client.infer(
                image_path=str(path),
                save_pcd=bool(save_pcd),
                upsample_ratio=upsample_ratio,
                output_resolution_mode=output_resolution_mode,
                output_dir=output_dir,
            )

            if result and result.get("success"):
                depth_path = result.get("depth_path")
                if not depth_path:
                    return ToolResult.fail(
                        "InfiniDepth completed without a depth output.",
                        category=DEPTH,
                    )
                if not Path(depth_path).is_file():
                    return ToolResult.fail(
                        f"InfiniDepth depth output does not exist: {depth_path}",
                        category=DEPTH,
                    )
                point_cloud_path = result.get("point_cloud_path")
                if save_pcd and not (point_cloud_path and Path(point_cloud_path).is_file()):
                    return ToolResult.fail(
                        "InfiniDepth completed without the requested point cloud output.",
                        category=DEPTH,
                    )
                return ToolResult(
                    success=True,
                    payload=DepthPayload(depth_path=depth_path, shape=result.get("shape")),
                    description="InfiniDepth estimated relative depth for the input image.",
                    output_path=result.get("colored_depth_path") or depth_path,
                    result=result,
                    colored_depth_path=result.get("colored_depth_path"),
                    point_cloud_path=point_cloud_path,
                    depth_shape=result.get("depth_shape"),
                    source_shape=result.get("source_shape"),
                    output_resolution_mode=result.get("output_resolution_mode"),
                    output_dir=result.get("output_dir"),
                )

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return ToolResult.fail(f"InfiniDepth failed: {error_msg}", category=DEPTH)
        except Exception as e:
            logger.error("InfiniDepth tool error: %s", e)
            return ToolResult.fail(str(e), category=DEPTH)
