"""
Wan Video Generation Tool

Wraps Alibaba Wan (万相) video generation for the SPAgent system.
Supports text-to-video and image-to-video generation via DashScope API.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class WanTool(Tool):
    """Tool for video generation using Alibaba Wan via the DashScope API."""

    def __init__(self, use_mock: bool = True, api_key: str = None):
        super().__init__(
            name="video_generation_wan_tool",
            description=(
                "Generate a video from a text prompt (and optionally a reference image) "
                "using Alibaba Wan (万相). Returns the path to the generated .mp4 video file. "
                "Use this when the task requires creating a video visualization, animation, "
                "or video content from a description. Supports high-quality 720P/1080P output "
                "with durations from 2 to 15 seconds."
            ),
        )
        self.use_mock = use_mock
        self.api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            try:
                from external_experts.Wan.mock_wan_service import MockWanService
                self._client = MockWanService()
                logger.info("Using mock Wan service")
            except ImportError as e:
                logger.error(f"Failed to import mock Wan service: {e}")
                raise
        else:
            try:
                from external_experts.Wan.mock_wan_client import WanClient
                self._client = WanClient(api_key=self.api_key)
                logger.info("Using real Wan service (DashScope API)")
            except ImportError as e:
                logger.error(f"Failed to import Wan client: {e}")
                raise

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the video to generate.",
                },
                "image_path": {
                    "type": "string",
                    "description": "Optional path to a reference image for image-to-video generation (first frame).",
                },
                "duration": {
                    "type": "integer",
                    "description": "Video duration in seconds (2-15). Default is 5.",
                    "default": 5,
                },
                "size": {
                    "type": "string",
                    "description": (
                        "Resolution in 'width*height' format. "
                        "Options: '1280*720' (720P 16:9), '720*1280' (720P 9:16), "
                        "'1920*1080' (1080P 16:9), '1080*1920' (1080P 9:16). "
                        "Default is '1280*720'."
                    ),
                    "enum": [
                        "1280*720", "720*1280", "960*960",
                        "1920*1080", "1080*1920", "1440*1440",
                    ],
                    "default": "1280*720",
                },
            },
            "required": ["prompt"],
        }

    def call(
        self,
        prompt: str,
        image_path: str = None,
        duration: int = 5,
        size: str = "1280*720",
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Generating video with Wan: {prompt[:80]}...")

            if image_path and not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            result = self._client.generate_video(
                prompt=prompt,
                image_path=image_path,
                duration=duration,
                size=size,
            )

            if result and result.get("success"):
                logger.info(f"Wan video generated: {result.get('output_path')}")
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                }
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"Wan generation failed: {error_msg}")
                return {"success": False, "error": f"Wan generation failed: {error_msg}"}

        except Exception as e:
            logger.error(f"Wan tool error: {e}")
            return {"success": False, "error": str(e)}
