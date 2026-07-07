"""
Wan Video Generation Tool

Wraps Alibaba Wan video generation (via DashScope) for the SPAgent system.
Supports text-to-video and image-to-video generation.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import MediaPayload, ToolResult, VIDEO_GENERATION

logger = logging.getLogger(__name__)


class WanTool(Tool):
    """Tool for video generation using Alibaba Wan (DashScope)."""

    def __init__(self, use_mock: bool = True, api_key: str = None, model: str = "wanx2.1-t2v-turbo"):
        super().__init__(
            name="video_generation_wan_tool",
            description=(
                "Generate a video from a text prompt (and optionally a reference image) "
                "using Alibaba Wan. Returns the path to the generated .mp4 video file. "
                "Use this when the task requires creating a video visualization, animation, "
                "or video content from a description. Supports text-to-video and "
                "image-to-video generation."
            ),
        )
        self.use_mock = use_mock
        self.api_key = api_key
        self.model_name = model
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
                from external_experts.Wan.wan_client import WanClient
                self._client = WanClient(api_key=self.api_key, model=self.model_name)
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
                    "description": (
                        "Optional path or URL to a reference image for image-to-video generation. "
                        "When provided, the model generates a video conditioned on this image."
                    ),
                },
                "duration": {
                    "type": "integer",
                    "description": "Video duration in seconds (3–10). Default is 5.",
                    "default": 5,
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio: '16:9', '9:16', or '1:1'. Default is '16:9'.",
                    "enum": ["16:9", "9:16", "1:1"],
                    "default": "16:9",
                },
            },
            "required": ["prompt"],
        }

    def call(
        self,
        prompt: str,
        image_path: str = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Generating video with Wan: {prompt[:80]}...")

            if image_path and not image_path.startswith("http"):
                from pathlib import Path as _Path
                if not _Path(image_path).exists():
                    return {"success": False, "error": f"Image file not found: {image_path}"}

            result = self._client.generate_video(
                prompt=prompt,
                image_path=image_path,
                duration=duration,
                aspect_ratio=aspect_ratio,
            )

            if result and result.get("success"):
                output_path = result.get("output_path")
                logger.info(f"Wan video generated: {output_path}")
                # Standardized output: MediaPayload carries the .mp4 path;
                # the legacy `result` key is preserved as an extra so
                # existing consumers see the same dict shape.
                payload = MediaPayload(
                    category=VIDEO_GENERATION,
                    output_path=output_path,
                    metadata={"duration": result.get("duration", duration)},
                )
                return ToolResult(
                    success=True,
                    payload=payload,
                    description=f"Wan video generated at {output_path}.",
                    output_path=output_path,
                    result=result,
                )
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"Wan generation failed: {error_msg}")
                return {"success": False, "error": f"Wan generation failed: {error_msg}"}

        except Exception as e:
            logger.error(f"Wan tool error: {e}")
            return {"success": False, "error": str(e)}
