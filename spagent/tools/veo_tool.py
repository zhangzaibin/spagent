"""
Veo Video Generation Tool

Wraps Google Veo video generation for the SPAgent system.
Supports text-to-video and image-to-video generation.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class VeoTool(Tool):
    """Tool for video generation using Google Veo via the Gemini API."""

    def __init__(self, use_mock: bool = True, api_key: str = None):
        super().__init__(
            name="video_generation_veo_tool",
            description=(
                "Generate a video from a text prompt (and optionally a reference image) "
                "using Google Veo. Returns the path to the generated .mp4 video file. "
                "Use this when the task requires creating a video visualization, animation, "
                "or video content from a description."
            ),
        )
        self.use_mock = use_mock
        self.api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            try:
                from external_experts.Veo.mock_veo_service import MockVeoService
                self._client = MockVeoService()
                logger.info("Using mock Veo service")
            except ImportError as e:
                logger.error(f"Failed to import mock Veo service: {e}")
                raise
        else:
            try:
                from external_experts.Veo.veo_client import VeoClient
                self._client = VeoClient(api_key=self.api_key)
                logger.info("Using real Veo service (Gemini API)")
            except ImportError as e:
                logger.error(f"Failed to import Veo client: {e}")
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
                    "description": "Optional path to a reference image for image-to-video generation.",
                },
                "duration": {
                    "type": "integer",
                    "description": "Video duration in seconds (5 or 8). Default is 8.",
                    "default": 8,
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio: '16:9' (landscape) or '9:16' (portrait). Default is '16:9'.",
                    "enum": ["16:9", "9:16"],
                    "default": "16:9",
                },
            },
            "required": ["prompt"],
        }

    def call(self, prompt: str, image_path: str = None, duration: int = 8, aspect_ratio: str = "16:9") -> Dict[str, Any]:
        try:
            logger.info(f"Generating video with Veo: {prompt[:80]}...")

            if image_path and not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            result = self._client.generate_video(
                prompt=prompt,
                image_path=image_path,
                duration=duration,
                aspect_ratio=aspect_ratio,
            )

            if result and result.get("success"):
                logger.info(f"Veo video generated: {result.get('output_path')}")
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                }
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"Veo generation failed: {error_msg}")
                return {"success": False, "error": f"Veo generation failed: {error_msg}"}

        except Exception as e:
            logger.error(f"Veo tool error: {e}")
            return {"success": False, "error": str(e)}
