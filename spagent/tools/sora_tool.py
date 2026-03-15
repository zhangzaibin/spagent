"""
Sora Video Generation Tool

Wraps OpenAI Sora video generation for the SPAgent system.
Supports text-to-video and image-to-video generation.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class SoraTool(Tool):
    """Tool for video generation using OpenAI Sora."""

    def __init__(self, use_mock: bool = True, api_key: str = None, model: str = "sora-2"):
        super().__init__(
            name="video_generation_sora_tool",
            description=(
                "Generate a video from a text prompt (and optionally a reference image) "
                "using OpenAI Sora. Returns the path to the generated .mp4 video file. "
                "Use this when the task requires creating a video visualization, animation, "
                "or video content from a description."
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
                from external_experts.Sora.mock_sora_service import MockSoraService
                self._client = MockSoraService()
                logger.info("Using mock Sora service")
            except ImportError as e:
                logger.error(f"Failed to import mock Sora service: {e}")
                raise
        else:
            try:
                from external_experts.Sora.sora_client import SoraClient
                self._client = SoraClient(api_key=self.api_key, model=self.model_name)
                logger.info("Using real Sora service (OpenAI API)")
            except ImportError as e:
                logger.error(f"Failed to import Sora client: {e}")
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
                    "description": "Video duration in seconds (5-20). Default is 10.",
                    "default": 10,
                },
                "resolution": {
                    "type": "string",
                    "description": "Video resolution: '480p', '720p', or '1080p'. Default is '1080p'.",
                    "enum": ["480p", "720p", "1080p"],
                    "default": "1080p",
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
        duration: int = 10,
        resolution: str = "1080p",
        aspect_ratio: str = "16:9",
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Generating video with Sora: {prompt[:80]}...")

            if image_path and not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            result = self._client.generate_video(
                prompt=prompt,
                image_path=image_path,
                duration=duration,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
            )

            if result and result.get("success"):
                logger.info(f"Sora video generated: {result.get('output_path')}")
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                }
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"Sora generation failed: {error_msg}")
                return {"success": False, "error": f"Sora generation failed: {error_msg}"}

        except Exception as e:
            logger.error(f"Sora tool error: {e}")
            return {"success": False, "error": str(e)}
