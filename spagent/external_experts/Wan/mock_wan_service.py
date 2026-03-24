"""
Mock Wan video generation service for development and testing.

Returns a fake .mp4 file path without calling any real API.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)


class MockWanService:
    """Lightweight mock that mimics WanClient.generate_video() interface."""

    def generate_video(
        self,
        prompt: str,
        image_path: str = None,
        duration: int = 5,
        size: str = "1280*720",
    ) -> dict:
        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/wan_mock_{int(time.time())}.mp4"

        with open(output_path, "wb") as f:
            f.write(b"\x00" * 1024)

        mode = "image-to-video" if image_path else "text-to-video"
        logger.info(
            f"MockWanService: generated fake video ({mode}, "
            f"duration={duration}s, size={size}) -> {output_path}"
        )

        return {
            "success": True,
            "output_path": output_path,
            "file_size_bytes": 1024,
            "mock": True,
        }
