import os
import time
import logging

logger = logging.getLogger(__name__)


class MockWanService:
    """Mock Wan service for testing without API access."""

    def generate_video(
        self,
        prompt: str,
        image_path: str = None,
        duration: int = 5,
        size: str = "1280*720",
    ) -> dict:
        logger.info(f"[Mock Wan] Generating video for prompt: {prompt[:80]}...")
        if image_path and not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}

        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/mock_wan_{int(time.time())}.mp4"
        with open(output_path, "wb") as f:
            f.write(b"\x00" * 1024)

        return {
            "success": True,
            "output_path": output_path,
            "file_size_bytes": 1024,
            "mock": True,
        }
