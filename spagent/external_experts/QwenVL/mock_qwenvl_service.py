import os
import logging

logger = logging.getLogger(__name__)


class MockQwenVLService:
    """Mock Qwen VL service for testing without API access."""

    def detect(self, image_path: str, text_prompt: str, task: str = "ref_detection") -> dict:
        logger.info(f"[Mock QwenVL] {task} on {image_path}: {text_prompt[:60]}")

        if not image_path.startswith("http") and not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}

        raw = f"<ref>{text_prompt}</ref><box>(120,200),(450,680)</box>"

        return {
            "success": True,
            "boxes": [[0.12, 0.20, 0.45, 0.68]],
            "labels": [text_prompt],
            "raw_response": raw,
            "mock": True,
        }
