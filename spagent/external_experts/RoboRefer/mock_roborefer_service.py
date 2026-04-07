from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MockRoboReferService:
    """Mock RoboRefer service for development."""

    def __init__(self, default_answer: str = "[(0.5, 0.5)]"):
        self.default_answer = default_answer

    def query(
        self,
        image_path: str,
        prompt: str,
        enable_depth: int = 1,
        depth_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not Path(image_path).exists():
            return {
                "result": 0,
                "error": f"Image not found: {image_path}",
                "answer": None,
                "mock": True,
            }

        if depth_path is not None and not Path(depth_path).exists():
            return {
                "result": 0,
                "error": f"Depth image not found: {depth_path}",
                "answer": None,
                "mock": True,
            }

        prompt_lower = prompt.lower()
        if "left" in prompt_lower:
            answer = "[(0.25, 0.5)]"
        elif "right" in prompt_lower:
            answer = "[(0.75, 0.5)]"
        elif "top" in prompt_lower or "upper" in prompt_lower:
            answer = "[(0.5, 0.25)]"
        elif "bottom" in prompt_lower or "lower" in prompt_lower:
            answer = "[(0.5, 0.75)]"
        else:
            answer = self.default_answer

        logger.info("Mock RoboRefer query: %s", prompt)

        return {
            "result": 1,
            "answer": answer,
            "mock": True,
            "prompt": prompt,
            "enable_depth": int(enable_depth),
            "depth_path": depth_path,
        }