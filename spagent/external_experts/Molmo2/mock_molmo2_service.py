"""Lightweight mock for Molmo2 tool tests without a GPU server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class MockMolmo2:
    def infer_path(self, image_path: str, prompt: str = "Describe this image.", max_new_tokens: int = 200) -> Dict[str, Any]:
        _ = max_new_tokens
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image file not found: {image_path}"}
        return {
            "success": True,
            "text": f"[mock] {prompt[:80]} on {Path(image_path).name}",
            "model": "mock",
        }
