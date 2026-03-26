"""
Orient Anything V2 HTTP Client

Communicates with the Orient Anything V2 Flask server on port 20034 (default).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OrientAnythingV2Client:
    """HTTP client for the Orient Anything V2 inference server."""

    def __init__(self, server_url: str = "http://localhost:20034", timeout: float = 60.0):
        self.server_url = server_url
        self.timeout = timeout

    def infer(
        self,
        image_path: str,
        object_category: str,
        task: str = "orientation",
        image_path2: Optional[str] = None,
    ) -> dict:
        """Send an inference request to the Orient Anything V2 server.

        Args:
            image_path: Path to the input image.
            object_category: Semantic category of the target object.
            task: One of 'orientation', 'symmetry', 'relative_rotation'.
            image_path2: Path to second image (for relative_rotation).

        Returns:
            Dict with task-specific result keys.
        """
        payload = {
            "image": self._encode_image(image_path),
            "object_category": object_category,
            "task": task,
        }
        if image_path2 is not None:
            payload["image2"] = self._encode_image(image_path2)

        logger.info(f"Sending {task} request to {self.server_url}/infer")
        resp = requests.post(
            f"{self.server_url}/infer",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        """Return True if the server is alive."""
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def test_infer(self) -> dict:
        """Quick test via the /test endpoint."""
        try:
            r = requests.get(f"{self.server_url}/test", timeout=10.0)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read image file and return base64-encoded string."""
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")
