"""
SoFar HTTP Client

Communicates with the SoFar Flask server on port 20036 (default).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class SoFarClient:
    """HTTP client for the SoFar inference server."""

    def __init__(self, server_url: str = "http://localhost:20036", timeout: float = 60.0):
        self.server_url = server_url
        self.timeout = timeout

    def infer(
        self,
        image_path: str,
        instruction: str,
        camera_intrinsics: Optional[dict] = None,
    ) -> dict:
        """Send an inference request to the SoFar server.

        Args:
            image_path: Path to the RGB scene image.
            instruction: Natural language manipulation instruction.
            camera_intrinsics: Optional dict with fx, fy, cx, cy.

        Returns:
            Dict with pose estimation results.
        """
        payload = {
            "image": base64.b64encode(Path(image_path).read_bytes()).decode(),
            "instruction": instruction,
            "camera_intrinsics": camera_intrinsics or {},
        }

        logger.info(f"Sending infer request to {self.server_url}/infer")
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
