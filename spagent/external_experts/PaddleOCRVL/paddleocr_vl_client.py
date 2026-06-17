"""
PaddleOCR-VL-1.5 HTTP client.

Sends inference requests to the PaddleOCR-VL Flask server on port 20037.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Dict, Any

import requests

logger = logging.getLogger(__name__)


class PaddleOCRVLClient:
    """HTTP client for the PaddleOCR-VL-1.5 inference server."""

    def __init__(self, server_url: str = "http://localhost:20037", timeout: float = 120.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def recognize(self, image_path: str, task: str = "ocr") -> Dict[str, Any]:
        """Send a recognition request to the server.

        Args:
            image_path: Path to the input image.
            task: One of 'ocr', 'table', 'chart', 'formula', 'spotting', 'seal'.

        Returns:
            Dict with keys: success, text, task (or error on failure).
        """
        payload = {
            "image": self._encode_image(image_path),
            "task": task,
        }
        logger.info("Sending PaddleOCR-VL request to %s/infer (task=%s)", self.server_url, task)
        resp = requests.post(
            f"{self.server_url}/infer",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _encode_image(image_path: str) -> str:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")
