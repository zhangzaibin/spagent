from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


def encode_file_to_base64(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class RoboReferClient:
    """HTTP client for RoboRefer API."""

    def __init__(self, server_url: str = "http://127.0.0.1:25547", timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def query(
        self,
        image_path: str,
        prompt: str,
        enable_depth: int = 1,
        depth_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a query to RoboRefer API.
        """
        image_b64 = encode_file_to_base64(image_path)

        payload: Dict[str, Any] = {
            "image_url": [image_b64],
            "depth_url": [],
            "enable_depth": int(enable_depth),
            "text": prompt,
        }

        if depth_path:
            payload["depth_url"] = [encode_file_to_base64(depth_path)]

        logger.info(
            "Sending RoboRefer request to %s/query (enable_depth=%s, has_depth=%s)",
            self.server_url,
            enable_depth,
            bool(depth_path),
        )

        response = requests.post(
            f"{self.server_url}/query",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected response type from RoboRefer: {type(data)}")

        return data