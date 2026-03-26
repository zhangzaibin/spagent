"""
D4RT HTTP Client

Communicates with the D4RT Flask server on port 20035 (default).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class D4RTClient:
    """HTTP client for the D4RT inference server."""

    def __init__(self, server_url: str = "http://localhost:20035", timeout: float = 120.0):
        self.server_url = server_url
        self.timeout = timeout

    def reconstruct(
        self,
        frame_dir: str,
        task: str = "full_4d",
        query_points: Optional[List[List[int]]] = None,
        output_dir: str = "outputs/d4rt",
        max_frames: int = -1,
    ) -> dict:
        """Send a reconstruction request to the D4RT server.

        Args:
            frame_dir: Path to directory containing video frames.
            task: One of 'depth_and_camera', 'tracking', 'full_4d'.
            query_points: List of [x, y] pixel coordinates to track.
            output_dir: Directory to save output files.
            max_frames: Maximum frames to process (-1 for all).

        Returns:
            Dict with reconstruction results.
        """
        frame_paths = sorted(Path(frame_dir).glob("*.jpg")) + \
                      sorted(Path(frame_dir).glob("*.png"))
        frame_paths = sorted(frame_paths)
        if max_frames > 0:
            frame_paths = frame_paths[:max_frames]

        frames_b64 = [
            base64.b64encode(p.read_bytes()).decode() for p in frame_paths
        ]

        payload = {
            "frames": frames_b64,
            "task": task,
            "query_points": query_points or [],
            "output_dir": output_dir,
        }

        logger.info(f"Sending {task} request ({len(frames_b64)} frames) to {self.server_url}/reconstruct")
        resp = requests.post(
            f"{self.server_url}/reconstruct",
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
