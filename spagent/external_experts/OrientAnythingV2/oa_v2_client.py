"""
Orient Anything V2 HTTP Client

Communicates with the Orient Anything V2 Flask server on port 20034 (default).

Output fields (all integer degrees):
  azimuth        — 0-360°      absolute azimuth of the reference object
  elevation      — -90..90°    absolute elevation
  rotation       — -180..180°  in-plane rotation
  symmetry_alpha — 0/1/2/4     rotational symmetry order
                   (0=none/uncertain, 1=bilateral, 2=2-fold, 4=4-fold)

When a second image is provided the response additionally contains:
  rel_azimuth    — relative azimuth  of target w.r.t. reference
  rel_elevation  — relative elevation
  rel_rotation   — relative rotation
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
        image_path2: Optional[str] = None,
        remove_background: bool = False,
        object_category: str = "object",   # ignored by V2 server, kept for compat
        task: str = "orientation",          # ignored by V2 server, kept for compat
    ) -> dict:
        """Send an inference request to the Orient Anything V2 server.

        The V2 server determines single- vs two-image inference from whether
        ``image_path2`` is supplied; ``task`` and ``object_category`` are
        accepted for API compatibility but are not forwarded.

        Args:
            image_path: Path to the reference image.
            image_path2: Path to a second image (enables relative rotation output).
            remove_background: Ask the server to remove background before inference.
            object_category: Ignored — the V2 model is category-agnostic.
            task: Ignored — kept so callers that pass task= still work.

        Returns:
            Dict with keys: azimuth, elevation, rotation, symmetry_alpha,
            and optionally rel_azimuth, rel_elevation, rel_rotation.
        """
        payload: dict = {
            "image": self._encode_image(image_path),
            "remove_background": remove_background,
        }
        if image_path2 is not None:
            payload["image2"] = self._encode_image(image_path2)

        logger.info("Sending infer request to %s/infer", self.server_url)
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
        """Quick connectivity test via the /test endpoint."""
        try:
            r = requests.get(f"{self.server_url}/test", timeout=10.0)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _encode_image(image_path: str) -> str:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")
