import base64
import io
import logging
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class DepthAnything3Client:
    """
    HTTP client for Depth Anything V3 server.

    The model runs in a separate process (depth_anything3_server.py).
    Use this client when use_mock=False in DepthAnything3Tool.
    """

    def __init__(self, server_url: str = "http://localhost:20032", request_timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.request_timeout = request_timeout

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Run depth estimation via server.

        Returns:
            dict with "success", "depth" (np.ndarray float32 HxW), or "error"
        """
        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image not found: {image_path}", "depth": None}

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                return {"success": False, "error": f"Failed to read image: {image_path}", "depth": None}

            _, buf = cv2.imencode(".jpg", image_bgr)
            image_b64 = base64.b64encode(buf).decode("ascii")

            resp = requests.post(
                f"{self.server_url}/infer",
                json={"image": image_b64, "input_size": 518},
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("success"):
                return {"success": False, "error": data.get("error", "Server error"), "depth": None}

            depth_b64 = data.get("depth_b64")
            if not depth_b64:
                return {"success": False, "error": "Server did not return depth_b64", "depth": None}

            depth_bytes = base64.b64decode(depth_b64)
            depth = np.load(io.BytesIO(depth_bytes)).astype(np.float32)
            return {"success": True, "depth": depth, "error": None}

        except requests.RequestException as e:
            logger.exception("DepthAnything3Client request failed")
            return {"success": False, "error": str(e), "depth": None}
        except Exception as e:
            logger.exception("DepthAnything3Client predict error")
            return {"success": False, "error": str(e), "depth": None}
