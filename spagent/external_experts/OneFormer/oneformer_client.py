"""
OneFormer HTTP client.

Communicates with oneformer_server.py. Use this when the model runs as a
separate service and you want to avoid loading weights into the current process.

Usage:
    from external_experts.OneFormer.oneformer_client import OneFormerClient
    client = OneFormerClient("http://localhost:20035")
    result = client.segment("image.jpg", task="panoptic")
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class OneFormerClient:
    """Lightweight HTTP client for the OneFormer server."""

    def __init__(self, server_url: str, timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> Optional[Dict]:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return None

    def segment(
        self,
        image_path: str,
        task: str = "panoptic",
    ) -> Dict[str, Any]:
        """
        Segment an image using the remote OneFormer server.

        Args:
            image_path: Local path to the input image.
            task: One of 'semantic', 'instance', 'panoptic'.

        Returns:
            dict with keys: success, task, segments, num_segments,
                            output_path, description
        """
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": f"Could not read image: {image_path}"}

            _, buffer = cv2.imencode(".jpg", img)
            image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            payload = {"image": image_b64, "task": task}

            resp = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            if not result.get("success"):
                return result

            # Save annotated image locally if server returned one
            output_path = image_path  # fallback
            if result.get("annotated_image"):
                annotated_bytes = base64.b64decode(result["annotated_image"])
                os.makedirs("outputs", exist_ok=True)
                stem = Path(image_path).stem
                output_path = f"outputs/oneformer_{task}_{stem}.png"
                with open(output_path, "wb") as f:
                    f.write(annotated_bytes)
                logger.info("Annotated image saved: %s", output_path)

            return {
                "success": True,
                "task": result["task"],
                "segments": result["segments"],
                "num_segments": result["num_segments"],
                "output_path": output_path,
                "description": result["description"],
            }

        except Exception as e:
            logger.exception("OneFormerClient.segment error")
            return {"success": False, "error": str(e)}
