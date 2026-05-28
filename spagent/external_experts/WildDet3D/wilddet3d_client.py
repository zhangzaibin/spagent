"""
WildDet3D HTTP client.

Communicates with wilddet3d_server.py. Use this when the model runs as a
separate service and you want to avoid loading weights into the current process.

Usage:
    from external_experts.WildDet3D.wilddet3d_client import WildDet3DClient
    client = WildDet3DClient("http://localhost:20027")
    result = client.detect("image.jpg", prompt_text="chair")
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class WildDet3DClient:
    """Lightweight HTTP client for the WildDet3D server."""

    def __init__(self, server_url: str, timeout: int = 60):
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

    def detect(
        self,
        image_path: str,
        prompt_text: str = "object",
        input_boxes: Optional[List[float]] = None,
        input_points: Optional[List[List]] = None,
    ) -> Dict[str, Any]:
        """
        Detect objects in an image using the remote WildDet3D server.

        Args:
            image_path: Local path to the input image.
            prompt_text: Text prompt describing the object(s) to detect.
            input_boxes: Optional 2D box prompt [x1, y1, x2, y2].
            input_points: Optional point prompts [[x, y, label], ...].

        Returns:
            dict with keys: success, boxes2d, boxes3d, scores, num_detections,
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

            payload = {"image": image_b64, "prompt_text": prompt_text}
            if input_boxes is not None:
                payload["input_boxes"] = input_boxes
            if input_points is not None:
                payload["input_points"] = input_points

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
                output_path = f"outputs/wilddet3d_{stem}.png"
                with open(output_path, "wb") as f:
                    f.write(annotated_bytes)
                logger.info("Annotated image saved: %s", output_path)

            return {
                "success": True,
                "boxes2d": result["boxes2d"],
                "boxes3d": result["boxes3d"],
                "scores": result["scores"],
                "num_detections": result["num_detections"],
                "output_path": output_path,
                "description": result["description"],
            }

        except Exception as e:
            logger.exception("WildDet3DClient.detect error")
            return {"success": False, "error": str(e)}
