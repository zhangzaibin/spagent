import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from PIL import Image

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

logger = logging.getLogger(__name__)


class Molmo2Client:
    """HTTP client for the Molmo2 Flask server."""

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = (server_url or os.environ.get("MOLMO2_SERVER_URL", "http://localhost:20025")).rstrip(
            "/"
        )

    def _encode_image(self, image: Union[str, "np.ndarray", Image.Image]) -> str:
        if isinstance(image, str):
            with open(image, "rb") as f:
                image_bytes = f.read()
        elif np is not None and isinstance(image, np.ndarray):
            try:
                import cv2

                _, buffer = cv2.imencode(".jpg", image)
                image_bytes = buffer.tobytes()
            except ImportError:
                arr = image[..., ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="JPEG")
                image_bytes = buf.getvalue()
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        else:
            raise ValueError("Unsupported image type")
        return base64.b64encode(image_bytes).decode("utf-8")

    def health_check(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Molmo2 health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def infer(
        self,
        image: Union[str, np.ndarray, Image.Image],
        prompt: str = "Describe this image.",
        max_new_tokens: int = 200,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "image": self._encode_image(image),
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        try:
            response = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                timeout=600,
            )
            out = response.json()
            if response.status_code >= 400:
                return {"success": False, "error": out.get("error", response.text)}
            return out
        except Exception as e:
            logger.error("Molmo2 infer failed: %s", e)
            return {"success": False, "error": str(e)}

    def infer_path(self, image_path: str, prompt: str = "Describe this image.", max_new_tokens: int = 200) -> Dict[str, Any]:
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image file not found: {image_path}"}
        return self.infer(image_path, prompt=prompt, max_new_tokens=max_new_tokens)
