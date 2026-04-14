"""
HTTP client for Molmo2 server.
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

from .point_utils import save_annotated_images

logger = logging.getLogger(__name__)


class Molmo2Client:
    """Client for a Molmo2 inference server."""

    def __init__(self, server_url: str = "http://localhost:20035", output_dir: Union[str, Path, None] = None):
        self.server_url = server_url.rstrip("/")
        self.output_dir = output_dir

    def health_check(self) -> Dict:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Molmo2 health check failed: %s", e)
            return {"success": False, "error": str(e)}

    def infer(
        self,
        image_paths: List[str],
        task: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        save_annotated: bool = True,
    ) -> Dict:
        try:
            payload = {
                "images": [self._encode_image(path) for path in image_paths],
                "image_names": [Path(path).name for path in image_paths],
                "task": task,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "save_annotated": save_annotated,
            }
            response = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300,
            )
            if not response.ok:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = {"success": False, "error": response.text}
                return error_payload
            result = response.json()
            if result.get("success") and save_annotated and result.get("annotated_images"):
                output_paths = save_annotated_images(
                    result["annotated_images"],
                    output_dir=self.output_dir,
                )
                result["output_paths"] = output_paths
                result["output_path"] = output_paths[0] if output_paths else None
            return result
        except Exception as e:
            logger.error("Molmo2 infer request failed: %s", e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _encode_image(image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
