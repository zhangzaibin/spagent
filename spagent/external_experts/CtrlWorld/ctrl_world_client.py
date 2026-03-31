import base64
import logging
import os
import time
from typing import Any, Dict, List

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CtrlWorldClient:
    """HTTP client for Ctrl-World local server."""

    def __init__(self, server_url: str = "http://localhost:20040", timeout: int = 300):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Ctrl-World health check failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_trajectory(
        self,
        image_path: str,
        actions: List[List[float]],
        instruction: str = "",
        task_type: str = "pickplace",
    ) -> Dict[str, Any]:
        try:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                return {
                    "success": False,
                    "error": "Remote image URLs are not supported by Ctrl-World local server.",
                }

            if not os.path.exists(image_path):
                return {"success": False, "error": f"Image not found: {image_path}"}

            image_data = self._encode_image(image_path)
            payload = {
                "image": image_data,
                "actions": actions,
                "instruction": instruction,
                "task_type": task_type,
            }

            logger.info(
                f"Sending Ctrl-World generate request: task={task_type}, "
                f"actions={len(actions)}"
            )
            resp = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )

            if not resp.ok:
                logger.error(f"Ctrl-World server error ({resp.status_code}): {resp.text}")
                return {
                    "success": False,
                    "error": f"Ctrl-World server {resp.status_code}: {resp.text}",
                }

            result = resp.json()
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Unknown error")}

            output_path = result.get("output_path")
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                return {
                    "success": True,
                    "output_path": output_path,
                    "file_size_bytes": file_size,
                    "result": result,
                }

            video_b64 = result.get("video_base64")
            if not video_b64:
                return {"success": False, "error": "No output_path or video_base64 returned by server."}

            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/ctrl_world_{int(time.time())}.mp4"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(video_b64))

            return {
                "success": True,
                "output_path": output_path,
                "file_size_bytes": os.path.getsize(output_path),
                "result": result,
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ctrl-World request failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Ctrl-World client error: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _encode_image(image_path: str) -> Dict[str, str]:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        mime = f"image/{ext}" if ext in ("png", "jpeg", "webp") else "image/jpeg"
        return {"mime_type": mime, "data_base64": image_b64}
