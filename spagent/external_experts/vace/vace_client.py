import base64
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VaceClient:
    """HTTP client for VACE firstframe generation service."""

    def __init__(self, server_url: str = "http://localhost:20034", timeout_seconds: int = 1800):
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def health_check(self) -> Optional[Dict[str, Any]]:
        try:
            resp = self.session.get(f"{self.server_url}/health", timeout=30)
            if resp.status_code == 200:
                return resp.json()
            logger.error("Health check failed, status=%s, body=%s", resp.status_code, resp.text)
            return None
        except Exception as exc:
            logger.error("Health check request failed: %s", exc)
            return None

    def infer_firstframe(
        self,
        image_path: str,
        prompt: str,
        base: str = "wan",
        task: str = "frameref",
        mode: str = "firstframe",
        timeout_seconds: Optional[int] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "image_path": image_path,
            "prompt": prompt,
            "base": base,
            "task": task,
            "mode": mode,
            "timeout_seconds": timeout_seconds or self.timeout_seconds,
        }
        if extra_args:
            payload["extra_args"] = extra_args

        # Encode the image as base64 so the server doesn't need access to this
        # machine's local filesystem (handles remote server deployments).
        if os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                payload["image_base64"] = base64.b64encode(f.read()).decode("utf-8")
            payload["image_ext"] = os.path.splitext(image_path)[1].lower() or ".jpg"

        try:
            resp = self.session.post(
                f"{self.server_url}/infer",
                json=payload,
                timeout=(timeout_seconds or self.timeout_seconds) + 30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    # The output_path is a server-side path; download the video
                    # so the caller gets a usable local path.
                    remote_path = data.get("output_path")
                    if remote_path:
                        local_path = self._download_output(remote_path)
                        if local_path:
                            data["output_path"] = local_path
                    return data
                logger.error("Infer returned success=false: %s", data)
                return data
            logger.error("Infer failed, status=%s, body=%s", resp.status_code, resp.text)
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as exc:
            logger.error("Infer request failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def _download_output(self, remote_path: str) -> Optional[str]:
        """Download a server-side output file and return its local temp path."""
        try:
            resp = self.session.get(
                f"{self.server_url}/download",
                params={"path": remote_path},
                timeout=120,
                stream=True,
            )
            if resp.status_code != 200:
                logger.warning("Download failed, status=%s for path=%s", resp.status_code, remote_path)
                return None
            suffix = os.path.splitext(remote_path)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp.write(chunk)
                local_path = tmp.name
            logger.info("Downloaded output video to local path: %s", local_path)
            return local_path
        except Exception as exc:
            logger.warning("Failed to download output from server: %s", exc)
            return None


if __name__ == "__main__":
    client = VaceClient()
    health = client.health_check()
    logger.info("Health: %s", health)
