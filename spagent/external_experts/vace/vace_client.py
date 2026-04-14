import logging
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

        try:
            resp = self.session.post(
                f"{self.server_url}/infer",
                json=payload,
                timeout=(timeout_seconds or self.timeout_seconds) + 30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data
                logger.error("Infer returned success=false: %s", data)
                return data
            logger.error("Infer failed, status=%s, body=%s", resp.status_code, resp.text)
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as exc:
            logger.error("Infer request failed: %s", exc)
            return {"success": False, "error": str(exc)}


if __name__ == "__main__":
    client = VaceClient()
    health = client.health_check()
    logger.info("Health: %s", health)
