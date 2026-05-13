import base64
import io
import logging
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class InfiniDepthClient:
    """HTTP client for the InfiniDepth Flask service."""

    def __init__(self, server_url: Optional[str] = None, output_dir: Optional[str] = None):
        self.server_url = (server_url or os.environ.get("INFINIDEPTH_SERVER_URL", "http://127.0.0.1:20037")).rstrip("/")
        self.output_dir = Path(output_dir) if output_dir else Path(gettempdir()) / "spagent_infinidepth"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def health_check(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("InfiniDepth health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def infer(
        self,
        image_path: str,
        save_pcd: bool = False,
        upsample_ratio: float = 2,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"Image file not found: {image_path}"}
        payload = {
            "image": self._encode_image(path),
            "filename": path.name,
            "save_pcd": save_pcd,
            "upsample_ratio": upsample_ratio,
        }
        try:
            response = requests.post(f"{self.server_url}/infer", json=payload, timeout=1800)
            data = response.json()
            if response.status_code >= 400:
                return {"success": False, "error": data.get("error", response.text)}
            if not data.get("success"):
                return data
            return self._save_outputs(data, path.stem, output_dir)
        except Exception as e:
            logger.error("InfiniDepth inference failed: %s", e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _encode_image(path: Path) -> str:
        with Image.open(path) as image:
            image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _save_outputs(self, data: Dict[str, Any], stem: str, output_dir: Optional[str]) -> Dict[str, Any]:
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for key, suffix, field in [
            ("depth_image", "depth.png", "depth_path"),
            ("colored_depth_image", "colored.png", "colored_depth_path"),
            ("point_cloud", "points.ply", "point_cloud_path"),
        ]:
            encoded = data.pop(key, None)
            if encoded:
                path = out_dir / f"{stem}_infinidepth_{suffix}"
                path.write_bytes(base64.b64decode(encoded))
                data[field] = str(path)

        data["output_dir"] = str(out_dir)
        return data
