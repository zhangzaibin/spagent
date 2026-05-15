import base64
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class LingBotMapClient:
    """HTTP client for the LingBot-Map Flask service."""

    def __init__(self, server_url: Optional[str] = None, output_dir: Optional[str] = None):
        self.server_url = (server_url or os.environ.get("LINGBOT_MAP_SERVER_URL", "http://127.0.0.1:20038")).rstrip("/")
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "spagent_lingbot_map"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def health_check(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("LingBot-Map health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def infer(
        self,
        image_folder: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        mask_sky: bool = False,
        keyframe_interval: int = 1,
        max_frames: int = 128,
        output_dir: Optional[str] = None,
        wait_for_completion: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mask_sky": bool(mask_sky),
            "keyframe_interval": int(keyframe_interval),
            "max_frames": int(max_frames),
            "wait_for_completion": bool(wait_for_completion),
        }
        if output_dir:
            payload["output_dir"] = output_dir
        if image_folder:
            payload["image_folder"] = image_folder
        if image_paths:
            payload["images"] = [
                {"filename": Path(path).name, "data": self._encode_image(Path(path))}
                for path in image_paths
            ]

        try:
            response = requests.post(f"{self.server_url}/infer", json=payload, timeout=1800)
            data = response.json()
            if response.status_code >= 400:
                return {"success": False, "error": data.get("error", response.text)}
            if not data.get("success"):
                return data
            return self._save_outputs(data, output_dir)
        except Exception as e:
            logger.error("LingBot-Map inference failed: %s", e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _encode_image(path: Path) -> str:
        with Image.open(path) as image:
            image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _save_outputs(self, data: Dict[str, Any], output_dir: Optional[str]) -> Dict[str, Any]:
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for key, filename, path_field in [
            ("preview_image", "lingbot_map_preview.png", "preview_path"),
            ("trajectory_json", "trajectory.json", "trajectory_path"),
            ("point_cloud", "point_cloud.ply", "point_cloud_path"),
            ("video", "lingbot_map_render.mp4", "video_path"),
        ]:
            encoded = data.pop(key, None)
            if encoded:
                path = out_dir / filename
                path.write_bytes(base64.b64decode(encoded))
                data[path_field] = str(path)

        data.setdefault("output_dir", str(out_dir))
        return data
