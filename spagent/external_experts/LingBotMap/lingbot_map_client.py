import base64
import io
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class LingBotMapClient:
    """HTTP client for the LingBot-Map Flask service."""

    def __init__(self, server_url: Optional[str] = None, output_dir: Optional[str] = None):
        self.server_url = (server_url or os.environ.get("LINGBOT_MAP_SERVER_URL", "http://127.0.0.1:20040")).rstrip("/")
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
        wait_for_completion: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mask_sky": bool(mask_sky),
            "keyframe_interval": int(keyframe_interval),
            "max_frames": int(max_frames),
            "wait_for_completion": bool(wait_for_completion),
        }
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

        data = dict(data)
        run_id = uuid.uuid4().hex[:12]
        decoded_outputs = []
        for key, filename, path_field in [
            ("preview_image", "lingbot_map_preview.png", "preview_path"),
            ("trajectory_json", "trajectory.json", "trajectory_path"),
            ("point_cloud", "point_cloud.ply", "point_cloud_path"),
            ("video", "lingbot_map_render.mp4", "video_path"),
            ("metadata_json", "reconstruction_metadata.json", "metadata_path"),
            ("log", "lingbot_map.log", "log_path"),
        ]:
            encoded = data.pop(key, None)
            if encoded:
                content = self._decode_and_validate_artifact(key, encoded)
                path = out_dir / f"{run_id}_{filename}"
                decoded_outputs.append((path, content, path_field))

        for path, content, path_field in decoded_outputs:
            path.write_bytes(content)
            data[path_field] = str(path)

        data["output_dir"] = str(out_dir)
        return data

    @staticmethod
    def _decode_and_validate_artifact(key: str, encoded: str) -> bytes:
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid base64 for LingBot-Map artifact '{key}'.") from exc
        if not content:
            raise ValueError(f"LingBot-Map artifact '{key}' is empty.")

        if key == "preview_image":
            try:
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError("LingBot-Map preview_image is not a valid image.") from exc
        elif key in {"trajectory_json", "metadata_json"}:
            try:
                json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"LingBot-Map artifact '{key}' is not valid JSON.") from exc
        elif key == "point_cloud" and not content.lstrip().startswith((b"ply\n", b"ply\r\n", b"# .PCD")):
            raise ValueError("LingBot-Map point_cloud is not a valid PLY or PCD artifact.")
        elif key == "video" and (len(content) < 12 or b"ftyp" not in content[4:12]):
            raise ValueError("LingBot-Map video is not a valid MP4 artifact.")

        return content
