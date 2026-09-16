import base64
import io
import logging
import os
import uuid
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class WildDet3DClient:
    """HTTP client for the WildDet3D Flask service."""

    def __init__(self, server_url: Optional[str] = None, output_dir: Optional[str] = None):
        self.server_url = (server_url or os.environ.get("WILDDET3D_SERVER_URL", "http://127.0.0.1:20027")).rstrip("/")
        self.output_dir = Path(output_dir) if output_dir else Path(gettempdir()) / "spagent_wilddet3d"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def health_check(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("WildDet3D health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def infer(
        self,
        image_path: str,
        text_prompt: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
        score_threshold: float = 0.3,
        save_visualization: bool = True,
    ) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"Image file not found: {image_path}"}

        payload = {
            "image": self._encode_image(path),
            "filename": path.name,
            "text_prompt": text_prompt,
            "boxes": boxes,
            "points": points,
            "score_threshold": score_threshold,
            "save_visualization": save_visualization,
        }
        try:
            response = requests.post(f"{self.server_url}/infer", json=payload, timeout=900)
            data = response.json()
            if response.status_code >= 400:
                return {"success": False, "error": data.get("error", response.text)}
            if not data.get("success"):
                return data
            return self._save_outputs(data, path.stem)
        except Exception as e:
            logger.error("WildDet3D inference failed: %s", e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _encode_image(path: Path) -> str:
        with Image.open(path) as image:
            image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _save_outputs(self, data: Dict[str, Any], stem: str) -> Dict[str, Any]:
        run_id = uuid.uuid4().hex[:8]
        output_image = data.pop("output_image", None)
        if output_image:
            output_path = self.output_dir / f"{stem}_wilddet3d_{run_id}.png"
            output_path.write_bytes(self._decode_image_output(output_image, "visualization"))
            data["output_path"] = str(output_path)

        depth_image = data.pop("depth_image", None)
        if depth_image:
            depth_path = self.output_dir / f"{stem}_wilddet3d_depth_{run_id}.png"
            depth_path.write_bytes(self._decode_image_output(depth_image, "depth"))
            data["depth_path"] = str(depth_path)

        return data

    @staticmethod
    def _decode_image_output(encoded: str, kind: str) -> bytes:
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            return raw
        except Exception as e:
            raise ValueError(f"WildDet3D returned an invalid {kind} image: {e}") from e
