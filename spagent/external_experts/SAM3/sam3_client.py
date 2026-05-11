import base64
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class SAM3Client:
    """HTTP client for the SAM3 image/video segmentation service."""

    def __init__(self, server_url: str = "http://127.0.0.1:20035", output_dir: str = "outputs/sam3_client"):
        self.server_url = server_url.rstrip("/")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def health_check(self) -> Optional[Dict]:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("SAM3 health check failed: %s", e)
            return None

    def test(self) -> Optional[Dict]:
        try:
            response = requests.get(f"{self.server_url}/test", timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("SAM3 test request failed: %s", e)
            return None

    def infer(
        self,
        image_path: str,
        text_prompt: str,
        score_threshold: float = 0.5,
        max_instances: int = 20,
        save_overlay: bool = True,
    ) -> Optional[Dict]:
        try:
            if not os.path.exists(image_path):
                return {"success": False, "error": f"Image file not found: {image_path}"}

            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": f"Unable to read image: {image_path}"}

            ok, buffer = cv2.imencode(".jpg", image)
            if not ok:
                return {"success": False, "error": f"Unable to encode image: {image_path}"}

            payload = {
                "image": base64.b64encode(buffer.tobytes()).decode("utf-8"),
                "text_prompt": text_prompt,
                "score_threshold": float(score_threshold),
                "max_instances": int(max_instances),
                "save_overlay": bool(save_overlay),
            }
            response = requests.post(f"{self.server_url}/infer", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            if not result.get("success"):
                return result

            return self._save_image_outputs(image_path=image_path, image=image, result=result, save_overlay=save_overlay)
        except Exception as e:
            logger.error("SAM3 image inference request failed: %s", e)
            return {"success": False, "error": str(e)}

    def infer_video(
        self,
        video_path: str,
        text_prompt: str,
        frame_index: int = 0,
        score_threshold: float = 0.5,
        max_instances: int = 20,
        save_overlay: bool = True,
    ) -> Optional[Dict]:
        try:
            if not os.path.exists(video_path):
                return {"success": False, "error": f"Video file not found: {video_path}"}

            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "video": video_b64,
                "filename": os.path.basename(video_path),
                "text_prompt": text_prompt,
                "frame_index": int(frame_index),
                "score_threshold": float(score_threshold),
                "max_instances": int(max_instances),
                "save_overlay": bool(save_overlay),
            }
            response = requests.post(f"{self.server_url}/infer_video", json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            if not result.get("success"):
                return result

            if result.get("video"):
                output_path = self._save_video(video_path, result["video"])
                result["output_path"] = output_path
                result["video_path"] = output_path
            return result
        except Exception as e:
            logger.error("SAM3 video inference request failed: %s", e)
            return {"success": False, "error": str(e)}

    def _save_image_outputs(self, image_path: str, image: np.ndarray, result: Dict, save_overlay: bool) -> Dict:
        stem = Path(image_path).stem
        timestamp = int(time.time())
        masks = result.get("masks", [])
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        overlay = image.copy()
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_records: List[Dict] = []

        for idx, mask_info in enumerate(masks):
            mask_array = self._decode_mask(mask_info.get("mask"))
            if mask_array is None:
                continue
            if mask_array.shape[:2] != image.shape[:2]:
                mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            combined_mask = np.maximum(combined_mask, mask_array)
            color = self._color(idx)
            colored = np.zeros_like(image)
            colored[mask_array > 0] = color
            indices = mask_array > 0
            overlay[indices] = cv2.addWeighted(overlay[indices], 0.6, colored[indices], 0.4, 0)

            if idx < len(boxes):
                x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            mask_path = os.path.join(self.output_dir, f"sam3_mask_{stem}_{timestamp}_{idx}.png")
            cv2.imwrite(mask_path, mask_array)
            mask_record = dict(mask_info)
            mask_record.pop("mask", None)
            mask_record["mask_path"] = mask_path
            mask_records.append(mask_record)

        mask_path = os.path.join(self.output_dir, f"sam3_mask_{stem}_{timestamp}.png")
        overlay_path = os.path.join(self.output_dir, f"sam3_overlay_{stem}_{timestamp}.png")
        output_path = os.path.join(self.output_dir, f"sam3_combined_{stem}_{timestamp}.png")

        cv2.imwrite(mask_path, combined_mask)
        if save_overlay:
            cv2.imwrite(overlay_path, overlay)
            combined = np.vstack([image, overlay])
            cv2.imwrite(output_path, combined)
        else:
            overlay_path = None
            output_path = None

        return {
            "success": True,
            "task": "image",
            "text_prompt": result.get("text_prompt"),
            "result": result,
            "output_path": output_path,
            "overlay_path": overlay_path,
            "mask_path": mask_path,
            "shape": result.get("shape", list(image.shape[:2])),
            "masks": mask_records,
            "boxes": boxes,
            "scores": scores,
        }

    def _save_video(self, video_path: str, video_b64: str) -> str:
        stem = Path(video_path).stem
        timestamp = int(time.time())
        output_path = os.path.join(self.output_dir, f"sam3_video_{stem}_{timestamp}.mp4")
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(video_b64))
        return output_path

    def _decode_mask(self, mask_b64: Optional[str]) -> Optional[np.ndarray]:
        if not mask_b64:
            return None
        mask_bytes = base64.b64decode(mask_b64)
        return cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    def _color(self, idx: int):
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        return colors[idx % len(colors)]

