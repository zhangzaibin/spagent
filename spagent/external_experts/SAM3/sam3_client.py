import base64
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class SAM3Client:
    """HTTP client for the SAM3 image/video segmentation service."""

    def __init__(self, server_url: str = "http://127.0.0.1:20035", output_dir: Optional[str] = None):
        self.server_url = server_url.rstrip("/")
        self.output_dir = output_dir or str(Path(tempfile.gettempdir()) / "spagent_sam3_client")
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
            source = Path(video_path)
            if not source.exists():
                return {"success": False, "error": f"Video file not found: {video_path}"}

            payload = {
                "text_prompt": text_prompt,
                "frame_index": int(frame_index),
                "score_threshold": float(score_threshold),
                "max_instances": int(max_instances),
                "save_overlay": bool(save_overlay),
            }
            if source.is_dir():
                frames = self._encode_frame_directory(source)
                if not frames:
                    return {
                        "success": False,
                        "error": f"No readable JPEG frames found in directory: {video_path}",
                    }
                payload.update({"frames": frames, "filename": source.name})
            elif source.is_file():
                with source.open("rb") as f:
                    payload["video"] = base64.b64encode(f.read()).decode("utf-8")
                payload["filename"] = source.name
            else:
                return {"success": False, "error": f"Unsupported video input: {video_path}"}

            response = requests.post(f"{self.server_url}/infer_video", json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            if not result.get("success"):
                return result
            return self._save_video_outputs(
                video_path=video_path,
                result=result,
                save_overlay=save_overlay,
            )
        except Exception as e:
            logger.error("SAM3 video inference request failed: %s", e)
            return {"success": False, "error": str(e)}

    def _save_image_outputs(self, image_path: str, image: np.ndarray, result: Dict, save_overlay: bool) -> Dict:
        stem = Path(image_path).stem
        run_id = uuid.uuid4().hex[:12]
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

            mask_path = os.path.join(self.output_dir, f"sam3_mask_{stem}_{run_id}_{idx}.png")
            self._write_image(mask_path, mask_array)
            mask_record = dict(mask_info)
            mask_record.pop("mask", None)
            mask_record["mask_path"] = mask_path
            mask_records.append(mask_record)

        mask_path = os.path.join(self.output_dir, f"sam3_mask_{stem}_{run_id}.png")
        overlay_path = os.path.join(self.output_dir, f"sam3_overlay_{stem}_{run_id}.png")
        output_path = os.path.join(self.output_dir, f"sam3_combined_{stem}_{run_id}.png")

        self._write_image(mask_path, combined_mask)
        if save_overlay:
            self._write_image(overlay_path, overlay)
            combined = np.vstack([image, overlay])
            self._write_image(output_path, combined)
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

    def _save_video_outputs(self, video_path: str, result: Dict, save_overlay: bool) -> Dict:
        stem = Path(video_path).stem
        run_id = uuid.uuid4().hex[:12]
        result = dict(result)
        output_path = os.path.join(self.output_dir, f"sam3_video_{stem}_{run_id}.mp4")
        video_b64 = result.pop("video", None)
        if save_overlay:
            if not video_b64:
                raise ValueError("SAM3 server did not return the requested overlay video.")
            video_bytes = base64.b64decode(video_b64, validate=True)
            if len(video_bytes) < 12 or b"ftyp" not in video_bytes[4:12]:
                raise ValueError("SAM3 server returned an invalid MP4 overlay.")
            with open(output_path, "wb") as f:
                f.write(video_bytes)
            result["output_path"] = output_path
            result["video_path"] = output_path
        else:
            result["output_path"] = None
            result["video_path"] = None

        mask_dir = Path(self.output_dir) / f"sam3_video_{stem}_{run_id}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        frame_records = []
        flat_mask_paths = []
        for frame_record in result.pop("frame_masks", []) or []:
            frame_index = int(frame_record.get("frame_index", len(frame_records)))
            if frame_index < 0:
                raise ValueError("SAM3 server returned a negative frame index.")
            saved_paths = []
            for instance_index, mask_record in enumerate(frame_record.get("masks", []) or []):
                encoded = mask_record.get("mask") if isinstance(mask_record, dict) else mask_record
                mask_array = self._decode_mask(encoded)
                if mask_array is None:
                    continue
                mask_path = mask_dir / f"frame_{frame_index:06d}_instance_{instance_index:03d}.png"
                if not cv2.imwrite(str(mask_path), mask_array):
                    raise OSError(f"Unable to save SAM3 video mask: {mask_path}")
                saved_paths.append(str(mask_path))
                flat_mask_paths.append(str(mask_path))
            frame_records.append({"frame_index": frame_index, "mask_paths": saved_paths})

        result["masks"] = frame_records
        result["frame_mask_paths"] = flat_mask_paths
        result["mask_frames_dir"] = str(mask_dir)
        return result

    @staticmethod
    def _encode_frame_directory(frame_dir: Path) -> List[str]:
        frame_paths = [
            path for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ]
        frame_paths.sort(key=SAM3Client._frame_sort_key)

        encoded = []
        expected_size = None
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Unable to read JPEG frame: {frame_path}")
            size = frame.shape[:2]
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError("All JPEG frames must have the same dimensions.")
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError(f"Unable to encode JPEG frame: {frame_path}")
            encoded.append(base64.b64encode(buffer.tobytes()).decode("utf-8"))
        return encoded

    @staticmethod
    def _frame_sort_key(path: Path):
        if path.stem.isdigit():
            return 0, int(path.stem)
        return 1, path.name.lower()

    def _decode_mask(self, mask_b64: Optional[str]) -> Optional[np.ndarray]:
        if not mask_b64:
            return None
        mask_bytes = base64.b64decode(mask_b64, validate=True)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("SAM3 server returned an invalid PNG mask.")
        return mask

    @staticmethod
    def _write_image(path: str, image: np.ndarray) -> None:
        if not cv2.imwrite(path, image):
            raise OSError(f"Unable to save SAM3 image artifact: {path}")

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
