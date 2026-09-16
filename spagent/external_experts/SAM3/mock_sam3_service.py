import hashlib
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class MockSAM3Service:
    """Mock SAM3 service for image/video text-prompt segmentation tests."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or str(Path(tempfile.gettempdir()) / "spagent_sam3_mock")
        os.makedirs(self.output_dir, exist_ok=True)

    def infer(
        self,
        image_path: str,
        text_prompt: str,
        score_threshold: float = 0.5,
        max_instances: int = 20,
        save_overlay: bool = True,
        **kwargs,
    ) -> Dict:
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            count = max(1, min(int(max_instances), 2))
            boxes = self._boxes_for_prompt(text_prompt, width, height, count)
            run_id = uuid.uuid4().hex[:12]
            stem = Path(image_path).stem

            overlay = image.copy()
            overlay_draw = ImageDraw.Draw(overlay, "RGBA")
            combined_mask = Image.new("L", image.size, 0)
            masks: List[Dict] = []
            scores: List[float] = []

            for idx, box in enumerate(boxes):
                score = max(float(score_threshold), 0.72 - idx * 0.08)
                scores.append(round(score, 4))

                mask = Image.new("L", image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle(box, fill=255)
                combined_mask = Image.fromarray(
                    np.maximum(np.asarray(combined_mask), np.asarray(mask)).astype(np.uint8)
                )

                color = self._color(idx)
                overlay_draw.rectangle(box, fill=color + (90,), outline=color + (255,), width=3)
                overlay_draw.text((box[0] + 4, box[1] + 4), f"{text_prompt}:{idx}", fill=color + (255,))

                mask_path = os.path.join(self.output_dir, f"sam3_mock_mask_{stem}_{run_id}_{idx}.png")
                mask.save(mask_path)
                masks.append(
                    {
                        "id": idx,
                        "mask_path": mask_path,
                        "bbox": list(box),
                        "score": scores[-1],
                        "label": text_prompt,
                    }
                )

            overlay_path = os.path.join(self.output_dir, f"sam3_mock_overlay_{stem}_{run_id}.png")
            output_path = os.path.join(self.output_dir, f"sam3_mock_combined_{stem}_{run_id}.png")
            mask_path = os.path.join(self.output_dir, f"sam3_mock_mask_{stem}_{run_id}.png")

            if save_overlay:
                overlay.save(overlay_path)
                Image.new("RGB", (width, height * 2), "white").save(output_path)
                combined = Image.open(output_path)
                combined.paste(image, (0, 0))
                combined.paste(overlay, (0, height))
                combined.save(output_path)
            else:
                overlay_path = None
                output_path = None
            combined_mask.save(mask_path)

            return {
                "success": True,
                "task": "image",
                "text_prompt": text_prompt,
                "output_path": output_path,
                "overlay_path": overlay_path,
                "mask_path": mask_path,
                "shape": [height, width],
                "masks": masks,
                "boxes": [list(box) for box in boxes],
                "scores": scores,
            }
        except Exception as e:
            logger.error("Mock SAM3 image inference failed: %s", e)
            return {"success": False, "error": str(e)}

    def infer_video(
        self,
        video_path: str,
        text_prompt: str,
        frame_index: int = 0,
        score_threshold: float = 0.5,
        max_instances: int = 20,
        save_overlay: bool = True,
        **kwargs,
    ) -> Dict:
        try:
            import cv2
        except ImportError as e:
            return {"success": False, "error": f"OpenCV is required for video mock: {e}"}

        source = Path(video_path)
        cap = None
        frame_paths = None
        if source.is_dir():
            frame_paths = [
                path for path in source.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            ]
            frame_paths.sort(key=self._frame_sort_key)
            if not frame_paths:
                return {"success": False, "error": f"No JPEG frames found: {video_path}"}
            first_frame = cv2.imread(str(frame_paths[0]))
            if first_frame is None:
                return {"success": False, "error": f"Unable to read frame: {frame_paths[0]}"}
            height, width = first_frame.shape[:2]
            fps = 5.0
            expected_frames = len(frame_paths)
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": f"Unable to open video: {video_path}"}
            fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if int(frame_index) < 0 or (expected_frames > 0 and int(frame_index) >= expected_frames):
            if cap is not None:
                cap.release()
            return {
                "success": False,
                "error": f"frame_index {frame_index} is outside {expected_frames} frames",
            }
        os.makedirs(self.output_dir, exist_ok=True)
        run_id = uuid.uuid4().hex[:12]
        stem = Path(video_path).stem
        output_path = os.path.join(self.output_dir, f"sam3_mock_video_{stem}_{run_id}.mp4")
        writer = None
        if save_overlay:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if writer is not None and not writer.isOpened():
            if cap is not None:
                cap.release()
            return {"success": False, "error": "Unable to create output video writer."}

        boxes = self._boxes_for_prompt(text_prompt, width, height, max(1, min(int(max_instances), 2)))
        frame_count = 0
        mask_dir = Path(self.output_dir) / f"sam3_mock_video_{stem}_{run_id}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        frame_masks = []
        while True:
            if frame_paths is not None:
                if frame_count >= len(frame_paths):
                    break
                frame = cv2.imread(str(frame_paths[frame_count]))
                if frame is None:
                    if writer is not None:
                        writer.release()
                    return {"success": False, "error": f"Unable to read frame: {frame_paths[frame_count]}"}
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            if frame.shape[:2] != (height, width):
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()
                return {"success": False, "error": "All video frames must have the same dimensions."}
            overlay = frame.copy()
            mask_paths = []
            for idx, box in enumerate(boxes):
                color = self._color(idx)[:3][::-1]
                x1, y1, x2, y2 = box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                mask_path = mask_dir / f"frame_{frame_count:06d}_instance_{idx:03d}.png"
                if not cv2.imwrite(str(mask_path), mask):
                    if cap is not None:
                        cap.release()
                    if writer is not None:
                        writer.release()
                    return {"success": False, "error": f"Unable to save mask: {mask_path}"}
                mask_paths.append(str(mask_path))
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            cv2.putText(frame, text_prompt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if writer is not None:
                writer.write(frame)
            frame_masks.append({"frame_index": frame_count, "mask_paths": mask_paths})
            frame_count += 1

        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        return {
            "success": True,
            "task": "video",
            "text_prompt": text_prompt,
            "output_path": output_path if save_overlay else None,
            "video_path": output_path if save_overlay else None,
            "frames": frame_count,
            "fps": fps,
            "size": [width, height],
            "frame_index": int(frame_index),
            "boxes": [list(box) for box in boxes],
            "scores": [round(max(float(score_threshold), 0.72 - i * 0.08), 4) for i in range(len(boxes))],
            "masks": frame_masks,
            "frame_mask_paths": [path for record in frame_masks for path in record["mask_paths"]],
            "mask_frames_dir": str(mask_dir),
        }

    @staticmethod
    def _frame_sort_key(path: Path):
        if path.stem.isdigit():
            return 0, int(path.stem)
        return 1, path.name.lower()

    def _boxes_for_prompt(self, text_prompt: str, width: int, height: int, count: int) -> List[Tuple[int, int, int, int]]:
        digest = hashlib.sha256(text_prompt.encode("utf-8")).digest()
        boxes = []
        for idx in range(count):
            base = digest[idx]
            box_w = min(width, max(1, max(16, width // (3 + idx))))
            box_h = min(height, max(1, max(16, height // (3 + idx))))
            x1 = int((base / 255.0) * max(1, width - box_w))
            y1 = int((digest[-idx - 1] / 255.0) * max(1, height - box_h))
            boxes.append((x1, y1, x1 + box_w, y1 + box_h))
        return boxes

    def _color(self, idx: int) -> Tuple[int, int, int]:
        palette = [
            (245, 88, 88),
            (52, 168, 83),
            (66, 133, 244),
            (251, 188, 5),
            (171, 71, 188),
        ]
        return palette[idx % len(palette)]
