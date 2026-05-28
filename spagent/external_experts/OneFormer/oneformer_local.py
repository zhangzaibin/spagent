"""
OneFormer local inference client.

Loads model via HuggingFace Transformers — no manual checkpoint download needed.
Supports three segmentation tasks: semantic, instance, panoptic.

Environment variable:
    ONEFORMER_MODEL_ID  — HuggingFace model ID (default: shi-labs/oneformer_ade20k_swin_large)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "shi-labs/oneformer_ade20k_swin_large"

# ADE20k color palette (150 classes), truncated/extended as needed
_ADE20K_PALETTE = [
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255],
]


class OneFormerLocalClient:
    """In-process OneFormer client using HuggingFace Transformers."""

    def __init__(self, model_id: Optional[str] = None, device: str = "cuda"):
        self.model_id = model_id or os.environ.get("ONEFORMER_MODEL_ID", _DEFAULT_MODEL_ID)
        self.device = device
        self._processor = None
        self._model = None

    def _ensure_model_loaded(self):
        if self._model is not None:
            return

        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        logger.info("Loading OneFormer processor: %s", self.model_id)
        self._processor = OneFormerProcessor.from_pretrained(self.model_id)

        logger.info("Loading OneFormer model: %s", self.model_id)
        self._model = OneFormerForUniversalSegmentation.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        logger.info("OneFormer model loaded on %s", self.device)

    def segment(
        self,
        image_path: str,
        task: str = "panoptic",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run segmentation on an image.

        Args:
            image_path: Path to input image.
            task: One of 'semantic', 'instance', 'panoptic'.
            output_path: Where to save annotated image. Auto-generated if None.

        Returns:
            dict with keys: success, task, output_path, description,
                            segments (list), num_segments
        """
        import torch
        from PIL import Image

        if task not in ("semantic", "instance", "panoptic"):
            return {"success": False, "error": f"Invalid task '{task}'. Choose semantic/instance/panoptic."}

        try:
            self._ensure_model_loaded()

            pil_image = Image.open(image_path).convert("RGB")
            w, h = pil_image.size

            inputs = self._processor(
                images=pil_image,
                task_inputs=[task],
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            segments = []
            seg_map = None

            if task == "semantic":
                seg_map = self._processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(h, w)]
                )[0].cpu().numpy()
                unique_ids = np.unique(seg_map)
                id2label = self._model.config.id2label
                segments = [
                    {"id": int(uid), "label": id2label.get(int(uid), str(uid))}
                    for uid in unique_ids
                ]

            elif task == "instance":
                result = self._processor.post_process_instance_segmentation(
                    outputs, target_sizes=[(h, w)]
                )[0]
                seg_map = result["segmentation"].cpu().numpy()
                id2label = self._model.config.id2label
                segments = [
                    {
                        "id": int(s["id"]),
                        "label": id2label.get(int(s["label_id"]), str(s["label_id"])),
                        "score": round(float(s["score"]), 4) if "score" in s else None,
                    }
                    for s in result["segments_info"]
                ]

            elif task == "panoptic":
                result = self._processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=[(h, w)]
                )[0]
                seg_map = result["segmentation"].cpu().numpy()
                id2label = self._model.config.id2label
                segments = [
                    {
                        "id": int(s["id"]),
                        "label": id2label.get(int(s["label_id"]), str(s["label_id"])),
                        "score": round(float(s["score"]), 4) if "score" in s else None,
                    }
                    for s in result["segments_info"]
                ]

            # Build color visualization
            vis = self._colorize(seg_map, task, segments)

            # Save
            if output_path is None:
                stem = Path(image_path).stem
                os.makedirs("outputs", exist_ok=True)
                output_path = f"outputs/oneformer_{task}_{stem}.png"

            import cv2
            cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            labels = [s["label"] for s in segments[:5]]
            more = f" and {len(segments) - 5} more" if len(segments) > 5 else ""
            description = (
                f"OneFormer {task} segmentation: {len(segments)} segment(s). "
                f"Labels: {', '.join(labels)}{more}."
            )

            return {
                "success": True,
                "task": task,
                "output_path": output_path,
                "description": description,
                "segments": segments,
                "num_segments": len(segments),
            }

        except Exception as e:
            logger.exception("OneFormerLocalClient.segment error")
            return {"success": False, "error": str(e)}

    def _colorize(
        self,
        seg_map: np.ndarray,
        task: str,
        segments: List[Dict],
    ) -> np.ndarray:
        """Produce an RGB visualization of the segmentation map."""
        h, w = seg_map.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        if task == "semantic":
            for uid in np.unique(seg_map):
                color = _ADE20K_PALETTE[int(uid) % len(_ADE20K_PALETTE)]
                vis[seg_map == uid] = color
        else:
            # instance / panoptic: color by segment id
            for i, seg in enumerate(segments):
                sid = seg["id"]
                color = _ADE20K_PALETTE[i % len(_ADE20K_PALETTE)]
                vis[seg_map == sid] = color

        return vis
