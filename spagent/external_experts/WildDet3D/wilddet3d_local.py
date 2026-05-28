"""
Local WildDet3D inference client.

Requires:
  - WILDDET3D_ROOT env var pointing to a clone of https://github.com/allenai/WildDet3D
    (cloned with --recurse-submodules)
  - WILDDET3D_CHECKPOINT env var pointing to the .pt checkpoint file

Setup:
  git clone --recurse-submodules https://github.com/allenai/WildDet3D.git /your/path
  export WILDDET3D_ROOT=/your/path/WildDet3D
  export WILDDET3D_CHECKPOINT=/your/path/wilddet3d_alldata_all_prompt_v1.0.pt
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _setup_wilddet3d_path() -> str:
    root = os.environ.get("WILDDET3D_ROOT")
    if not root:
        raise EnvironmentError(
            "WILDDET3D_ROOT is not set. "
            "Clone https://github.com/allenai/WildDet3D (with --recurse-submodules) "
            "and set: export WILDDET3D_ROOT=/path/to/WildDet3D"
        )
    sys.path.insert(0, root)
    sys.path.insert(0, str(Path(root) / "third_party" / "sam3"))
    sys.path.insert(0, str(Path(root) / "third_party" / "lingbot_depth"))
    return root


class WildDet3DLocalClient:
    """Lazy-loading local WildDet3D client."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        score_threshold: float = 0.3,
        score_3d_threshold: float = 0.1,
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint or os.environ.get("WILDDET3D_CHECKPOINT")
        if not self.checkpoint:
            raise EnvironmentError(
                "No checkpoint path provided. Pass checkpoint= or set WILDDET3D_CHECKPOINT."
            )
        self.score_threshold = score_threshold
        self.score_3d_threshold = score_3d_threshold
        self.device = device
        self._model = None

    def _ensure_model_loaded(self):
        if self._model is not None:
            return
        _setup_wilddet3d_path()
        try:
            from wilddet3d import build_model
        except ImportError as e:
            raise ImportError(
                f"Failed to import wilddet3d. Check WILDDET3D_ROOT is set correctly. Error: {e}"
            ) from e

        logger.info("Loading WildDet3D from %s", self.checkpoint)
        self._model = build_model(
            checkpoint=self.checkpoint,
            score_threshold=self.score_threshold,
            score_3d_threshold=self.score_3d_threshold,
            device=self.device,
            skip_pretrained=True,
        )
        self._model.eval()
        logger.info("WildDet3D loaded on %s", self.device)

    def detect(
        self,
        image_path: str,
        prompt_text: str = "object",
        input_boxes: Optional[List[float]] = None,
        input_points: Optional[List[List]] = None,
        intrinsics: Optional[List[List[float]]] = None,
    ) -> Dict:
        """
        Run 3D detection on a single image.

        Args:
            image_path: Path to RGB image.
            prompt_text: Text prompt (e.g. "chair"). Used when no box/point provided.
            input_boxes: Optional 2D box prompt [x1, y1, x2, y2] in pixel coords.
            input_points: Optional point prompts [[x, y, label], ...] where label=1
                          (foreground) or 0 (background).
            intrinsics: Optional 3x3 camera intrinsics as nested list.

        Returns:
            dict with keys: success, boxes2d, boxes3d, scores, output_path, description
        """
        import numpy as np
        from PIL import Image

        self._ensure_model_loaded()
        _setup_wilddet3d_path()
        from wilddet3d import preprocess

        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        try:
            import torch

            img_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)

            K = None
            if intrinsics is not None:
                K = torch.tensor(intrinsics, dtype=torch.float32)

            inputs = preprocess(img_np, intrinsics=K)

            valid_keys = {"images", "intrinsics", "input_hw", "original_hw", "padding", "depth_gt"}
            inputs = {k: v for k, v in inputs.items() if k in valid_keys}

            inputs["input_hw"] = [inputs["input_hw"]]
            inputs["original_hw"] = [inputs["original_hw"]]
            inputs["padding"] = [tuple(inputs["padding"])]
            if inputs.get("intrinsics") is not None:
                inputs["intrinsics"] = inputs["intrinsics"].unsqueeze(0)

            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            # Build prompt kwargs — box/point take priority over text
            prompt_kwargs: Dict = {}
            if input_boxes is not None:
                prompt_kwargs["input_boxes"] = [input_boxes]
                prompt_kwargs["prompt_text"] = "geometric"
            elif input_points is not None:
                prompt_kwargs["input_points"] = [[(p[0], p[1], int(p[2])) for p in input_points]]
                prompt_kwargs["prompt_text"] = "geometric"
            else:
                prompt_kwargs["input_texts"] = [prompt_text]
                prompt_kwargs["prompt_text"] = "object"

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self._model(**inputs, **prompt_kwargs)

            boxes2d, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = outputs

            boxes2d_list = boxes2d[0].float().cpu().numpy().tolist() if len(boxes2d) > 0 else []
            scores_list = scores[0].float().cpu().numpy().tolist() if len(scores) > 0 else []

            output_path = self._visualize(image_path, boxes2d_list, scores_list, prompt_text)

            return {
                "success": True,
                "boxes2d": boxes2d_list,
                "boxes3d": boxes3d[0].float().cpu().numpy().tolist() if len(boxes3d) > 0 else [],
                "scores": scores_list,
                "num_detections": len(boxes2d_list),
                "output_path": output_path,
                "description": (
                    f"WildDet3D detected {len(boxes2d_list)} object(s) matching '{prompt_text}'. "
                    f"Results visualized with 2D bounding boxes."
                ),
            }

        except Exception as e:
            logger.exception("WildDet3D inference error")
            return {"success": False, "error": str(e)}

    def _visualize(
        self,
        image_path: str,
        boxes2d: List,
        scores: List,
        prompt_text: str,
    ) -> str:
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            return image_path

        for i, box in enumerate(boxes2d):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            score = scores[i] if i < len(scores) else 0.0
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{prompt_text} {score:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        os.makedirs("outputs", exist_ok=True)
        stem = Path(image_path).stem
        output_path = f"outputs/wilddet3d_{stem}.png"
        cv2.imwrite(output_path, img)
        return output_path
