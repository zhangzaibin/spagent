"""
Depth Anything 3 client for real inference.

Supports:
1. Official ByteDance API (depth_anything_3) when available.
2. Fallback: legacy depth_anything_v3.dpt style if present.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DepthAnything3Client:
    """
    Client that loads and runs Depth Anything V3 for monocular depth estimation.

    Prefers official depth_anything_3 API (ByteDance); falls back to
    depth_anything_v3.dpt if the former is not available.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        encoder: str = "vitl",
        input_size: int = 518,
    ):
        self.checkpoint_path_str = checkpoint_path
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.encoder = encoder
        self.input_size = input_size
        self.model = None
        self._backend = None  # "official" | "legacy"
        self._init_model()

    def _is_hf_model_id(self) -> bool:
        """Treat as HuggingFace model id if path does not exist and looks like 'org/name'."""
        if self.checkpoint_path.exists():
            return False
        s = self.checkpoint_path_str
        return "/" in s and not s.startswith("/") and not Path(s).anchor

    def _init_model(self) -> None:
        # Try official depth_anything_3 API first (HF model id or local dir)
        try:
            import torch
            from depth_anything_3.api import DepthAnything3

            if self._is_hf_model_id():
                model_id = self.checkpoint_path_str  # e.g. "depth-anything/DA3MONO-LARGE"
                model = DepthAnything3.from_pretrained(model_id)
                logger.info(
                    "DepthAnything3Client using official API, HF model_id=%s", model_id
                )
            elif self.checkpoint_path.exists():
                # Local: directory or file; from_pretrained accepts local dir
                model_dir = str(self.checkpoint_path) if self.checkpoint_path.is_dir() else str(self.checkpoint_path.parent)
                model = DepthAnything3.from_pretrained(model_dir)
                logger.info(
                    "DepthAnything3Client using official API, model_dir=%s", model_dir
                )
            else:
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path_str}")

            self.model = model.to(torch.device(self.device))
            self._torch = torch
            self._backend = "official"
            return
        except Exception as e:
            logger.debug("Official depth_anything_3 not used: %s", e)

        # Legacy path: require local .pth file
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path_str}")

        # Fallback: legacy depth_anything_v3.dpt
        try:
            import torch
            from depth_anything_v3.dpt import DepthAnythingV3

            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }
            if self.encoder not in model_configs:
                raise ValueError(f"Unsupported encoder: {self.encoder}")

            self.model = DepthAnythingV3(**model_configs[self.encoder])
            state_dict = torch.load(str(self.checkpoint_path), map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device).eval()
            self._torch = torch
            self._backend = "legacy"
            logger.info(
                "DepthAnything3Client using legacy depth_anything_v3.dpt, "
                "encoder=%s, checkpoint=%s", self.encoder, self.checkpoint_path
            )
            return
        except Exception as e:
            logger.error("DepthAnything3Client failed to load model: %s", e, exc_info=True)
            raise

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Run depth estimation on one image.

        Returns:
            dict with "success", "depth" (np.ndarray float32 HxW), or "error"
        """
        if self.model is None:
            return {"success": False, "error": "Model not initialized", "depth": None}

        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image not found: {image_path}", "depth": None}

            if self._backend == "official":
                depth = self._predict_official(image_path)
            else:
                image_bgr = cv2.imread(image_path)
                if image_bgr is None:
                    return {"success": False, "error": f"Failed to read image: {image_path}", "depth": None}
                depth = self._predict_legacy(image_bgr)

            if depth is None:
                return {"success": False, "error": "Inference returned no depth", "depth": None}
            return {"success": True, "depth": depth.astype(np.float32), "error": None}
        except Exception as e:
            logger.exception("DepthAnything3Client predict error")
            return {"success": False, "error": str(e), "depth": None}

    def _predict_official(self, image_path: str) -> Optional[np.ndarray]:
        """Use depth_anything_3 API: inference(images) -> prediction.depth [N, H, W]."""
        pred = self.model.inference([image_path])
        depth = pred.depth
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        if isinstance(depth, np.ndarray) and depth.ndim == 3:
            depth = depth[0]
        return depth

    def _predict_legacy(self, image_bgr: np.ndarray) -> np.ndarray:
        """Use depth_anything_v3.dpt: infer_image(image_rgb, input_size=...)."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if hasattr(self.model, "infer_image"):
            depth = self.model.infer_image(image_rgb, input_size=self.input_size)
        else:
            raise NotImplementedError("Legacy model has no infer_image")
        if not isinstance(depth, np.ndarray):
            depth = np.array(depth, dtype=np.float32)
        return depth.astype(np.float32)
