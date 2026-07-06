"""
Local FlowSeek optical flow inference client.

FlowSeek source is vendored into external_experts/FlowSeek/src/.
Only model weights need to be downloaded separately.

Setup (M variant, recommended):
  # Download Depth Anything V2 vitb weights
  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth \
      -O /your/path/depth_anything_v2_vitb.pth

  # Download FlowSeek M checkpoint
  gdown 1gbZ-6NE3muAnGqvypiS2s_BADHrI4ySf -O /your/path/flowseek_M_TartanCT_TSKH.pth

  export FLOWSEEK_CHECKPOINT=/your/path/flowseek_M_TartanCT_TSKH.pth
  export FLOWSEEK_DAV2_CHECKPOINT=/your/path/depth_anything_v2_vitb.pth

Setup (T variant, faster):
  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth \
      -O /your/path/depth_anything_v2_vits.pth
  gdown 1IQoyY5PpKSadtiGuhWwVCqvgD3y8CyFd -O /your/path/flowseek_T_TartanCT_TSKH.pth

  export FLOWSEEK_CHECKPOINT=/your/path/flowseek_T_TartanCT_TSKH.pth
  export FLOWSEEK_DAV2_CHECKPOINT=/your/path/depth_anything_v2_vits.pth
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parent / "src"

# Embedded model configs
_CONFIGS = {
    "M": {
        "name": "flowseek-M",
        "pretrain": "resnet34",
        "initial_dim": 64,
        "block_dims": [64, 128, 256],
        "radius": 4,
        "dim": 128,
        "num_blocks": 2,
        "iters": 4,
        "da_size": "vitb",
        "scale": 0,
        "use_var": False,
        "var_max": 10.0,
        "var_min": -10.0,
    },
    "T": {
        "name": "flowseek-T",
        "pretrain": "resnet18",
        "initial_dim": 64,
        "block_dims": [64, 128, 256],
        "radius": 4,
        "dim": 128,
        "num_blocks": 2,
        "iters": 4,
        "da_size": "vits",
        "scale": 0,
        "use_var": False,
        "var_max": 10.0,
        "var_min": -10.0,
    },
}


def _setup_src_path() -> None:
    src = str(_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


class FlowSeekLocalClient:
    """Lazy-loading local FlowSeek optical flow client (vendored source)."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        variant: str = "M",
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint or os.environ.get("FLOWSEEK_CHECKPOINT")
        if not self.checkpoint:
            raise EnvironmentError(
                "No checkpoint provided. Pass checkpoint= or set FLOWSEEK_CHECKPOINT.\n"
                "Download: gdown 1gbZ-6NE3muAnGqvypiS2s_BADHrI4ySf -O /path/flowseek_M_TartanCT_TSKH.pth"
            )
        if not Path(self.checkpoint).exists():
            raise EnvironmentError(f"Checkpoint not found: {self.checkpoint}")

        dav2_default = str(_SRC / "weights" / f"depth_anything_v2_{_CONFIGS[variant]['da_size']}.pth")
        self._dav2_checkpoint = os.environ.get("FLOWSEEK_DAV2_CHECKPOINT", dav2_default)
        if not Path(self._dav2_checkpoint).exists():
            raise EnvironmentError(
                f"Depth Anything V2 checkpoint not found: {self._dav2_checkpoint}\n"
                "Set FLOWSEEK_DAV2_CHECKPOINT or download:\n"
                "  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/"
                "depth_anything_v2_vitb.pth -O /your/path/depth_anything_v2_vitb.pth"
            )

        if variant not in _CONFIGS:
            raise ValueError(f"variant must be 'M' or 'T', got '{variant}'")
        self._variant = variant
        self._device = device
        self._model = None
        self._args = None

    def _ensure_model_loaded(self):
        if self._model is not None:
            return

        _setup_src_path()

        os.environ.setdefault("FLOWSEEK_DAV2_CHECKPOINT", self._dav2_checkpoint)

        try:
            from flowseek import FlowSeek
            from utils.utils import load_ckpt
        except ImportError as e:
            raise ImportError(
                f"Failed to import FlowSeek from vendored src. Error: {e}"
            ) from e

        cfg = _CONFIGS[self._variant]
        args = argparse.Namespace(**cfg)
        self._args = args

        logger.info("Loading FlowSeek-%s from %s", self._variant, self.checkpoint)
        model = FlowSeek(args)
        load_ckpt(model, self.checkpoint)
        model = model.to(self._device)
        model.eval()
        self._model = model
        logger.info("FlowSeek-%s loaded on %s", self._variant, self._device)

    def estimate_flow(
        self,
        image1_path: str,
        image2_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Estimate optical flow between two images.

        Args:
            image1_path: Path to the first (source) image.
            image2_path: Path to the second (target) image.
            output_path: Where to save the colorized flow image. Auto-generated if None.

        Returns:
            dict with keys: success, output_path, description, flow_magnitude_mean
        """
        import torch

        self._ensure_model_loaded()
        _setup_src_path()

        if not Path(image1_path).exists():
            return {"success": False, "error": f"Image not found: {image1_path}"}
        if not Path(image2_path).exists():
            return {"success": False, "error": f"Image not found: {image2_path}"}

        try:
            img1_bgr = cv2.imread(image1_path)
            img2_bgr = cv2.imread(image2_path)
            if img1_bgr is None:
                return {"success": False, "error": f"Could not read: {image1_path}"}
            if img2_bgr is None:
                return {"success": False, "error": f"Could not read: {image2_path}"}

            img1 = torch.from_numpy(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
            img2 = torch.from_numpy(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(self._device)

            with torch.no_grad():
                output = self._model(img1, img2, iters=self._args.iters, test_mode=True)

            flow = output["flow"][-1]  # (1, 2, H, W)
            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

            from utils.flow_viz import flow_to_image
            flow_vis = flow_to_image(flow_np, convert_to_bgr=True)

            if output_path is None:
                stem1 = Path(image1_path).stem
                stem2 = Path(image2_path).stem
                os.makedirs("outputs", exist_ok=True)
                output_path = f"outputs/flowseek_{stem1}_{stem2}.png"

            os.makedirs(Path(output_path).parent, exist_ok=True)
            cv2.imwrite(output_path, flow_vis)

            # Persist the raw dense (H, W, 2) u,v flow field alongside the
            # visualization. Without this the flow field — the tool's actual
            # informative output — is discarded, leaving only a scalar mean and
            # a colorized PNG. Consumers that need per-pixel flow load this .npy.
            flow_path = str(Path(output_path).with_suffix(".npy"))
            np.save(flow_path, flow_np.astype(np.float32))

            magnitude = float(np.sqrt(flow_np[..., 0] ** 2 + flow_np[..., 1] ** 2).mean())

            return {
                "success": True,
                "output_path": output_path,
                "flow_path": flow_path,
                "flow_shape": list(flow_np.shape),
                "flow_magnitude_mean": round(magnitude, 4),
                "description": (
                    f"FlowSeek-{self._variant} estimated optical flow from "
                    f"{Path(image1_path).name} to {Path(image2_path).name}. "
                    f"Mean flow magnitude: {magnitude:.2f} px. "
                    f"Colorized flow saved to {output_path}; "
                    f"raw (H,W,2) flow field saved to {flow_path}."
                ),
            }

        except Exception as e:
            logger.exception("FlowSeek inference error")
            return {"success": False, "error": str(e)}
