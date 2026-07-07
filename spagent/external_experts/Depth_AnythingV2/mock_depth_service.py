"""
Mock depth estimation service — format-faithful to the real DepthClient.

``DepthEstimationTool(use_mock=True)`` imports ``MockDepthService`` from this
module and RAISES if the import fails; this file was missing from the repo,
so mock mode could never construct. The real client (depth_client.py) returns:

    {'depth_array': <HxWx3 array>, 'combined_array': <...>, 'shape': [...],
     'output_path': <combined png>, 'depth_only_path': <depth png>,
     'success': True}

This mock reproduces that shape with a small synthetic gradient depth map so
downstream code (tool payload, renderer, tests) sees real-format data with
zero model/server dependencies.
"""

import os
from pathlib import Path
from typing import Any, Dict

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard repo dependency
    np = None


class MockDepthService:
    """Drop-in mock exposing the real client's ``infer`` method/shape."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir

    def infer(self, image_path: str, **kwargs: Any) -> Dict[str, Any]:
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        h, w = 64, 64
        depth_array = None
        if np is not None:
            # horizontal near->far gradient, colored-shaped (H, W, 3) uint8
            ramp = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
            depth_array = np.stack([ramp] * 3, axis=-1)

        os.makedirs(self.output_dir, exist_ok=True)
        stem = Path(image_path).stem
        combined_path = os.path.join(self.output_dir, f"depth_combined_{stem}.png")
        depth_only_path = os.path.join(self.output_dir, f"depth_only_{stem}.png")
        try:
            import cv2
            if depth_array is not None:
                cv2.imwrite(depth_only_path, depth_array)
                cv2.imwrite(combined_path, depth_array)
        except Exception:
            # No cv2/numpy: still return the real shape; paths may not exist,
            # which the contract tolerates (carrier presence, not file I/O).
            pass

        return {
            "depth_array": depth_array,
            "combined_array": depth_array,
            "shape": [h, w],
            "output_path": combined_path,
            "depth_only_path": depth_only_path,
            "success": True,
        }
