"""
Mock Depth Anything 3 client for testing and CI.

Returns fake depth (vertical gradient) without loading any model.
"""

from typing import Dict, Any

import cv2
import numpy as np


class MockDepthAnything3:
    """Mock client that returns a simple fake depth map."""

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Return fake depth for the given image.

        Returns:
            dict with "success", "depth" (np.ndarray float32 HxW), or "error"
        """
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                return {"success": False, "error": f"Failed to read image: {image_path}", "depth": None}

            h, w = image_bgr.shape[:2]
            depth = np.tile(
                np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1),
                (1, w),
            )
            return {"success": True, "depth": depth, "error": None}
        except Exception as e:
            return {"success": False, "error": str(e), "depth": None}
