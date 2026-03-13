"""
Depth Anything 3 Tool

SPAgent tool for monocular depth estimation. Uses the server/client pattern:
- Real: external_experts.depth_anything3.depth_anything3_client.DepthAnything3Client
- Mock: external_experts.depth_anything3.mock_depth_anything3.MockDepthAnything3
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class DepthAnything3Tool(Tool):
    """Tool for monocular depth estimation using Depth Anything V3."""

    def __init__(
        self,
        use_mock: bool = True,
        device: str = "cuda",
        encoder: str = "vitl",
        checkpoint_path: Optional[str] = None,
        save_dir: Optional[str] = None,
        input_size: int = 518,
    ):
        """
        Args:
            use_mock: If True, use mock client (no real model).
            device: Device for inference, e.g. "cuda" or "cpu".
            encoder: Depth Anything V3 encoder variant: "vits", "vitb", or "vitl".
            checkpoint_path: Path to model checkpoint (.pth or model dir). Required for real inference.
            save_dir: Directory to save outputs. If None, save next to input image.
            input_size: Input size for the model (used in real inference).
        """
        super().__init__(
            name="depth_anything3_tool",
            description=(
                "Estimate monocular depth from a single RGB image using Depth Anything V3. "
                "Use this tool when you need a depth map, per-pixel relative depth, or geometric "
                "scene understanding from one image."
            ),
        )
        self.use_mock = use_mock
        self.device = device
        self.encoder = encoder
        self.checkpoint_path = checkpoint_path
        self.save_dir = Path(save_dir) if save_dir else None
        self.input_size = input_size
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize mock or real client from external_experts (per ADDING_NEW_TOOLS.md)."""
        if self.use_mock:
            from external_experts.depth_anything3.mock_depth_anything3 import MockDepthAnything3
            self._client = MockDepthAnything3()
            logger.info("DepthAnything3Tool initialized in mock mode.")
        else:
            from external_experts.depth_anything3.depth_anything3_client import DepthAnything3Client
            if not self.checkpoint_path:
                raise ValueError("checkpoint_path is required when use_mock=False.")
            self._client = DepthAnything3Client(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                encoder=self.encoder,
                input_size=self.input_size,
            )
            logger.info(
                "DepthAnything3Tool real client loaded: encoder=%s, checkpoint=%s",
                self.encoder, self.checkpoint_path,
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input RGB image."
                },
                "output_format": {
                    "type": "string",
                    "enum": ["png", "npy", "both"],
                    "description": (
                        "Output format for the depth result. "
                        "'png' saves a visualized depth map, "
                        "'npy' saves the raw depth array, "
                        "'both' saves both."
                    ),
                    "default": "both"
                },
                "colormap": {
                    "type": "string",
                    "enum": ["gray", "inferno", "magma", "viridis", "plasma"],
                    "description": "Colormap for depth PNG visualization.",
                    "default": "inferno"
                },
                "normalize": {
                    "type": "boolean",
                    "description": "Whether to normalize depth before visualization.",
                    "default": True
                }
            },
            "required": ["image_path"]
        }

    @staticmethod
    def _apply_colormap(depth_uint8: np.ndarray, colormap: str) -> np.ndarray:
        cmap_dict = {
            "gray": None,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
        }
        cmap = cmap_dict[colormap]
        if cmap is None:
            return depth_uint8
        return cv2.applyColorMap(depth_uint8, cmap)

    def _save_outputs(
        self,
        image_path: Path,
        depth: np.ndarray,
        output_format: str,
        colormap: str,
        normalize: bool,
    ) -> Dict[str, Optional[str]]:
        """Save depth outputs to disk."""
        save_root = self.save_dir if self.save_dir else image_path.parent
        save_root.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        png_path = save_root / f"{stem}_depth.png"
        npy_path = save_root / f"{stem}_depth.npy"

        output_path = None

        if output_format in ["png", "both"]:
            depth_vis = depth.copy()
            if normalize:
                depth_vis = depth_vis - depth_vis.min()
                if depth_vis.max() > 1e-8:
                    depth_vis = depth_vis / depth_vis.max()
                depth_vis = (depth_vis * 255.0).astype(np.uint8)
            else:
                depth_vis = np.clip(depth_vis, 0, 255).astype(np.uint8)

            depth_color = self._apply_colormap(depth_vis, colormap)
            ok = cv2.imwrite(str(png_path), depth_color)
            if not ok:
                raise IOError(f"Failed to save depth PNG to {png_path}")
            output_path = str(png_path)

        if output_format in ["npy", "both"]:
            np.save(str(npy_path), depth)

        return {
            "depth_png_path": str(png_path) if output_format in ["png", "both"] else None,
            "depth_npy_path": str(npy_path) if output_format in ["npy", "both"] else None,
            "output_path": output_path,
        }

    def call(
        self,
        image_path: str,
        output_format: str = "both",
        colormap: str = "inferno",
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Run depth estimation via external_experts client; save and return result.
        """
        try:
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            if output_format not in {"png", "npy", "both"}:
                return {
                    "success": False,
                    "error": f"Invalid output_format: {output_format}"
                }

            result = self._client.predict(image_path=str(image_path_obj))
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error", "Unknown depth estimation error"),
                }

            depth = result.get("depth")
            if depth is None or depth.ndim != 2:
                return {
                    "success": False,
                    "error": f"Depth output invalid (shape={getattr(depth, 'shape', None)})",
                }

            outputs = self._save_outputs(
                image_path=image_path_obj,
                depth=depth,
                output_format=output_format,
                colormap=colormap,
                normalize=normalize,
            )

            out_result = {
                "depth_min": float(depth.min()),
                "depth_max": float(depth.max()),
                "shape": list(depth.shape),
                "depth_png_path": outputs["depth_png_path"],
                "depth_npy_path": outputs["depth_npy_path"],
            }

            return {
                "success": True,
                "result": out_result,
                "output_path": outputs["output_path"],
                "summary": (
                    f"Depth estimation completed for {image_path_obj.name}. "
                    f"Depth shape: {depth.shape[0]}x{depth.shape[1]}."
                ),
            }

        except Exception as e:
            logger.error(f"DepthAnything3Tool error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
