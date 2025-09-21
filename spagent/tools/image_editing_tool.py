"""
Image Editing Tool
Direct implementation of image editing operations (no external server).
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class ImageEditingTool(Tool):
    """Tool for performing various image editing operations"""

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize image editing tool

        Args:
            output_dir: directory to save outputs
        """
        super().__init__(
            name="image_editing_tool",
            description="Perform various image editing tasks on an input image. "
                        "Supported tasks: crop, rotate, enhance_contrast, segmentation, "
                        "detect_crop, mark_points, render_aux_lines, histogram, avg_rgb."
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for editing."
                },
                "task": {
                    "type": "string",
                    "description": "The image editing task to perform.",
                    "enum": [
                        "crop", "rotate", "enhance_contrast", "segmentation",
                        "detect_crop", "mark_points", "render_aux_lines",
                        "histogram", "avg_rgb"
                    ]
                },
                "params": {
                    "type": "object",
                    "description": "Additional task-specific parameters (e.g., bbox, angle, mask, points)."
                }
            },
            "required": ["image_path", "task"]
        }

    def call(self, image_path: str, task: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute image editing task

        Args:
            image_path: Path to input image
            task: Image editing task name
            params: Additional parameters for the task

        Returns:
            Image editing result dictionary
        """
        try:
            logger.info(f"Running image editing {task} on: {image_path}")
            params = params or {}

            # Check if image exists
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            # Load image
            img = Image.open(image_path).convert("RGB")
            filename = Path(image_path).stem
            out_path = self.output_dir / f"{filename}_{task}.jpg"

            # ---- Dispatch ----
            if task == "crop":
                bbox = params.get("bbox", [0, 0, img.width, img.height])
                result_img = img.crop(bbox)
                result_img.save(out_path)

            elif task == "rotate":
                angle = params.get("angle", 0)
                result_img = img.rotate(angle, expand=True)
                result_img.save(out_path)

            elif task == "enhance_contrast":
                enhancer = ImageEnhance.Contrast(img)
                result_img = enhancer.enhance(params.get("factor", 1.5))
                result_img.save(out_path)

            elif task == "segmentation":
                mask_path = params.get("mask")
                if not mask_path or not Path(mask_path).exists():
                    return {"success": False, "error": "Mask file not found"}
                mask = Image.open(mask_path).convert("L")
                result_img = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)
                result_img.save(out_path)

            elif task == "mark_points":
                points: List[Tuple[int, int]] = params.get("points", [])
                draw = ImageDraw.Draw(img)
                for (x, y) in points:
                    r = 5
                    draw.ellipse((x-r, y-r, x+r, y+r), fill="red")
                img.save(out_path)

            elif task == "render_aux_lines":
                draw = ImageDraw.Draw(img)
                # draw grid lines
                step_x, step_y = img.width // 4, img.height // 4
                for x in range(step_x, img.width, step_x):
                    draw.line([(x, 0), (x, img.height)], fill="blue", width=1)
                for y in range(step_y, img.height, step_y):
                    draw.line([(0, y), (img.width, y)], fill="blue", width=1)
                img.save(out_path)

            elif task == "histogram":
                np_img = np.array(img)
                colors = ('r', 'g', 'b')
                plt.figure()
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([np_img], [i], None, [256], [0, 256])
                    plt.plot(hist, color=col)
                    plt.xlim([0, 256])
                hist_path = str(out_path).replace(".jpg", "_hist.png")
                plt.savefig(hist_path)
                plt.close()
                return {"success": True, "task": task, "output_path": hist_path}

            elif task == "avg_rgb":
                np_img = np.array(img)
                mean_rgb = np_img.mean(axis=(0, 1)).tolist()  # [R, G, B]
                return {"success": True, "task": task, "avg_rgb": mean_rgb}

            else:
                return {"success": False, "error": f"Invalid task type: {task}"}

            return {"success": True, "task": task, "output_path": str(out_path)}

        except Exception as e:
            logger.error(f"Image editing tool error: {e}")
            return {"success": False, "error": str(e)}
