"""
Crop Tool

Local image cropping utility for box, multi-box, mask, and polygon crops.
"""

import logging
import sys
import uuid
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class CropTool(Tool):
    """Tool for extracting image regions of interest."""

    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(
            name="crop_tool",
            description=(
                "Crop regions of interest from an image using one box, multiple boxes, "
                "a mask image, or a polygon. Supports pixel and relative coordinates."
            ),
        )
        self.output_dir = Path(output_dir) if output_dir else Path(gettempdir()) / "spagent_crops"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to the input image."},
                "box": {
                    "type": "array",
                    "description": "Single crop box in xyxy format: [x1, y1, x2, y2].",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "boxes": {
                    "type": "array",
                    "description": "Multiple crop boxes in xyxy format: [[x1, y1, x2, y2], ...].",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "mask_path": {"type": "string", "description": "Optional binary mask image path for mask crop."},
                "polygon": {
                    "type": "array",
                    "description": "Optional polygon points: [[x, y], ...].",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "padding": {
                    "type": "number",
                    "description": "Padding around the crop region. Pixel value by default; relative value when relative_coords=True.",
                    "default": 0,
                },
                "relative_coords": {
                    "type": "boolean",
                    "description": "Whether box, boxes, polygon, and padding are normalized to [0, 1].",
                    "default": False,
                },
                "output_dir": {"type": "string", "description": "Optional output directory for cropped images."},
            },
            "required": ["image_path"],
            "anyOf": [
                {"required": ["box"]},
                {"required": ["boxes"]},
                {"required": ["mask_path"]},
                {"required": ["polygon"]},
            ],
        }

    def call(
        self,
        image_path: str,
        box: Optional[List[float]] = None,
        boxes: Optional[List[List[float]]] = None,
        mask_path: Optional[str] = None,
        polygon: Optional[List[List[float]]] = None,
        padding: float = 0,
        relative_coords: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            path = Path(image_path)
            if not path.exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            destination = Path(output_dir) if output_dir else self.output_dir
            destination.mkdir(parents=True, exist_ok=True)

            with Image.open(path) as image:
                image = image.convert("RGBA")
                width, height = image.size

                modes = sum(value is not None for value in [box, boxes, mask_path, polygon])
                if modes != 1:
                    return {
                        "success": False,
                        "error": "Provide exactly one crop input: box, boxes, mask_path, or polygon.",
                    }

                if box is not None:
                    crop = self._crop_box(image, path.stem, [box], padding, relative_coords, destination)[0]
                    return {"success": True, **crop, "original_size": [width, height], "mode": "box"}

                if boxes is not None:
                    crops = self._crop_box(image, path.stem, boxes, padding, relative_coords, destination)
                    return {
                        "success": True,
                        "crops": crops,
                        "output_paths": [crop["output_path"] for crop in crops],
                        "original_size": [width, height],
                        "mode": "boxes",
                    }

                if mask_path is not None:
                    crop = self._crop_mask(image, path.stem, mask_path, padding, relative_coords, destination)
                    return {"success": True, **crop, "original_size": [width, height], "mode": "mask"}

                crop = self._crop_polygon(image, path.stem, polygon, padding, relative_coords, destination)
                return {"success": True, **crop, "original_size": [width, height], "mode": "polygon"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error("Crop tool error: %s", e)
            return {"success": False, "error": str(e)}

    def _crop_box(
        self,
        image: Image.Image,
        stem: str,
        boxes: List[List[float]],
        padding: float,
        relative: bool,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        crops = []
        run_id = uuid.uuid4().hex[:8]
        for idx, raw_box in enumerate(boxes):
            crop_box = _normalize_box(raw_box, image.size, padding, relative)
            cropped = image.crop(crop_box)
            output_path = output_dir / f"{stem}_crop_{run_id}_{idx}.png"
            cropped.save(output_path)
            crops.append(_record(output_path, crop_box, cropped.size))
        return crops

    def _crop_mask(
        self,
        image: Image.Image,
        stem: str,
        mask_path: str,
        padding: float,
        relative: bool,
        output_dir: Path,
    ) -> Dict[str, Any]:
        mask_file = Path(mask_path)
        if not mask_file.exists():
            raise ValueError(f"Mask file not found: {mask_path}")
        with Image.open(mask_file) as mask_image:
            mask = mask_image.convert("L").resize(image.size)
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError("Mask is empty.")
        crop_box = _apply_padding(bbox, image.size, padding, relative)
        cropped = image.crop(crop_box)
        cropped_mask = mask.crop(crop_box)
        cropped.putalpha(cropped_mask)
        output_path = output_dir / f"{stem}_mask_crop.png"
        cropped.save(output_path)
        return _record(output_path, crop_box, cropped.size)

    def _crop_polygon(
        self,
        image: Image.Image,
        stem: str,
        polygon: Optional[List[List[float]]],
        padding: float,
        relative: bool,
        output_dir: Path,
    ) -> Dict[str, Any]:
        if not polygon or len(polygon) < 3:
            raise ValueError("polygon must contain at least three [x, y] points.")
        points = _normalize_polygon(polygon, image.size, relative)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=255)
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError("Polygon crop is empty.")
        crop_box = _apply_padding(bbox, image.size, padding, relative)
        cropped = image.crop(crop_box)
        cropped_mask = mask.crop(crop_box)
        cropped.putalpha(cropped_mask)
        output_path = output_dir / f"{stem}_polygon_crop.png"
        cropped.save(output_path)
        return _record(output_path, crop_box, cropped.size)


def _normalize_box(box: Sequence[float], size: Tuple[int, int], padding: float, relative: bool) -> Tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("box must have four values: [x1, y1, x2, y2].")
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in box]
    if relative:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must satisfy x2 > x1 and y2 > y1.")
    return _apply_padding((x1, y1, x2, y2), size, padding, relative)


def _normalize_polygon(points: List[List[float]], size: Tuple[int, int], relative: bool) -> List[Tuple[float, float]]:
    width, height = size
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each polygon point must be [x, y].")
        x, y = float(point[0]), float(point[1])
        if relative:
            x, y = x * width, y * height
        normalized.append((x, y))
    return normalized


def _apply_padding(box, size: Tuple[int, int], padding: float, relative: bool) -> Tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = float(padding) * width if relative else float(padding)
    pad_y = float(padding) * height if relative else float(padding)
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Crop region is empty after clamping.")
    return x1, y1, x2, y2


def _record(output_path: Path, crop_box: Tuple[int, int, int, int], crop_size: Tuple[int, int]) -> Dict[str, Any]:
    return {
        "output_path": str(output_path),
        "box": list(crop_box),
        "crop_size": [crop_size[0], crop_size[1]],
    }
