"""
Point parsing and annotation helpers for Molmo2 integration.
"""

import base64
import io
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from PIL import Image, ImageDraw

_UNIFIED_COORD_RE = re.compile(r'<(?:points|tracks).*?coords="([0-9\t:;, .]+)"\s*/?>')
_UNIFIED_FRAME_RE = re.compile(r'(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)')
_UNIFIED_POINT_RE = re.compile(r'([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})')
_LEGACY_POINT_PATTERNS = [
    re.compile(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)"),
    re.compile(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)"),
    re.compile(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"'),
]


def default_output_dir() -> Path:
    return Path(tempfile.gettempdir()) / "spagent_molmo2"


def extract_points_from_text(
    text: str,
    image_sizes: List[Tuple[int, int]],
) -> List[Tuple[int, float, float]]:
    if len(image_sizes) == 1:
        width, height = image_sizes[0]
        points = _extract_unified_points(text, width, height)
        if points:
            return points
        return _extract_legacy_points(text, width, height)

    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    return _extract_unified_points(text, widths, heights)


def group_points_by_image(
    points: List[Tuple[int, float, float]],
    image_paths: List[str],
) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    for image_index, image_path in enumerate(image_paths, start=1):
        image_points = [
            {"x": round(x, 2), "y": round(y, 2)}
            for frame_id, x, y in points
            if frame_id == image_index
        ]
        grouped.append({
            "image_index": image_index,
            "image_path": image_path,
            "points": image_points,
        })
    return grouped


def annotate_images_as_base64(
    image_sources: List[Union[str, Image.Image]],
    grouped_points: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    annotated_images: List[Dict[str, Any]] = []
    for group in grouped_points:
        if not group["points"]:
            continue

        image_index = group["image_index"]
        image_path = group["image_path"]
        image_source = image_sources[image_index - 1]

        if isinstance(image_source, Image.Image):
            image = image_source.convert("RGB").copy()
        else:
            with Image.open(image_source) as loaded_image:
                image = loaded_image.convert("RGB")

        draw = ImageDraw.Draw(image)
        radius = max(4, int(max(image.size) * 0.01))
        for point_index, point in enumerate(group["points"], start=1):
            x = point["x"]
            y = point["y"]
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill="rgb(240, 82, 156)",
            )
            draw.text((x + radius + 2, y - radius), str(point_index), fill="white")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        annotated_images.append({
            "image_index": image_index,
            "image_path": image_path,
            "image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        })
    return annotated_images


def save_annotated_images(
    annotated_images: List[Dict[str, Any]],
    output_dir: Union[str, Path, None] = None,
) -> List[str]:
    base_dir = Path(output_dir) if output_dir else default_output_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for item in annotated_images:
        image_path = item["image_path"]
        image_index = item["image_index"]
        stem = Path(image_path).stem
        suffix = f"_img{image_index}" if len(annotated_images) > 1 else ""
        output_path = base_dir / f"{stem}{suffix}_molmo2_point.jpg"
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(item["image_base64"]))
        saved_paths.append(str(output_path))
    return saved_paths


def _extract_unified_points(
    text: str,
    widths: Union[int, List[int]],
    heights: Union[int, List[int]],
) -> List[Tuple[int, float, float]]:
    all_points: List[Tuple[int, float, float]] = []
    multi_image = isinstance(widths, (list, tuple)) and isinstance(heights, (list, tuple))

    for coord_match in _UNIFIED_COORD_RE.finditer(text):
        for frame_match in _UNIFIED_FRAME_RE.finditer(coord_match.group(1)):
            frame_id = int(float(frame_match.group(1))) if multi_image else 1
            if multi_image:
                if frame_id < 1 or frame_id > len(widths):
                    continue
                width = widths[frame_id - 1]
                height = heights[frame_id - 1]
            else:
                width = widths
                height = heights

            for point_match in _UNIFIED_POINT_RE.finditer(frame_match.group(2)):
                x = float(point_match.group(2)) / 1000.0 * width
                y = float(point_match.group(3)) / 1000.0 * height
                if 0 <= x <= width and 0 <= y <= height:
                    all_points.append((frame_id, x, y))
    return all_points


def _extract_legacy_points(text: str, width: int, height: int) -> List[Tuple[int, float, float]]:
    all_points: List[Tuple[int, float, float]] = []
    for pattern in _LEGACY_POINT_PATTERNS:
        for match in pattern.finditer(text):
            x = float(match.group(1))
            y = float(match.group(2))
            if max(x, y) > 100:
                continue
            all_points.append((1, x / 100.0 * width, y / 100.0 * height))
    return all_points
