"""Deterministic WildDet3D mock service for tests and local development."""

from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw


class MockWildDet3DService:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(gettempdir()) / "spagent_wilddet3d"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def infer(
        self,
        image_path: str,
        text_prompt: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
        score_threshold: float = 0.3,
        save_visualization: bool = True,
    ) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"Image file not found: {image_path}"}

        class_names = _class_names(text_prompt, boxes, points)
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            boxes_2d = _mock_boxes(width, height, boxes, points, len(class_names))
            boxes_3d = [_mock_box3d(box, idx) for idx, box in enumerate(boxes_2d)]
            scores = [round(max(float(score_threshold), 0.3) + 0.05 * (idx % 3), 3) for idx in range(len(boxes_2d))]

            output_path = None
            if save_visualization:
                output_path = str(self.output_dir / f"{path.stem}_wilddet3d_mock.png")
                _draw_visualization(image, boxes_2d, class_names, scores).save(output_path)

            depth_path = str(self.output_dir / f"{path.stem}_wilddet3d_depth_mock.png")
            _mock_depth(width, height).save(depth_path)

        return {
            "success": True,
            "model": "mock-wilddet3d",
            "boxes_2d": boxes_2d,
            "boxes_3d": boxes_3d,
            "scores": scores,
            "class_names": class_names,
            "depth_path": depth_path,
            "output_path": output_path,
        }


def _class_names(text_prompt, boxes, points) -> List[str]:
    if text_prompt and text_prompt.strip():
        names = [part.strip() for part in text_prompt.split(",") if part.strip()]
        return names or ["object"]
    count = len(boxes or points or [None])
    return ["object"] * count


def _mock_boxes(width: int, height: int, boxes, points, count: int) -> List[List[float]]:
    if boxes:
        return [[float(v) for v in box] for box in boxes]
    if points:
        out = []
        box_w = max(16, width // 4)
        box_h = max(16, height // 4)
        for x, y, _label in points:
            out.append(
                [
                    max(0.0, float(x) - box_w / 2),
                    max(0.0, float(y) - box_h / 2),
                    min(float(width), float(x) + box_w / 2),
                    min(float(height), float(y) + box_h / 2),
                ]
            )
        return out
    out = []
    for idx in range(max(1, count)):
        offset = idx * min(width, height) * 0.08
        out.append(
            [
                width * 0.18 + offset,
                height * 0.18 + offset,
                width * 0.68 + offset,
                height * 0.72 + offset,
            ]
        )
    return out


def _mock_box3d(box: List[float], idx: int) -> List[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return [round(cx, 3), round(cy, 3), round(2.0 + idx, 3), round(x2 - x1, 3), round(y2 - y1, 3), 1.5, 0.0]


def _draw_visualization(image: Image.Image, boxes: List[List[float]], class_names: List[str], scores: List[float]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = ["red", "lime", "cyan", "yellow", "magenta"]
    for idx, box in enumerate(boxes):
        color = colors[idx % len(colors)]
        label = class_names[idx] if idx < len(class_names) else class_names[-1]
        score = scores[idx] if idx < len(scores) else 0.0
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 3, max(0, box[1] - 14)), f"{label} {score:.2f}", fill=color)
    return canvas


def _mock_depth(width: int, height: int) -> Image.Image:
    image = Image.new("L", (width, height))
    pixels = image.load()
    for y in range(height):
        value = int(255 * y / max(1, height - 1))
        for x in range(width):
            pixels[x, y] = value
    return image
