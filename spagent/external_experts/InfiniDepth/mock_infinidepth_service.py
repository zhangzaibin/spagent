"""Deterministic mock service for InfiniDepth tests."""

from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Optional

from PIL import Image


class MockInfiniDepthService:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(gettempdir()) / "spagent_infinidepth"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def infer(
        self,
        image_path: str,
        save_pcd: bool = False,
        upsample_ratio: float = 2,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"Image file not found: {image_path}"}
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(path) as image:
            width, height = image.size
        depth = _gradient_depth(width, height)
        colored = _colored_depth(width, height)

        depth_path = out_dir / f"{path.stem}_infinidepth_depth.png"
        colored_path = out_dir / f"{path.stem}_infinidepth_colored.png"
        depth.save(depth_path)
        colored.save(colored_path)

        point_cloud_path = None
        if save_pcd:
            point_cloud_path = out_dir / f"{path.stem}_infinidepth.ply"
            _write_mock_ply(point_cloud_path)

        return {
            "success": True,
            "model": "mock-infinidepth",
            "depth_path": str(depth_path),
            "colored_depth_path": str(colored_path),
            "point_cloud_path": str(point_cloud_path) if point_cloud_path else None,
            "output_dir": str(out_dir),
            "shape": [height, width],
            "upsample_ratio": upsample_ratio,
        }


def _gradient_depth(width: int, height: int) -> Image.Image:
    image = Image.new("L", (width, height))
    pixels = image.load()
    for y in range(height):
        value = int(255 * y / max(1, height - 1))
        for x in range(width):
            pixels[x, y] = value
    return image


def _colored_depth(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            r = int(255 * x / max(1, width - 1))
            g = int(255 * y / max(1, height - 1))
            b = 255 - g
            pixels[x, y] = (r, g, b)
    return image


def _write_mock_ply(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 0 0.5",
                "0 1 1.0",
            ]
        )
        + "\n"
    )
