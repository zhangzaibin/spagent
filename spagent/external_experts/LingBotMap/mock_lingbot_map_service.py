import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MockLingBotMapService:
    """Deterministic local mock for LingBot-Map mapping outputs."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "spagent_lingbot_map"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "model_loaded": True, "backend": "mock"}

    def infer(
        self,
        image_folder: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        mask_sky: bool = False,
        keyframe_interval: int = 1,
        max_frames: int = 128,
        output_dir: Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> Dict[str, Any]:
        frames = self._collect_frames(image_folder=image_folder, image_paths=image_paths)
        if not frames:
            return {"success": False, "error": "No input images found"}

        frames = frames[:: max(1, int(keyframe_interval))][: max(1, int(max_frames))]
        run_dir = Path(output_dir) if output_dir else self.output_dir / f"mock_{int(time.time() * 1000)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        preview_path = run_dir / "lingbot_map_preview.png"
        trajectory_path = run_dir / "trajectory.json"
        point_cloud_path = run_dir / "point_cloud.ply"

        self._write_preview(frames, preview_path, mask_sky=mask_sky)
        self._write_trajectory(frames, trajectory_path)
        self._write_point_cloud(point_cloud_path)

        return {
            "success": True,
            "output_dir": str(run_dir),
            "preview_path": str(preview_path),
            "trajectory_path": str(trajectory_path),
            "point_cloud_path": str(point_cloud_path),
            "viewer_url": "http://127.0.0.1:8080",
            "num_frames": len(frames),
            "mask_sky": bool(mask_sky),
            "wait_for_completion": bool(wait_for_completion),
            "command": ["mock_lingbot_map"],
        }

    def _collect_frames(self, image_folder: Optional[str], image_paths: Optional[List[str]]) -> List[Path]:
        if image_paths:
            return [Path(p) for p in image_paths if Path(p).exists()]
        if image_folder:
            folder = Path(image_folder)
            if folder.exists():
                return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()])
        return []

    @staticmethod
    def _write_preview(frames: List[Path], output_path: Path, mask_sky: bool) -> None:
        thumbs = []
        for frame in frames[:6]:
            with Image.open(frame) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((180, 120))
                tile = Image.new("RGB", (180, 120), (245, 245, 245))
                tile.paste(thumb, ((180 - thumb.width) // 2, (120 - thumb.height) // 2))
                thumbs.append(tile)
        if not thumbs:
            thumbs = [Image.new("RGB", (180, 120), (220, 220, 220))]

        width = 360
        height = 260
        canvas = Image.new("RGB", (width, height), (28, 31, 36))
        draw = ImageDraw.Draw(canvas)
        for idx, thumb in enumerate(thumbs[:4]):
            x = 12 + (idx % 2) * 184
            y = 12 + (idx // 2) * 124
            canvas.paste(thumb, (x, y))
            draw.rectangle([x, y, x + 179, y + 119], outline=(80, 190, 255), width=2)
        draw.rectangle([18, 214, 342, 246], fill=(245, 245, 245))
        label = f"LingBot-Map mock | frames={len(frames)} | mask_sky={mask_sky}"
        draw.text((28, 224), label, fill=(20, 20, 20))
        canvas.save(output_path)

    @staticmethod
    def _write_trajectory(frames: List[Path], output_path: Path) -> None:
        trajectory = [
            {"frame": frame.name, "pose": [[1, 0, 0, idx * 0.05], [0, 1, 0, 0], [0, 0, 1, 1.0], [0, 0, 0, 1]]}
            for idx, frame in enumerate(frames)
        ]
        output_path.write_text(json.dumps({"trajectory": trajectory}, indent=2), encoding="utf-8")

    @staticmethod
    def _write_point_cloud(output_path: Path) -> None:
        points = [
            (-0.5, -0.5, 1.0, 255, 80, 80),
            (0.5, -0.5, 1.1, 80, 255, 80),
            (0.5, 0.5, 1.2, 80, 80, 255),
            (-0.5, 0.5, 1.1, 255, 255, 80),
        ]
        header = (
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        body = "\n".join(f"{x} {y} {z} {r} {g} {b}" for x, y, z, r, g, b in points)
        output_path.write_text(header + body + "\n", encoding="utf-8")
