"""Non-interactive LingBot-Map inference and artifact export."""

import argparse
import importlib.util
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


def _load_official_demo(repo_path: Path):
    demo_path = repo_path / "demo.py"
    if not demo_path.is_file():
        raise FileNotFoundError(f"LingBot-Map demo.py not found: {demo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    spec = importlib.util.spec_from_file_location("spagent_lingbot_map_demo", demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load LingBot-Map demo: {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_args(args) -> Namespace:
    return Namespace(
        mode="streaming",
        image_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=max(1024, args.num_frames),
        kv_cache_sliding_window=64,
        num_scale_frames=min(8, args.num_frames),
        use_sdpa=bool(args.use_sdpa),
        camera_num_iterations=max(1, int(args.camera_num_iterations)),
        model_path=str(args.model_path),
    )


def run_inference(args) -> Dict:
    repo_path = Path(args.repo_path).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    frame_dir = Path(args.image_folder).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"LingBot-Map checkpoint not found: {model_path}")

    demo = _load_official_demo(repo_path)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("LingBot-Map inference requires CUDA")

    images, paths, resolved_folder = demo.load_images(
        image_folder=str(frame_dir),
        image_ext=".png",
        image_size=518,
        patch_size=14,
    )
    if len(paths) < 8:
        raise ValueError("LingBot-Map requires at least 8 frames")
    args.num_frames = len(paths)

    device = torch.device("cuda")
    model = demo.load_model(_model_args(args), device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    images_device = images.to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            images_device,
            num_scale_frames=min(8, len(paths)),
            keyframe_interval=1,
            output_device=torch.device("cpu"),
        )

    predictions, images_cpu = demo.postprocess(predictions, predictions["images"])
    visual_predictions = demo.prepare_for_visualization(predictions, images_cpu)
    return export_artifacts(
        predictions=visual_predictions,
        image_paths=paths,
        output_dir=output_dir,
        checkpoint_path=model_path,
        mask_sky=bool(args.mask_sky),
        image_folder=resolved_folder,
        confidence_threshold=float(args.confidence_threshold),
        point_stride=max(1, int(args.point_stride)),
        max_points=max(1, int(args.max_points)),
    )


def export_artifacts(
    predictions: Dict,
    image_paths: List[str],
    output_dir: Path,
    checkpoint_path: Path,
    mask_sky: bool = False,
    image_folder: Optional[str] = None,
    confidence_threshold: float = 1.5,
    point_stride: int = 10,
    max_points: int = 500_000,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    point_source = "world_points"
    points_value = predictions.get("world_points")
    if points_value is None:
        depth = predictions.get("depth")
        extrinsic = predictions.get("extrinsic")
        intrinsic = predictions.get("intrinsic")
        if depth is None or extrinsic is None or intrinsic is None:
            raise ValueError(
                f"Predictions lack point/depth reconstruction fields; keys={sorted(predictions)}"
            )
        from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

        world_to_camera = _camera_to_world_to_world_to_camera(extrinsic)
        points_value = unproject_depth_map_to_point_map(depth, world_to_camera, intrinsic)
        point_source = "depth_unprojection"
    points = np.asarray(points_value)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Invalid world_points shape: {points.shape}")

    confidence = predictions.get("world_points_conf")
    if confidence is None:
        confidence = predictions.get("depth_conf")
    confidence = np.ones(points.shape[:-1], dtype=np.float32) if confidence is None else np.asarray(confidence)
    if confidence.shape != points.shape[:-1]:
        raise ValueError(
            f"Confidence shape {confidence.shape} does not match world_points {points.shape[:-1]}"
        )

    images = _images_nhwc(predictions.get("images"), points.shape[:-1])
    if mask_sky:
        from lingbot_map.vis.sky_segmentation import apply_sky_segmentation

        confidence = apply_sky_segmentation(
            confidence,
            image_folder=image_folder,
            image_paths=image_paths,
            images=images,
            sky_mask_dir=str(output_dir / "sky_masks"),
        )

    flat_points = points.reshape(-1, 3)
    flat_colors = images.reshape(-1, 3)
    flat_confidence = confidence.reshape(-1)
    sampled = np.arange(0, len(flat_points), max(1, point_stride))
    flat_points = flat_points[sampled]
    flat_colors = flat_colors[sampled]
    flat_confidence = flat_confidence[sampled]

    finite = np.isfinite(flat_points).all(axis=1) & np.isfinite(flat_confidence)
    positive = finite & (flat_confidence > 1e-5)
    keep = positive & (flat_confidence > confidence_threshold)
    if not np.any(keep):
        keep = positive
    flat_points = flat_points[keep].astype(np.float32, copy=False)
    flat_colors = _colors_uint8(flat_colors[keep])
    if len(flat_points) == 0:
        raise ValueError("LingBot-Map produced no finite positive-confidence points")

    if len(flat_points) > max_points:
        selected = np.linspace(0, len(flat_points) - 1, max_points, dtype=np.int64)
        flat_points = flat_points[selected]
        flat_colors = flat_colors[selected]

    point_cloud_path = output_dir / "point_cloud.ply"
    trajectory_path = output_dir / "trajectory.json"
    preview_path = output_dir / "preview.png"
    metadata_path = output_dir / "reconstruction_metadata.json"
    _write_binary_ply(point_cloud_path, flat_points, flat_colors)
    _write_trajectory(
        trajectory_path,
        predictions.get("extrinsic"),
        predictions.get("intrinsic"),
        image_paths,
    )
    Image.fromarray(_colors_uint8(images[0])).save(preview_path)

    metadata = {
        "points_count": int(len(flat_points)),
        "num_frames": int(points.shape[0]),
        "point_stride": int(point_stride),
        "confidence_threshold": float(confidence_threshold),
        "mask_sky": bool(mask_sky),
        "point_source": point_source,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "point_cloud_path": str(point_cloud_path),
        "trajectory_path": str(trajectory_path),
        "preview_path": str(preview_path),
        "created_at": time.time(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _camera_to_world_to_world_to_camera(extrinsic) -> np.ndarray:
    camera_to_world = np.asarray(extrinsic)
    if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (3, 4):
        raise ValueError(f"Invalid camera-to-world extrinsic shape: {camera_to_world.shape}")
    homogeneous = np.broadcast_to(
        np.eye(4, dtype=camera_to_world.dtype),
        (len(camera_to_world), 4, 4),
    ).copy()
    homogeneous[:, :3, :4] = camera_to_world
    return np.linalg.inv(homogeneous)[:, :3, :4]


def _images_nhwc(value, expected_shape) -> np.ndarray:
    images = np.asarray(value)
    if images.ndim != 4:
        raise ValueError(f"Invalid images shape: {images.shape}")
    if images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
    if images.shape[:3] != expected_shape or images.shape[-1] != 3:
        raise ValueError(f"Images shape {images.shape} does not match points {expected_shape}")
    return images


def _colors_uint8(colors: np.ndarray) -> np.ndarray:
    colors = np.nan_to_num(np.asarray(colors), nan=0.0, posinf=1.0, neginf=0.0)
    if colors.dtype == np.uint8:
        return colors
    if colors.size and float(np.max(colors)) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8)


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def _write_trajectory(path: Path, extrinsic, intrinsic, image_paths: List[str]) -> None:
    extrinsic_array = np.asarray(extrinsic)
    intrinsic_array = np.asarray(intrinsic)
    if extrinsic_array.ndim != 3 or extrinsic_array.shape[1:] != (3, 4):
        raise ValueError(f"Invalid extrinsic shape: {extrinsic_array.shape}")
    if intrinsic_array.ndim != 3 or intrinsic_array.shape[1:] != (3, 3):
        raise ValueError(f"Invalid intrinsic shape: {intrinsic_array.shape}")
    if len(extrinsic_array) != len(image_paths) or len(intrinsic_array) != len(image_paths):
        raise ValueError("Camera prediction count does not match input frame count")

    frames = []
    for index, image_path in enumerate(image_paths):
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :4] = extrinsic_array[index]
        frames.append(
            {
                "frame_index": index,
                "image_path": str(image_path),
                "camera_to_world": camera_to_world.tolist(),
                "intrinsic": intrinsic_array[index].tolist(),
            }
        )
    path.write_text(
        json.dumps({"convention": "camera_to_world", "frames": frames}, indent=2),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LingBot-Map reconstruction artifacts")
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_num_iterations", type=int, default=1)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--point_stride", type=int, default=10)
    parser.add_argument("--max_points", type=int, default=500_000)
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--mask_sky", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    parsed.num_frames = len(list(Path(parsed.image_folder).glob("*.png")))
    result = run_inference(parsed)
    print(json.dumps(result, indent=2))
