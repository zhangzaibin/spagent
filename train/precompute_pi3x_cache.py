#!/usr/bin/env python3
"""
precompute_pi3x_cache.py — offline Pi3X point-cloud pre-computation.

For each unique scene referenced in one or more RL training JSONL files,
run Pi3X inference once and cache the filtered point cloud + camera poses
as a compressed .npz file.  Training can then load the cache without any
GPU model or network calls.

Usage
-----
python train/precompute_pi3x_cache.py \\
    --dataset dataset/crossviewQA_train_rl_fixed.jsonl \\
    --cache-dir dataset/pi3x_cache \\
    --checkpoint checkpoints/pi3x/model.safetensors \\
    [--gpu 0] [--workers 1]

The script is idempotent: already-cached scenes are skipped automatically,
so it is safe to re-run after a crash.

Cache format  (<cache_dir>/<scene_id>.npz)
------------------------------------------
  points      : (N, 3)  float32  – filtered 3-D world points
  colors      : (N, 3)  float32  – RGB in [0, 1]
  camera_poses: (M, 4, 4) float32 – camera-to-world matrices
  image_paths : object array of str – original image paths for this scene
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – find the project root (train/../) so we can import from spagent
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_PI3_DIR = _PROJECT_ROOT / "spagent" / "external_experts" / "Pi3"
if str(_PI3_DIR) not in sys.path:
    sys.path.insert(0, str(_PI3_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("precompute_pi3x")


# ---------------------------------------------------------------------------
# Scene collection from JSONL datasets
# ---------------------------------------------------------------------------

def collect_scenes(jsonl_paths: List[str]) -> Dict[str, List[str]]:
    """
    Return a dict: scene_id -> list[image_path].

    A "scene" is identified by the scene_id derived from the first image path
    (using the same extract_scene_id() used in pi3x_tool.py).
    """
    from spagent.tools.pi3x_tool import extract_scene_id

    scenes: Dict[str, List[str]] = {}
    for jsonl_path in jsonl_paths:
        if not os.path.exists(jsonl_path):
            logger.warning(f"Dataset file not found, skipping: {jsonl_path}")
            continue
        logger.info(f"Reading {jsonl_path} …")
        with open(jsonl_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"  Line {lineno}: JSON parse error – {e}")
                    continue

                # Support both 'image' (str | list) and 'images' fields
                raw = item.get("images") or item.get("image")
                if raw is None:
                    continue
                if isinstance(raw, str):
                    image_paths = [raw]
                else:
                    image_paths = list(raw)

                if not image_paths:
                    continue

                scene_id = extract_scene_id(image_paths[0])
                if scene_id not in scenes:
                    scenes[scene_id] = image_paths

    logger.info(f"Found {len(scenes)} unique scenes across {len(jsonl_paths)} dataset(s).")
    return scenes


# ---------------------------------------------------------------------------
# Pi3X inference + filtering (mirrors pi3x_server.py logic)
# ---------------------------------------------------------------------------

def _run_inference(model, image_paths: List[str], conf_threshold: float = 0.06, rtol: float = 0.02):
    """
    Load images, run Pi3X inference, apply conf/depth-edge/outlier filters.

    Returns
    -------
    points_filtered : (N, 3) np.float32
    colors_filtered : (N, 3) np.float32
    camera_poses    : (M, 4, 4) np.float32
    """
    import cv2
    import torch
    from pi3.utils.geometry import depth_edge
    from spagent.external_experts.Pi3.pi3x_server import remove_outliers_mahalanobis

    device = next(model.parameters()).device
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    images_bgr, images_rgb = [], []
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {p}")
        h, w = img_bgr.shape[:2]
        patch = 14
        new_h = ((h + patch - 1) // patch) * patch
        new_w = ((w + patch - 1) // patch) * patch
        if new_h != h or new_w != w:
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images_bgr.append(img_bgr)
        images_rgb.append(img_rgb)

    imgs_t = torch.stack([
        torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        for img in images_bgr
    ]).to(device)
    imgs_rgb_t = torch.stack([
        torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        for img in images_rgb
    ]).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            results = model(imgs=imgs_t[None])

    masks = torch.sigmoid(results['conf'][..., 0]) > conf_threshold
    non_edge = ~depth_edge(results['local_points'][..., 2], rtol=rtol)
    masks = torch.logical_and(masks, non_edge)[0]

    points_f = results['points'][0][masks].cpu().float()
    colors_f = imgs_rgb_t.permute(0, 2, 3, 1)[masks].cpu().float()

    # Convert to float32 numpy (bfloat16 can cause issues with np.isfinite)
    pts_np = points_f.float().numpy() if hasattr(points_f, 'numpy') else np.array(points_f, dtype=np.float32)
    col_np = colors_f.float().numpy() if hasattr(colors_f, 'numpy') else np.array(colors_f, dtype=np.float32)

    # Drop any residual NaN / Inf values before outlier removal
    valid = np.isfinite(pts_np).all(axis=1)
    pts_np = pts_np[valid]
    col_np = col_np[valid]

    if len(pts_np) > 10:
        pts_np, col_np, _ = remove_outliers_mahalanobis(pts_np, col_np, threshold_std=3.0)
    else:
        logger.warning(f"  Only {len(pts_np)} valid points after NaN filter, skipping outlier removal")
    if not isinstance(pts_np, np.ndarray):
        pts_np = np.array(pts_np)
    if not isinstance(col_np, np.ndarray):
        col_np = np.array(col_np)

    camera_poses = results['camera_poses'][0].cpu().numpy()

    return pts_np.astype(np.float32), col_np.astype(np.float32), camera_poses.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Pi3X point-cloud cache for RL training datasets."
    )
    parser.add_argument(
        "--dataset", "-d", action="append", required=True, metavar="JSONL",
        help="Path to a training JSONL file (may be repeated for multiple files).",
    )
    parser.add_argument(
        "--cache-dir", default="dataset/pi3x_cache",
        help="Directory where .npz cache files will be written (default: dataset/pi3x_cache).",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/pi3x/model.safetensors",
        help="Path to Pi3X model.safetensors (default: checkpoints/pi3x/model.safetensors).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.06,
        help="Confidence threshold for point-cloud filtering (default: 0.06).",
    )
    parser.add_argument(
        "--rtol", type=float, default=0.02,
        help="Depth-edge relative tolerance (default: 0.02).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just print the scenes that would be processed, do not run inference.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect scenes ---
    scenes = collect_scenes(args.dataset)
    if not scenes:
        logger.error("No scenes found. Check that --dataset points to valid JSONL files.")
        sys.exit(1)

    to_process = {
        sid: paths
        for sid, paths in scenes.items()
        if not (cache_dir / f"{sid}.npz").exists()
    }
    skipped = len(scenes) - len(to_process)
    logger.info(
        f"Scenes: {len(scenes)} total, {skipped} already cached, {len(to_process)} to compute."
    )

    if args.dry_run:
        for sid, paths in sorted(to_process.items()):
            print(f"  {sid}  ({len(paths)} images)  ->  {cache_dir / (sid + '.npz')}")
        return

    if not to_process:
        logger.info("All scenes already cached. Nothing to do.")
        return

    # --- Load model ---
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        logger.error(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Download it with:\n"
            f"  mkdir -p checkpoints/pi3x\n"
            f"  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \\\n"
            f"    wget https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors \\\n"
            f"    -O {checkpoint_path}"
        )
        sys.exit(1)

    import torch
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    # load_model() is defined in pi3x_server.py
    from spagent.external_experts.Pi3.pi3x_server import load_model
    if not load_model(checkpoint_path):
        logger.error("Pi3X model failed to load. Aborting.")
        sys.exit(1)

    import spagent.external_experts.Pi3.pi3x_server as _srv
    model = _srv.model

    # --- Process scenes ---
    done = 0
    failed = []
    total = len(to_process)
    for idx, (scene_id, image_paths) in enumerate(to_process.items(), 1):
        cache_file = cache_dir / f"{scene_id}.npz"
        logger.info(f"[{idx}/{total}] Scene '{scene_id}'  ({len(image_paths)} images) …")

        # Validate image files exist
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            logger.warning(f"  Skipping – missing images: {missing}")
            failed.append((scene_id, f"missing images: {missing}"))
            continue

        try:
            pts, cols, cam_poses = _run_inference(
                model, image_paths,
                conf_threshold=args.conf_threshold,
                rtol=args.rtol,
            )
            np.savez_compressed(
                cache_file,
                points=pts,
                colors=cols,
                camera_poses=cam_poses,
                image_paths=np.array(image_paths, dtype=object),
            )
            logger.info(
                f"  Saved {cache_file.name}  "
                f"({pts.shape[0]:,} points, {cam_poses.shape[0]} cameras)"
            )
            done += 1
        except Exception as e:
            logger.error(f"  FAILED for scene '{scene_id}': {e}")
            failed.append((scene_id, str(e)))

    logger.info(f"\nDone. {done}/{total} scenes cached, {len(failed)} failed.")
    if failed:
        logger.warning("Failed scenes:")
        for sid, reason in failed:
            logger.warning(f"  {sid}: {reason}")


if __name__ == "__main__":
    main()
