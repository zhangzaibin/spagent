"""
Pi3XOfflineTool — drop-in replacement for Pi3XTool during RL training.

Instead of calling a remote Flask server (which may not be available during
training), this tool loads pre-computed point-cloud cache files produced by
``train/precompute_pi3x_cache.py`` and renders novel viewpoints locally using
pure numpy / matplotlib – no GPU model, no network call.

Cache format (.npz produced by precompute_pi3x_cache.py)
---------------------------------------------------------
  points       : (N, 3) float32   world-space 3-D points
  colors       : (N, 3) float32   RGB in [0, 1]
  camera_poses : (M, 4, 4) float32 camera-to-world matrices
  image_paths  : object array of str

The tool is registered under the same name (``pi3x_tool``) as the online
Pi3XTool so that the model's tool-call vocabulary and the scheduler logic
remain unchanged.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Relative imports – identical pattern to the rest of the tools directory
# ---------------------------------------------------------------------------
import sys

_TOOL_DIR = Path(__file__).resolve().parent
_SPAGENT_DIR = _TOOL_DIR.parent
_PROJECT_ROOT = _SPAGENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SPAGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_SPAGENT_DIR))

from spagent.tools.pi3x_tool import Pi3XTool, extract_scene_id
from spagent.external_experts.Pi3.pi3x_render import render_view, subsample_points

logger = logging.getLogger(__name__)


class Pi3XOfflineTool(Pi3XTool):
    """
    Offline variant of Pi3XTool that serves cached point clouds without
    calling the Pi3X model or any remote server.

    Parameters
    ----------
    cache_dir : str
        Directory that contains ``<scene_id>.npz`` files written by
        ``train/precompute_pi3x_cache.py``.  Defaults to the value of the
        environment variable ``PI3X_CACHE_DIR`` or ``dataset/pi3x_cache``.
    max_points : int
        Subsample threshold for rendering (default 1 500 000).
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_points: int = 1_500_000,
    ):
        # Initialise as mock (no client needed) so the base class doesn't
        # try to connect to a server.
        super().__init__(use_mock=True)
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("PI3X_CACHE_DIR", "dataset/pi3x_cache")
        )
        self.max_points = max_points
        # Override use_mock flag so _validate_and_fix_tool_calls in the
        # scheduler can distinguish offline vs mock for logging purposes.
        self.use_mock = False
        logger.info(f"Pi3XOfflineTool initialised (cache_dir={self.cache_dir})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cache(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Load .npz cache for a scene. Returns None if not found."""
        cache_file = self.cache_dir / f"{scene_id}.npz"
        if not cache_file.exists():
            return None
        try:
            data = np.load(cache_file, allow_pickle=True)
            return {
                "points": data["points"].astype(np.float32),
                "colors": data["colors"].astype(np.float32),
                "camera_poses": data["camera_poses"].astype(np.float32),
            }
        except Exception as e:
            logger.error(f"Failed to load cache {cache_file}: {e}")
            return None

    def _save_image_from_b64(
        self,
        b64: str,
        image_path: str,
        azimuth_angle: float,
        elevation_angle: float,
    ) -> Optional[str]:
        """Decode base64 PNG and save it to the outputs directory."""
        try:
            scene_id = extract_scene_id(image_path)
            output_dir = _PROJECT_ROOT / "outputs"
            output_dir.mkdir(exist_ok=True)
            filename = f"pi3x_{scene_id}_azim{azimuth_angle:.1f}_elev{elevation_angle:.1f}.png"
            out_path = output_dir / filename
            img_data = base64.b64decode(b64)
            with open(out_path, "wb") as f:
                f.write(img_data)
            logger.info(f"Saved rendered image: {out_path}")
            return str(out_path)
        except Exception as e:
            logger.error(f"Failed to save rendered image: {e}")
            return None

    # ------------------------------------------------------------------
    # Public API (overrides Pi3XTool.call)
    # ------------------------------------------------------------------

    def call(
        self,
        image_path: List[str],
        azimuth_angle: float = 0,
        elevation_angle: float = 0,
        rotation_reference_camera: int = 1,
        camera_view: bool = False,
    ) -> Dict[str, Any]:
        """
        Render a novel viewpoint from the pre-computed point-cloud cache.

        Accepts the exact same signature as Pi3XTool.call() so that the
        scheduler (plugin.py / plugin_all_angles.py) needs no changes.
        """
        # --- Validate inputs ---
        if not image_path:
            return {"success": False, "error": "image_path list is required and cannot be empty"}

        try:
            azimuth_angle = float(azimuth_angle)
            elevation_angle = float(elevation_angle)
        except (ValueError, TypeError) as e:
            return {"success": False, "error": f"Invalid angle values: {e}"}

        if not -180 <= azimuth_angle <= 180:
            return {"success": False, "error": "azimuth_angle must be between -180 and 180"}
        if not -90 <= elevation_angle <= 90:
            return {"success": False, "error": "elevation_angle must be between -90 and 90"}

        # --- Check disk render cache (output PNG from a previous run) ---
        cached = self._check_cache(
            image_path[0], azimuth_angle, elevation_angle,
            rotation_reference_camera=rotation_reference_camera,
            camera_view=camera_view,
        )
        if cached:
            logger.info(
                f"Using render cache for azimuth={azimuth_angle}°, "
                f"elevation={elevation_angle}°"
            )
            return cached

        # --- Load point-cloud cache ---
        scene_id = extract_scene_id(image_path[0])
        pc_data = self._load_cache(scene_id)
        if pc_data is None:
            msg = (
                f"No cached point cloud found for scene '{scene_id}'. "
                f"Run train/precompute_pi3x_cache.py first. "
                f"(looked in {self.cache_dir})"
            )
            logger.error(msg)
            return {"success": False, "error": msg}

        points = pc_data["points"]
        colors = pc_data["colors"]
        camera_poses = pc_data["camera_poses"]
        n_cameras = camera_poses.shape[0]

        logger.info(
            f"Rendering scene '{scene_id}': {len(points):,} points, "
            f"{n_cameras} cameras, "
            f"azim={azimuth_angle}°, elev={elevation_angle}°, "
            f"ref_cam={rotation_reference_camera}, cam_view={camera_view}"
        )

        # --- Render ---
        try:
            b64 = render_view(
                points, colors, camera_poses,
                azimuth_angle=azimuth_angle,
                elevation_angle=elevation_angle,
                rotation_reference_camera=rotation_reference_camera,
                camera_view=camera_view,
                max_points=self.max_points,
            )
        except Exception as e:
            logger.error(f"Rendering failed for scene '{scene_id}': {e}")
            return {"success": False, "error": f"Rendering failed: {e}"}

        # --- Save PNG and return ---
        output_path = self._save_image_from_b64(
            b64, image_path[0], azimuth_angle, elevation_angle
        )

        return {
            "success": True,
            "result": {
                "success": True,
                "ply_filename": f"cached_{scene_id}.ply",
                "points_count": len(points),
                "camera_views": [{
                    "camera": max(1, min(int(rotation_reference_camera), n_cameras)),
                    "view": f"custom_azim_{int(azimuth_angle)}_elev_{int(elevation_angle)}",
                    "azimuth_angle": azimuth_angle,
                    "elevation_angle": elevation_angle,
                    "image": b64,
                }],
            },
            "points_count": len(points),
            "ply_filename": f"cached_{scene_id}.ply",
            "view_count": 1,
            "azimuth_angle": azimuth_angle,
            "elevation_angle": elevation_angle,
            "view_type": "custom_angle",
            "input_images_count": len(image_path),
            "output_path": output_path,
            "description": (
                f"Pi3XOfflineTool rendered a point cloud visualisation from cache "
                f"(scene='{scene_id}', {len(points):,} points, {n_cameras} cameras). "
                f"Viewing angle: azimuth={azimuth_angle}°, elevation={elevation_angle}°. "
                f"Camera positions are shown as cone-shaped markers numbered from cam1."
            ),
        }
