"""
Interactive Open3D GUI for a Pi3 PLY plus reconstructed cameras (GUI manipulation demo).

**Typical flow (Pi3 HTTP server + this script)**

1. **First time — call server, save PLY + JSON, then open GUI**  
   Uses ``Pi3Client`` against ``pi3_server`` ``/infer`` (same as elsewhere in the repo).
   Omit ``--ply`` to write under ``--out-dir`` (default ``outputs``) using the server’s
   ``ply_filename``, plus ``<stem>_cameras.json``::

    python gui/gui_manipulation/pi3_pointcloud_interactive.py \\
        --fetch -f outputs/gui_demo_image.txt --server http://localhost:20021

   Or set an explicit PLY path (created if missing)::

    python gui/gui_manipulation/pi3_pointcloud_interactive.py \\
        --ply outputs/result_front_125.ply \\
        --fetch -f outputs/gui_demo_image.txt --server http://localhost:20021

2. **Later — local files only, no server**  
   If the PLY and sidecar JSON already exist, run **without** ``--fetch``; nothing
   contacts the network::

    python gui/gui_manipulation/pi3_pointcloud_interactive.py \\
        --ply outputs/result_front_125.ply

3. **If you pass ``--fetch`` but files are already there**  
   By default the script **skips** ``/infer`` and opens the GUI. Use ``--force-fetch``
   to always re-run inference.

**Viewer behavior:** loads ``*_cameras.json`` (from Pi3 ``/infer``). Ego mode rotates the
point cloud and camera frustums at the reference camera; Global mode orbits the view.

Requires: ``pip install open3d`` (not pinned in this repo by default).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


def _c2w_from_position_angles(cam: Dict[str, Any]) -> np.ndarray:
    """
    Build an approximate camera-to-world from Pi3 server metadata if ``c2w`` is absent.

    The server provides:
    - position (camera center in world)
    - azimuth_angle / elevation_angle (relative to camera 1), extracted via 'yx' euler

    Without camera1 absolute rotation we approximate camera1 as identity and compose:
        R = R_y(az) * R_x(el)
    """
    from scipy.spatial.transform import Rotation as R

    t = np.asarray(cam.get("position", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    az = float(cam.get("azimuth_angle", 0.0))
    el = float(cam.get("elevation_angle", 0.0))
    r = R.from_euler("yx", [az, el], degrees=True).as_matrix()
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r
    c2w[:3, 3] = t
    return c2w


def _load_c2w_list(cameras: List[Dict[str, Any]]) -> np.ndarray:
    """Return (N, 4, 4) camera-to-world; use ``c2w`` if present, else approximate."""
    mats = []
    for cam in cameras:
        c2w = cam.get("c2w")
        if c2w is not None:
            m = np.asarray(c2w, dtype=np.float64)
            if m.shape == (4, 4):
                mats.append(m)
                continue
        mats.append(_c2w_from_position_angles(cam))
    if not mats:
        raise ValueError("camera_poses is empty")
    return np.stack(mats, axis=0)


def ego_rotate_points_world(points: np.ndarray, c2w_ref: np.ndarray, az_deg: float, el_deg: float) -> np.ndarray:
    """World-space rotation around reference camera axes (Pi3 ego-style)."""
    from scipy.spatial.transform import Rotation as R

    anchor = np.asarray(c2w_ref[:3, 3], dtype=np.float64)
    r_cw = c2w_ref[:3, :3]
    y_axis_w = r_cw[:, 1]
    x_axis_w = r_cw[:, 0]
    r_azim = (
        R.from_rotvec(np.radians(az_deg) * y_axis_w).as_matrix()
        if abs(az_deg) > 1e-6
        else np.eye(3)
    )
    r_elev = (
        R.from_rotvec(np.radians(el_deg) * x_axis_w).as_matrix()
        if abs(el_deg) > 1e-6
        else np.eye(3)
    )
    r_rel = r_elev @ r_azim
    return (r_rel @ (points - anchor).T).T + anchor


def ego_world_rotation(c2w_ref: np.ndarray, az_deg: float, el_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_rel, anchor) for the ego world rotation."""
    from scipy.spatial.transform import Rotation as R

    anchor = np.asarray(c2w_ref[:3, 3], dtype=np.float64)
    r_cw = c2w_ref[:3, :3]
    y_axis_w = r_cw[:, 1]
    x_axis_w = r_cw[:, 0]
    r_azim = (
        R.from_rotvec(np.radians(az_deg) * y_axis_w).as_matrix()
        if abs(az_deg) > 1e-6
        else np.eye(3)
    )
    r_elev = (
        R.from_rotvec(np.radians(el_deg) * x_axis_w).as_matrix()
        if abs(el_deg) > 1e-6
        else np.eye(3)
    )
    r_rel = r_elev @ r_azim
    return r_rel, anchor


def make_camera_frustum_lineset(
    c2w: np.ndarray,
    *,
    fov_deg: float,
    aspect: float,
    length: float,
    color: np.ndarray,
) -> "o3d.geometry.LineSet":
    import open3d as o3d

    o = np.asarray(c2w[:3, 3], dtype=np.float64)
    r = np.asarray(c2w[:3, :3], dtype=np.float64)

    fov = np.radians(float(fov_deg))
    half_h = np.tan(0.5 * fov) * float(length)
    half_w = float(aspect) * half_h

    p0 = np.zeros(3)  # center
    p1 = np.array([+half_w, +half_h, float(length)])
    p2 = np.array([-half_w, +half_h, float(length)])
    p3 = np.array([-half_w, -half_h, float(length)])
    p4 = np.array([+half_w, -half_h, float(length)])

    def cam_to_world(p: np.ndarray) -> np.ndarray:
        return o + r @ p

    pts = np.stack([cam_to_world(p) for p in [p0, p1, p2, p3, p4]], axis=0)
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    c = np.asarray(color, dtype=np.float64).reshape(3)
    c = np.clip(c, 0.0, 1.0)
    ls.colors = o3d.utility.Vector3dVector(np.tile(c[None, :], (len(lines), 1)))
    return ls


def merge_linesets(lsets: List["o3d.geometry.LineSet"]) -> "o3d.geometry.LineSet":
    import open3d as o3d

    if not lsets:
        return o3d.geometry.LineSet()
    pts_all = []
    lines_all = []
    cols_all = []
    off = 0
    for ls in lsets:
        pts = np.asarray(ls.points)
        n = len(pts)
        pts_all.append(pts)
        for (a, b) in np.asarray(ls.lines):
            lines_all.append([off + a, off + b])
        cols_all.append(np.asarray(ls.colors))
        off += n
    out = o3d.geometry.LineSet()
    out.points = o3d.utility.Vector3dVector(np.vstack(pts_all))
    out.lines = o3d.utility.Vector2iVector(np.asarray(lines_all, dtype=np.int32))
    out.colors = o3d.utility.Vector3dVector(np.vstack(cols_all))
    return out


def fetch_pi3_artifacts(
    ply_path: Optional[Path],
    image_paths: List[str],
    server_url: str,
    conf_threshold: float,
    rtol: float,
    *,
    out_dir: Optional[Path],
    write_ply: bool,
) -> Tuple[Path, Path]:
    """
    Call Pi3 ``/infer`` once via ``Pi3Client``. Writes ``<ply_stem>_cameras.json`` beside the PLY.

    - If ``ply_path`` is set: use that path for the PLY (and JSON stem).
    - If ``ply_path`` is None: ``out_dir`` must be set; PLY path is
      ``out_dir / <server ply_filename>``.

    If ``write_ply`` is False, the existing PLY on disk is not overwritten.

    Returns ``(resolved_ply_path, cameras_json_path)``.
    """
    from spagent.external_experts.Pi3.pi3_client import Pi3Client

    client = Pi3Client(server_url=server_url)
    if not client.health_check():
        raise RuntimeError(f"Pi3 server health check failed: {server_url}")
    result = client.infer_from_images(
        image_paths=[str(Path(p).resolve()) for p in image_paths],
        conf_threshold=conf_threshold,
        rtol=rtol,
        generate_views=False,
        use_filename=True,
    )
    if not result or not result.get("success"):
        raise RuntimeError("infer_from_images failed or returned no success flag")

    if ply_path is not None:
        target_ply = ply_path.resolve()
    else:
        if out_dir is None:
            raise RuntimeError("out_dir is required when --ply is omitted")
        out_root = out_dir.resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        ply_name = result.get("ply_filename") or "result.ply"
        target_ply = out_root / Path(ply_name).name

    if write_ply:
        ply_b64 = result.get("ply_file")
        if not ply_b64:
            raise RuntimeError("Server response has no ply_file; cannot write PLY.")
        target_ply.parent.mkdir(parents=True, exist_ok=True)
        target_ply.write_bytes(base64.b64decode(ply_b64))
        logger.info("Wrote PLY: %s", target_ply)
    cams = result.get("camera_poses")
    if not cams:
        raise RuntimeError("Server response has no camera_poses.")
    cam_out = target_ply.parent / f"{target_ply.stem}_cameras.json"
    cam_out.write_text(json.dumps(cams, indent=2), encoding="utf-8")
    logger.info("Wrote camera metadata: %s", cam_out)
    return target_ply, cam_out


def load_image_paths_from_file(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def run_gui(ply_path: Path, cameras_json: Path, *, voxel_size: Optional[float]) -> None:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    cams = json.loads(cameras_json.read_text(encoding="utf-8"))
    c2w = _load_c2w_list(cams)
    n_cam = len(c2w)

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_points():
        raise RuntimeError(f"No points in PLY: {ply_path}")

    bbox = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()))

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    base_pts = np.asarray(pcd.points, dtype=np.float64)
    base_cols = np.asarray(pcd.colors, dtype=np.float64)
    if base_cols.size == 0:
        base_cols = np.ones_like(base_pts) * 0.7

    class App:
        def __init__(self) -> None:
            gui.Application.instance.initialize()
            self.window = gui.Application.instance.create_window(
                "Pi3 point cloud + cameras", 1400, 900
            )
            self.window.set_on_layout(self._on_layout)
            self.window.set_on_close(self._on_close)

            self.scene = gui.SceneWidget()
            self.scene.scene = rendering.Open3DScene(self.window.renderer)
            try:
                self.scene.scene.show_ground_plane(False, rendering.Scene.GroundPlane.XY)
            except Exception:
                try:
                    self.scene.scene.show_ground_plane(False)
                except Exception:
                    pass

            self.mat_pcd = rendering.MaterialRecord()
            self.mat_pcd.shader = "defaultUnlit"
            self.mat_pcd.point_size = 6.0

            self.mat_ls = rendering.MaterialRecord()
            self.mat_ls.shader = "unlitLine"
            self.mat_ls.line_width = 8.0

            self._base_pts = base_pts.copy()
            self._base_cols = base_cols
            self._c2w = c2w
            self._n_cam = n_cam
            self._scene_center = np.mean(base_pts, axis=0)
            self._diag = diag

            # Fixed frustum params (small but visible).
            self._frustum_fov = 30.0
            self._frustum_aspect = 4.0 / 3.0
            self._frustum_len = 0.09
            self._global_backoff = 1.7

            self._pcd_geo = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(base_pts.copy())
            )
            self._pcd_geo.colors = o3d.utility.Vector3dVector(base_cols)
            self.scene.scene.add_geometry("pcd", self._pcd_geo, self.mat_pcd)

            self._cam_lines = self._build_all_frustums(self._c2w)
            self.scene.scene.add_geometry("cam_frustums", self._cam_lines, self.mat_ls)

            em = self.window.theme.font_size
            margin = gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em)
            self.panel = gui.Vert(0.5 * em, margin)

            self.panel.add_child(gui.Label("Reference camera (1-based, ego anchor)"))
            ref_items = [f"Camera {i + 1}" for i in range(self._n_cam)]
            self.ref_combo = gui.Combobox()
            self.ref_combo.clear_items()
            for label in ref_items:
                self.ref_combo.add_item(label)
            self.ref_combo.selected_index = 0
            self.ref_combo.set_on_selection_changed(self._on_param)
            self.panel.add_child(self.ref_combo)

            self.panel.add_child(gui.Label("View mode"))
            self.mode_ego = gui.ToggleSwitch("Ego (rotate cloud at ref)")
            self.mode_ego.is_on = True
            self.mode_ego.set_on_clicked(self._on_param)
            self.panel.add_child(self.mode_ego)

            self.panel.add_child(gui.Label("Azimuth (deg)"))
            self.az_slider = gui.Slider(gui.Slider.DOUBLE)
            self.az_slider.set_limits(-180, 180)
            self.az_slider.double_value = 0
            self.az_slider.set_on_value_changed(self._on_param)
            self.panel.add_child(self.az_slider)

            self.panel.add_child(gui.Label("Elevation (deg)"))
            self.el_slider = gui.Slider(gui.Slider.DOUBLE)
            self.el_slider.set_limits(-89, 89)
            self.el_slider.double_value = 0
            self.el_slider.set_on_value_changed(self._on_param)
            self.panel.add_child(self.el_slider)

            hint = gui.Label(
                "Drag azimuth / elevation sliders for dynamic motion. "
                "Ego rotates cloud + cameras together; Global orbits the view. "
                "Mouse: rotate / zoom scene."
            )
            self.panel.add_child(hint)

            self.window.add_child(self.scene)
            self.window.add_child(self.panel)

            self._apply_view()

        def _on_close(self) -> bool:
            return True

        def _on_layout(self, ctx: gui.LayoutContext) -> None:
            em = self.window.theme.font_size
            panel_width = 22 * em
            rect = self.window.content_rect
            self.scene.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
            self.panel.frame = gui.Rect(
                self.scene.frame.get_right(),
                rect.y,
                panel_width,
                rect.height,
            )

        def _ref_idx(self) -> int:
            return int(self.ref_combo.selected_index)

        def _on_param(self, *_) -> None:
            self._apply_geometry()
            self._apply_view()

        def _build_all_frustums(self, c2w_use: np.ndarray) -> "o3d.geometry.LineSet":
            palette = np.asarray(
                [
                    [1.0, 0.2, 0.2],
                    [0.2, 0.6, 1.0],
                    [0.2, 1.0, 0.4],
                    [1.0, 0.6, 0.2],
                    [0.8, 0.4, 1.0],
                    [0.6, 0.4, 0.2],
                ],
                dtype=np.float64,
            )
            lsets = []
            for i in range(self._n_cam):
                lsets.append(
                    make_camera_frustum_lineset(
                        c2w_use[i],
                        fov_deg=self._frustum_fov,
                        aspect=self._frustum_aspect,
                        length=self._frustum_len,
                        color=palette[i % len(palette)],
                    )
                )
            return merge_linesets(lsets)

        def _refresh_frustums(self, c2w_use: np.ndarray) -> None:
            self.scene.scene.remove_geometry("cam_frustums")
            self._cam_lines = self._build_all_frustums(c2w_use)
            self.scene.scene.add_geometry("cam_frustums", self._cam_lines, self.mat_ls)

        def _apply_geometry(self) -> None:
            ref_i = self._ref_idx()
            az = float(self.az_slider.double_value)
            el = float(self.el_slider.double_value)
            if self.mode_ego.is_on:
                pts = ego_rotate_points_world(self._base_pts, self._c2w[ref_i], az, el)
                r_rel, anchor = ego_world_rotation(self._c2w[ref_i], az, el)
                c2w_rot = self._c2w.copy()
                for i in range(self._n_cam):
                    r0 = c2w_rot[i, :3, :3]
                    t0 = c2w_rot[i, :3, 3]
                    c2w_rot[i, :3, :3] = r_rel @ r0
                    c2w_rot[i, :3, 3] = (r_rel @ (t0 - anchor)) + anchor
                self._refresh_frustums(c2w_rot)
            else:
                pts = self._base_pts
                self._refresh_frustums(self._c2w)

            self._pcd_geo.points = o3d.utility.Vector3dVector(pts)
            self._pcd_geo.colors = o3d.utility.Vector3dVector(self._base_cols)
            self.scene.scene.remove_geometry("pcd")
            self.scene.scene.add_geometry("pcd", self._pcd_geo, self.mat_pcd)

        def _apply_view(self) -> None:
            ref_i = self._ref_idx()
            c = self._c2w[ref_i]
            t = c[:3, 3]
            r = c[:3, :3]
            forward = r @ np.array([0.0, 0.0, 1.0])
            forward /= np.linalg.norm(forward) + 1e-9
            up_cv = r @ np.array([0.0, -1.0, 0.0])
            up_cv /= np.linalg.norm(up_cv) + 1e-9

            if self.mode_ego.is_on:
                eye = t
                lookat = t + forward * max(0.15 * self._diag, 0.05)
                up = up_cv
            else:
                az = float(self.az_slider.double_value)
                el = float(self.el_slider.double_value)
                r_rel, _ = ego_world_rotation(self._c2w[ref_i], az, el)
                view_dir = r_rel @ forward
                view_dir /= np.linalg.norm(view_dir) + 1e-9
                up = r_rel @ up_cv
                up /= np.linalg.norm(up) + 1e-9
                lookat = self._scene_center
                eye = t - view_dir * float(self._global_backoff)

            self.scene.look_at(lookat, eye, up)

    _app = App()  # noqa: F841
    gui.Application.instance.run()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="Path to PLY. Omit with --fetch to use server filename under --out-dir",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="With --fetch and no --ply, write PLY + JSON here (default: outputs)",
    )
    p.add_argument(
        "--cameras-json",
        type=Path,
        default=None,
        help="Camera metadata JSON (default: <ply_stem>_cameras.json next to PLY)",
    )
    p.add_argument(
        "--fetch",
        action="store_true",
        help="Contact Pi3 server /infer (via Pi3Client), write PLY and cameras JSON if needed",
    )
    p.add_argument(
        "--force-fetch",
        action="store_true",
        help="With --fetch, always call /infer even if PLY and JSON already exist",
    )
    p.add_argument(
        "-f",
        "--images-file",
        type=Path,
        help="Image list (one path per line) for --fetch",
    )
    p.add_argument("--server", default="http://localhost:20021")
    p.add_argument("--conf-threshold", type=float, default=0.1)
    p.add_argument("--rtol", type=float, default=0.03)
    p.add_argument("--voxel", type=float, default=0.002, help="Optional voxel downsample size")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        import open3d as o3d  # noqa: F401
    except ImportError:
        logger.error("open3d is required: pip install open3d")
        return 1

    args = _parse_args()
    out_dir_default = args.out_dir if args.out_dir is not None else Path("outputs")

    # Resolve ply + cameras paths for "skip server if ready" check
    if args.ply is not None:
        ply_path = args.ply.resolve()
        cam_json = args.cameras_json
        if cam_json is None:
            cam_json = ply_path.parent / f"{ply_path.stem}_cameras.json"
        else:
            cam_json = cam_json.resolve()
    else:
        ply_path = None  # type: ignore[assignment]
        cam_json = None  # type: ignore[assignment]

    if args.fetch:
        if not args.images_file or not args.images_file.is_file():
            logger.error("--fetch requires -f/--images-file with reconstruction images.")
            return 1
        paths = [
            str(_ROOT / p) if not Path(p).is_absolute() else p
            for p in load_image_paths_from_file(args.images_file)
        ]
        for p in paths:
            if not Path(p).is_file():
                logger.error("Image not found: %s", p)
                return 1

        do_infer = True
        if args.ply is not None and cam_json is not None and not args.force_fetch:
            if ply_path.is_file() and cam_json.is_file():
                logger.info(
                    "Using existing PLY and cameras JSON (skip Pi3 server). "
                    "Pass --force-fetch to re-run /infer."
                )
                do_infer = False
        if do_infer:
            if args.ply is not None:
                ply_missing = not ply_path.is_file()
                fetch_out: Optional[Path] = None
            else:
                ply_missing = True
                fetch_out = out_dir_default.resolve()
            try:
                ply_path, cam_json = fetch_pi3_artifacts(
                    ply_path,
                    paths,
                    args.server,
                    args.conf_threshold,
                    args.rtol,
                    out_dir=fetch_out,
                    write_ply=ply_missing,
                )
            except Exception as e:
                logger.error("%s", e)
                return 1
    else:
        if args.ply is None:
            logger.error("Without --fetch, --ply is required.")
            return 1
        assert cam_json is not None
        if not ply_path.is_file():
            logger.error(
                "PLY not found: %s\nUse: --fetch -f <images.txt> to build from Pi3 server.",
                ply_path,
            )
            return 1
        if not cam_json.is_file():
            logger.error(
                "Camera JSON not found: %s\nUse once: --fetch -f <images.txt>",
                cam_json,
            )
            return 1

    try:
        run_gui(ply_path, cam_json, voxel_size=args.voxel)
    except Exception as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

