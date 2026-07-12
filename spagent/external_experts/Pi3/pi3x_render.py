"""
pi3x_render.py — pure-numpy / matplotlib rendering helpers extracted from pi3x_server.py.

These functions have NO dependency on the Pi3X model, Flask, or any GPU code.
They operate directly on numpy arrays (points, colors, camera_poses) produced by
the offline precompute pipeline and saved as .npz cache files.

Public API
----------
render_view(points, colors, camera_poses,
            azimuth_angle, elevation_angle,
            rotation_reference_camera=1, camera_view=False,
            output_width=1024, output_height=768)
    -> base64-encoded PNG string

subsample_points(points, colors, max_points=1_500_000)
    -> (points_sub, colors_sub)
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int = 1_500_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic uniform sub-sampling of a point cloud."""
    n = len(points)
    if n <= max_points:
        return points, colors
    step = n // max_points
    idx = np.arange(0, n, step)[:max_points]
    return points[idx], colors[idx]


# ---------------------------------------------------------------------------
# Camera visualisation (extracted verbatim from pi3x_server.py)
# ---------------------------------------------------------------------------

def _draw_cameras_visualization(
    ax,
    camera_centers: np.ndarray,
    camera_poses: np.ndarray,
    current_view_cam_idx: int,
    view_R_cam: np.ndarray,
    view_t_cam: np.ndarray,
    max_range: float,
    show_cameras: bool = True,
):
    if not show_cameras:
        return None, None, None, None, None, None

    axis_length = max_range * 0.12
    all_x, all_y, all_z = [], [], []
    camera_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    is_identity = (
        np.allclose(view_R_cam, np.eye(3)) and np.allclose(view_t_cam, np.zeros(3))
    )

    for cam_idx, (cam_center, cam_pose) in enumerate(zip(camera_centers, camera_poses)):
        cam_color = camera_colors[cam_idx % len(camera_colors)]
        flip = np.diag([1, -1, -1])

        if is_identity:
            c_view = cam_center
        else:
            c_view = (flip @ ((view_R_cam @ cam_center.T).T + view_t_cam).T).T

        R_cam2world = cam_pose[:3, :3]
        if is_identity:
            R_pose = R_cam2world
        else:
            R_pose = flip @ (view_R_cam @ R_cam2world)

        fl = axis_length * 0.8
        fw = fl * 0.3
        fh = fl * 0.3
        forward = -R_pose[:, 2]
        right = R_pose[:, 0]
        up = -R_pose[:, 1]
        far_c = c_view + forward * fl
        corners = [
            far_c + right * fw + up * fh,
            far_c - right * fw + up * fh,
            far_c - right * fw - up * fh,
            far_c + right * fw - up * fh,
        ]
        lw = 1.2 if cam_idx == current_view_cam_idx else 1.0
        alpha = 0.65 if cam_idx == current_view_cam_idx else 0.5

        for corner in corners:
            ax.plot(
                [c_view[0], corner[0]],
                [c_view[1], corner[1]],
                [c_view[2], corner[2]],
                color=cam_color, linewidth=lw, alpha=alpha,
            )
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            ax.plot(
                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=cam_color, linewidth=lw, alpha=alpha,
            )
        ms = 35 if cam_idx == current_view_cam_idx else 20
        ax.scatter(c_view[0], c_view[1], c_view[2],
                   c=cam_color, s=ms, marker='o', alpha=0.8, depthshade=False,
                   edgecolors='black', linewidth=0.8)
        label_pos = c_view + np.array([axis_length * 0.3, 0, axis_length * 0.2])
        marker = '*' if cam_idx == current_view_cam_idx else ''
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                f'Cam{cam_idx+1}{marker}', fontsize=5, color='black', weight='bold',
                bbox=dict(boxstyle="round,pad=0.05", facecolor=cam_color, alpha=0.6))

        for coord in [c_view, far_c] + corners:
            all_x.append(coord[0]); all_y.append(coord[1]); all_z.append(coord[2])

    if all_x:
        return min(all_x), max(all_x), min(all_y), max(all_y), min(all_z), max(all_z)
    return None, None, None, None, None, None


# ---------------------------------------------------------------------------
# SfM-style first-person rendering
# ---------------------------------------------------------------------------

def _render_point_cloud_sfm(
    points_world: np.ndarray,
    colors: np.ndarray,
    camera_poses: np.ndarray,
    ref_cam_idx: int,
    azim_angle: float,
    elev_angle: float,
    output_width: int = 1024,
    output_height: int = 768,
) -> str:
    """Pinhole-camera projection with z-buffer. Returns base64 PNG."""
    import base64

    safe_ref = max(0, min(ref_cam_idx, len(camera_poses) - 1))
    ref_pose = camera_poses[safe_ref]
    R_cw = ref_pose[:3, :3]
    t_cw = ref_pose[:3, 3]

    R_local = np.eye(3)
    if abs(azim_angle) > 1e-6:
        R_local = R_local @ R.from_rotvec(np.radians(azim_angle) * np.array([0, 1, 0])).as_matrix()
    if abs(elev_angle) > 1e-6:
        R_local = R_local @ R.from_rotvec(np.radians(-elev_angle) * np.array([1, 0, 0])).as_matrix()

    R_virtual_cw = R_cw @ R_local
    R_wc = R_virtual_cw.T
    t_wc = -R_wc @ t_cw

    points_cam = (R_wc @ points_world.T).T + t_wc
    valid = points_cam[:, 2] > 1e-3
    points_cam = points_cam[valid]
    colors_v = np.clip(colors[valid], 0, 1)

    def _blank_b64():
        img = np.ones((output_height, output_width, 3), dtype=np.uint8) * 200
        buf = io.BytesIO()
        from PIL import Image
        Image.fromarray(img).save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    if len(points_cam) == 0:
        logger.warning("SfM rendering: no points visible")
        return _blank_b64()

    fov_h_deg = 140.0
    fx = output_width / (2 * np.tan(np.radians(fov_h_deg / 2)))
    fy = fx
    cx = output_width / 2.0
    cy = output_height / 2.0

    z = points_cam[:, 2]
    u = (fx * points_cam[:, 0] / z + cx)
    v = (fy * points_cam[:, 1] / z + cy)
    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)

    in_b = (u_i >= 0) & (u_i < output_width) & (v_i >= 0) & (v_i < output_height)
    u_i, v_i, z_v, c_v = u_i[in_b], v_i[in_b], z[in_b], colors_v[in_b]

    if len(u_i) == 0:
        return _blank_b64()

    image = np.ones((output_height, output_width, 3), dtype=np.float32)
    depth_buf = np.full((output_height, output_width), np.inf, dtype=np.float32)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            uu = np.clip(u_i + dx, 0, output_width - 1)
            vv = np.clip(v_i + dy, 0, output_height - 1)
            lin = vv * output_width + uu
            order = np.argsort(z_v)
            lin_o, z_o, c_o = lin[order], z_v[order], c_v[order]
            _, first = np.unique(lin_o, return_index=True)
            lin_u, z_u, c_u = lin_o[first], z_o[first], c_o[first]
            vv_u, uu_u = lin_u // output_width, lin_u % output_width
            closer = z_u < depth_buf[vv_u, uu_u]
            depth_buf[vv_u[closer], uu_u[closer]] = z_u[closer]
            image[vv_u[closer], uu_u[closer]] = c_u[closer]

    from PIL import Image as PILImage
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(img_uint8).save(buf, format='PNG')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    logger.info(f"SfM rendering done: {len(u_i)} pts, azim={azim_angle}°, elev={elev_angle}°")
    return b64


# ---------------------------------------------------------------------------
# Global-view matplotlib rendering
# ---------------------------------------------------------------------------

def _create_view_image(
    points_sample: np.ndarray,
    colors_sample: np.ndarray,
    camera_centers: np.ndarray,
    camera_poses: np.ndarray,
    cam_idx: int,
    azim_angle: float,
    elev_angle: float,
    view_name: str,
    show_camera_axes: bool = True,
    show_all_cameras: bool = True,
    ref_cam_idx: int = 0,
    camera_view: bool = False,
) -> str:
    import base64

    if camera_view:
        return _render_point_cloud_sfm(
            points_sample, colors_sample, camera_poses, ref_cam_idx, azim_angle, elev_angle
        )

    # Global-view mode
    view_cam_pose = camera_poses[cam_idx]
    R_cw = view_cam_pose[:3, :3]
    t_cw = view_cam_pose[:3, 3]
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    points_cam = (R_wc @ points_sample.T).T + t_wc

    flip = np.diag([1, -1, -1])
    points_cam = (flip @ points_cam.T).T

    try:
        if abs(azim_angle) > 1e-6 or abs(elev_angle) > 1e-6:
            safe_ref = max(0, min(ref_cam_idx, len(camera_poses) - 1))
            first_cam_world_pos = camera_poses[safe_ref][:3, 3]
            first_cam_R_cw = camera_poses[safe_ref][:3, :3]

            first_cam_in_current = (R_wc @ first_cam_world_pos.T).T + t_wc
            first_cam_center = (flip @ first_cam_in_current.T).T

            first_cam_R_in_current = flip @ (R_wc @ first_cam_R_cw)
            y_axis = first_cam_R_in_current[:, 1]
            x_axis = first_cam_R_in_current[:, 0]

            pts = points_cam - first_cam_center
            if abs(azim_angle) > 1e-6:
                pts = (R.from_rotvec(np.radians(azim_angle) * y_axis).as_matrix() @ pts.T).T
            if abs(elev_angle) > 1e-6:
                pts = (R.from_rotvec(np.radians(elev_angle) * x_axis).as_matrix() @ pts.T).T
            points_cam = pts + first_cam_center

            R_az = R.from_rotvec(np.radians(azim_angle) * y_axis).as_matrix() if abs(azim_angle) > 1e-6 else np.eye(3)
            R_el = R.from_rotvec(np.radians(elev_angle) * x_axis).as_matrix() if abs(elev_angle) > 1e-6 else np.eye(3)
            R_rel = R_el @ R_az

            view_R_cam = R_rel @ R_wc
            view_t_cam = R_rel @ t_wc

            rot_cens, rot_poses = [], []
            for cc, cp in zip(camera_centers, camera_poses):
                cc_cur = (flip @ ((R_wc @ cc.T).T + t_wc).T).T
                cc_c = cc_cur - first_cam_center
                cc_rot = (R_rel @ cc_c.T).T + first_cam_center
                rot_cens.append(cc_rot)
                cR = flip @ (R_wc @ cp[:3, :3])
                p4 = np.eye(4); p4[:3, :3] = R_rel @ cR; p4[:3, 3] = cc_rot
                rot_poses.append(p4)
            camera_centers = np.array(rot_cens)
            camera_poses = np.array(rot_poses)
        else:
            view_R_cam = R_wc
            view_t_cam = t_wc
    except Exception:
        view_R_cam = R_wc
        view_t_cam = t_wc

    lo = np.percentile(points_cam, 7, axis=0)
    hi = np.percentile(points_cam, 93, axis=0)
    max_range = max(hi - lo) or 1.0
    if max_range > 0:
        pt_size = max(0.15, min(0.6, 120.0 / max_range))
    else:
        pt_size = 2.0
    alpha = 0.85

    fig = plt.figure(figsize=(12, 10), dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False

    try:
        c_norm = np.clip(colors_sample, 0, 1)
        scatter = ax.scatter(
            points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
            c=c_norm, s=pt_size, alpha=alpha, edgecolors='none',
            depthshade=True, linewidth=0,
        )
        try:
            scatter.set_clip_on(False)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"RGB scatter failed, falling back: {e}")
        ax.scatter(points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
                   c='steelblue', s=pt_size * 2, alpha=0.9)

    show_cams = show_all_cameras and not camera_view
    if show_cams:
        x_min_c, x_max_c, y_min_c, y_max_c, z_min_c, z_max_c = _draw_cameras_visualization(
            ax, camera_centers, camera_poses, cam_idx, np.eye(3), np.zeros(3), max_range, True
        )
        if x_min_c is not None:
            x_min = min(points_cam[:, 0].min(), x_min_c)
            x_max = max(points_cam[:, 0].max(), x_max_c)
            y_min = min(points_cam[:, 1].min(), y_min_c)
            y_max = max(points_cam[:, 1].max(), y_max_c)
            z_min = min(points_cam[:, 2].min(), z_min_c)
            z_max = max(points_cam[:, 2].max(), z_max_c)
        else:
            x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
            y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
            z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()
    else:
        x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
        y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
        z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()

    ax.view_init(elev=0.0, azim=-90.0)
    ax.dist = 10

    mf = 0.02
    ax.set_xlim(x_min - (x_max - x_min) * mf, x_max + (x_max - x_min) * mf)
    ax.set_ylim(y_min - (y_max - y_min) * mf, y_max + (y_max - y_min) * mf)
    ax.set_zlim(z_min - (z_max - z_min) * mf, z_max + (z_max - z_min) * mf)

    ax.set_xlabel('X', fontsize=14, fontweight='bold', color='red')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold', color='green')
    ax.set_zlabel('Z', fontsize=14, fontweight='bold', color='blue')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('white')
    ax.set_facecolor('lightgray')

    buf = io.BytesIO()
    try:
        try:
            fig.canvas.draw()
        except Exception:
            pass
        plt.savefig(buf, format='png', dpi=500, bbox_inches='tight',
                    pad_inches=0.05, facecolor='white', edgecolor='none', transparent=False)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"savefig failed: {e}")
        b64 = base64.b64encode(b"").decode('utf-8')
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
        buf.close()

    return b64


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def render_view(
    points: np.ndarray,
    colors: np.ndarray,
    camera_poses: np.ndarray,
    azimuth_angle: float,
    elevation_angle: float,
    rotation_reference_camera: int = 1,
    camera_view: bool = False,
    max_points: int = 1_500_000,
) -> str:
    """
    Render a novel viewpoint from a cached point cloud.

    Parameters
    ----------
    points : (N, 3) float32 numpy array – world-space 3D points
    colors : (N, 3) float32 numpy array – RGB in [0, 1]
    camera_poses : (M, 4, 4) float32 numpy array – camera-to-world matrices
    azimuth_angle : float – left/right rotation (degrees)
    elevation_angle : float – up/down rotation (degrees)
    rotation_reference_camera : int – 1-based reference camera index
    camera_view : bool – if True use SfM first-person rendering
    max_points : int – subsample threshold for performance

    Returns
    -------
    str – base64-encoded PNG image
    """
    points_s, colors_s = subsample_points(points, colors, max_points)

    camera_centers = camera_poses[:, :3, 3]
    cam_idx = max(0, min(int(rotation_reference_camera) - 1, len(camera_centers) - 1))

    # For angle == (0, 0), return direct matplotlib front-view (no rotation applied)
    # (the server returns the original input image for 0°/0° but we don't have it
    # in offline mode, so we render the point cloud at 0°/0° instead)

    # Global-view mode: the server adds +100° elevation to account for coordinate
    # flip; we replicate that here.
    if not camera_view:
        adjusted_elevation = elevation_angle + 100.0
    else:
        adjusted_elevation = elevation_angle

    view_name = f"custom_azim_{azimuth_angle}_elev_{elevation_angle}"
    b64 = _create_view_image(
        points_s, colors_s, camera_centers, camera_poses,
        cam_idx, azimuth_angle, adjusted_elevation, view_name,
        show_camera_axes=False, show_all_cameras=True,
        ref_cam_idx=cam_idx, camera_view=camera_view,
    )
    return b64
