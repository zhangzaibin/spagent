# Copyright (c) Meta Platforms, Inc. and affiliates.
# MapAnything Server - Flask-based inference service
# Designed to match Pi3 server interface for easy comparison

import base64
import cv2
import gc
import io
import logging
import numpy as np
import torch
import os
import sys
import argparse
from flask import Flask, request, jsonify
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0
from scipy.spatial.transform import Rotation as R
from sklearn.covariance import EmpiricalCovariance

# Add mapanything to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None

# MapAnything configuration (same as gradio_app.py)
HIGH_LEVEL_CONFIG = {
    "path": "configs/train.yaml",
    "hf_model_name": "facebook/map-anything",
    "model_str": "mapanything",
    "config_overrides": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "checkpoint_name": "model.safetensors",
    "config_name": "config.json",
    "trained_with_amp": True,
    "trained_with_amp_dtype": "bf16",
    "data_norm_type": "dinov2",
    "patch_size": 14,
    "resolution": 518,
}


def load_model():
    """Load MapAnything model"""
    global model
    try:
        logger.info("Loading MapAnything model...")
        
        # Unset proxy variables to avoid SOCKS proxy issues
        for var in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
            if var in os.environ:
                del os.environ[var]
        
        # Set torch hub to use local cache only
        torch_hub_dir = os.path.expanduser('~/.cache/torch/hub')
        os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
        torch.hub.set_dir(torch_hub_dir)
        
        # Patch torch.hub.load to use cached repo path
        original_hub_load = torch.hub.load
        def patched_hub_load(*args, **kwargs):
            if args and isinstance(args[0], str) and '/' in args[0]:
                # Convert repo name to local cached path
                repo_name = args[0].replace('/', '_') + '_main'
                local_repo_path = os.path.join(torch_hub_dir, repo_name)
                if os.path.exists(local_repo_path):
                    # Replace repo with local path
                    args = (local_repo_path,) + args[1:]
                    kwargs['source'] = 'local'
                    kwargs.pop('force_reload', None)
                    logger.info(f"Loading from local torch hub cache: {local_repo_path}")
            return original_hub_load(*args, **kwargs)
        torch.hub.load = patched_hub_load
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Import MapAnything model class
        from mapanything.models.mapanything.model import MapAnything
        import json
        from safetensors.torch import load_file as load_safetensors
        
        # Local cache path
        cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--map-anything")
        
        # Find the snapshot directory
        snapshots_dir = os.path.join(cache_base, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshot_dirs = os.listdir(snapshots_dir)
            if snapshot_dirs:
                snapshot_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                config_path = os.path.join(snapshot_path, "config.json")
                weights_path = os.path.join(snapshot_path, "model.safetensors")
                
                logger.info(f"Loading from local cache: {snapshot_path}")
                
                if os.path.exists(config_path) and os.path.exists(weights_path):
                    # Load config
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Disable torch hub loading for encoder
                    if "encoder_config" in config:
                        config["encoder_config"]["uses_torch_hub"] = False
                        logger.info("Disabled torch hub loading for encoder")
                    
                    # Create model from config
                    logger.info("Creating model from config...")
                    model = MapAnything(**config)
                    
                    # Load weights
                    logger.info("Loading weights...")
                    state_dict = load_safetensors(weights_path)
                    model.load_state_dict(state_dict, strict=False)
                    
                    model = model.to(device)
                    model.eval()
                    
                    # Restore original torch.hub.load
                    torch.hub.load = original_hub_load
                    
                    logger.info("MapAnything model loaded successfully!")
                    return True
        
        # Fallback: try from_pretrained without local_files_only
        logger.info("Trying from_pretrained...")
        model = MapAnything.from_pretrained("facebook/map-anything").to(device)
        model.eval()
        
        # Restore original torch.hub.load
        torch.hub.load = original_hub_load
        
        logger.info("MapAnything model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        # Restore original torch.hub.load on error
        try:
            torch.hub.load = original_hub_load
        except:
            pass
        return False


def remove_outliers_mahalanobis(points, colors, threshold_std=3.0):
    """
    Remove outliers using Mahalanobis distance (same as Pi3)
    
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array
        threshold_std: Standard deviation threshold
        
    Returns:
        filtered_points, filtered_colors, inlier_mask
    """
    if len(points) == 0:
        return points, colors, np.ones(len(points), dtype=bool)
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()
    
    # Compute Mahalanobis distance
    cov = EmpiricalCovariance().fit(points)
    dist = cov.mahalanobis(points)
    
    # Filter based on threshold
    mean_dist = np.mean(dist)
    std_dist = np.std(dist)
    threshold = mean_dist + threshold_std * std_dist
    
    inlier_mask = dist < threshold
    filtered_points = points[inlier_mask]
    filtered_colors = colors[inlier_mask]
    
    removed_count = len(points) - len(filtered_points)
    removed_percentage = (removed_count / len(points)) * 100
    logger.info(f"Mahalanobis outlier removal: original={len(points)}, removed={removed_count} ({removed_percentage:.2f}%), kept={len(filtered_points)}")
    
    return filtered_points, filtered_colors, inlier_mask


def write_ply(points, colors, path):
    """Write PLY file and return base64 encoded content"""
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()
    
    # Ensure colors are in 0-255 range
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    
    # Reshape if needed
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    
    # Create PLY content
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # Write to file
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, 'w') as f:
        f.write(header)
        for i in range(len(points)):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
    
    # Read and encode as base64
    with open(path, 'rb') as f:
        ply_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    return ply_b64


def load_image_from_array(img_array, norm_type="dinov2", patch_size=14, resolution_set=518):
    """
    Convert numpy image array to MapAnything view format
    
    Args:
        img_array: RGB numpy array (H, W, 3)
        norm_type: Normalization type
        patch_size: Patch size
        resolution_set: Resolution set
        
    Returns:
        view dictionary
    """
    from mapanything.utils.image import find_closest_aspect_ratio
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
    import torchvision.transforms as tvf
    
    # Get image dimensions
    H, W = img_array.shape[:2]
    aspect_ratio = W / H
    
    # Find target size
    target_W, target_H = find_closest_aspect_ratio(aspect_ratio, resolution_set)
    
    # Resize image
    img_resized = cv2.resize(img_array, (target_W, target_H), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    
    # Apply normalization
    if norm_type in IMAGE_NORMALIZATION_DICT:
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        normalize = tvf.Normalize(mean=img_norm.mean, std=img_norm.std)
        img_tensor = normalize(img_tensor)
    
    # Create view dictionary
    view = {
        "img": img_tensor.unsqueeze(0),  # Add batch dimension: (1, 3, H, W)
        "data_norm_type": [norm_type],
        "true_shape": [(target_H, target_W)],
    }
    
    return view


def run_inference(images_bgr_list, apply_mask=True, conf_percentile=10):
    """
    Run MapAnything inference on a list of BGR images
    
    Args:
        images_bgr_list: List of BGR numpy arrays
        apply_mask: Whether to apply mask during inference
        conf_percentile: Confidence percentile threshold (0-100, default 10, removes bottom 10% confidence points)
    
    Returns:
        predictions dict with world_points, colors, camera_poses, etc.
    """
    global model
    
    from mapanything.utils.geometry import depthmap_to_world_frame
    
    device = next(model.parameters()).device
    
    # Prepare views
    views = []
    for img_bgr in images_bgr_list:
        view = load_image_from_array(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            norm_type=HIGH_LEVEL_CONFIG["data_norm_type"],
            patch_size=HIGH_LEVEL_CONFIG["patch_size"],
            resolution_set=HIGH_LEVEL_CONFIG["resolution"],
        )
        # Move to device
        view["img"] = view["img"].to(device)
        views.append(view)
    
    logger.info(f"Prepared {len(views)} views for inference")
    
    # Run model inference with built-in filtering
    # MapAnything model already handles:
    # - Edge filtering (mask_edges=True, edge_depth_threshold=0.03)
    # - Confidence filtering (apply_confidence_mask=True, using percentile)
    logger.info(f"Running MapAnything inference with conf_percentile={conf_percentile}...")
    outputs = model.infer(
        views, 
        apply_mask=apply_mask, 
        mask_edges=True,  # Built-in edge filtering
        apply_confidence_mask=(conf_percentile > 0),  # Enable built-in confidence filtering
        confidence_percentile=conf_percentile,  # Bottom N% will be filtered out
        memory_efficient_inference=False
    )
    
    logger.info("Inference completed!")
    
    # Process outputs
    all_points = []
    all_colors = []
    camera_poses_list = []
    intrinsics_list = []
    
    for view_idx, pred in enumerate(outputs):
        # Extract data
        depthmap = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics = pred["intrinsics"][0]  # (3, 3)
        camera_pose = pred["camera_poses"][0]  # (4, 4)
        
        # Get mask
        if "mask" in pred:
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        else:
            mask = np.ones_like(depthmap.cpu().numpy(), dtype=bool)
        
        # Compute 3D points in world frame
        pts3d, valid_mask = depthmap_to_world_frame(depthmap, intrinsics, camera_pose)
        pts3d = pts3d.cpu().numpy()
        valid_mask = valid_mask.cpu().numpy()
        
        # Combine masks
        # Note: The model's 'mask' already includes edge filtering and confidence filtering
        # when apply_confidence_mask=True and mask_edges=True are set in model.infer()
        final_mask = mask & valid_mask
        
        logger.debug(f"View {view_idx}: After filtering, kept {np.sum(final_mask)} points")
        
        # Get colors from denormalized image
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # (H, W, 3)
        
        # Extract valid points and colors
        valid_points = pts3d[final_mask]
        valid_colors = img_no_norm[final_mask]
        
        all_points.append(valid_points)
        all_colors.append(valid_colors)
        camera_poses_list.append(camera_pose.cpu().numpy())
        intrinsics_list.append(intrinsics.cpu().numpy())
    
    # Combine all points and colors
    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)
    
    predictions = {
        "world_points": combined_points,
        "colors": combined_colors,
        "camera_poses": np.stack(camera_poses_list, axis=0),
        "intrinsics": np.stack(intrinsics_list, axis=0),
    }
    
    return predictions


def _prepare_points_and_cameras(predictions, points_filtered=None, colors_filtered=None):
    """
    Prepare point cloud and camera data for visualization
    
    Returns:
        tuple: (points_sample, colors_sample, camera_centers, camera_poses)
    """
    # Use filtered points if provided
    if points_filtered is not None and colors_filtered is not None:
        if isinstance(points_filtered, np.ndarray):
            points_3d = points_filtered
        else:
            points_3d = points_filtered.cpu().numpy()
        
        if isinstance(colors_filtered, np.ndarray):
            colors_3d = colors_filtered
        else:
            colors_3d = colors_filtered.cpu().numpy()
    else:
        points_3d = predictions["world_points"]
        colors_3d = predictions["colors"]
    
    camera_poses = predictions["camera_poses"]
    
    # Extract camera centers from camera-to-world matrices
    camera_centers = []
    for pose in camera_poses:
        camera_centers.append(pose[:3, 3])
    camera_centers = np.array(camera_centers)
    
    # Subsample for visualization
    max_points_to_visualize = 500000
    if len(points_3d) > max_points_to_visualize:
        total_points = len(points_3d)
        step = total_points // max_points_to_visualize
        indices = np.arange(0, total_points, step)[:max_points_to_visualize]
        points_sample = points_3d[indices]
        colors_sample = colors_3d[indices]
        logger.info(f"Subsampled from {len(points_3d)} to {len(points_sample)} points")
    else:
        points_sample = points_3d
        colors_sample = colors_3d
    
    return points_sample, colors_sample, camera_centers, camera_poses


def _draw_cameras_visualization(ax, camera_centers, camera_poses, current_view_cam_idx, 
                              max_range, show_cameras=True):
    """
    Draw camera visualization in 3D plot
    """
    if not show_cameras:
        return None, None, None, None, None, None
    
    axis_length = max_range * 0.12
    
    all_x_coords = []
    all_y_coords = []
    all_z_coords = []
    
    camera_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for cam_idx, (cam_center, cam_pose) in enumerate(zip(camera_centers, camera_poses)):
        cam_color = camera_colors[cam_idx % len(camera_colors)]
        
        # Camera rotation (camera-to-world)
        R_cam2world = cam_pose[:3, :3]
        
        # Draw camera frustum
        frustum_length = axis_length * 0.8
        frustum_width = frustum_length * 0.3
        frustum_height = frustum_length * 0.3
        
        # Camera directions
        forward = -R_cam2world[:, 2]  # Camera looks at -Z
        right = R_cam2world[:, 0]
        up = -R_cam2world[:, 1]
        
        far_center = cam_center + forward * frustum_length
        
        corner1 = far_center + right * frustum_width + up * frustum_height
        corner2 = far_center - right * frustum_width + up * frustum_height
        corner3 = far_center - right * frustum_width - up * frustum_height
        corner4 = far_center + right * frustum_width - up * frustum_height
        
        line_width = 1.2 if cam_idx == current_view_cam_idx else 1
        alpha = 0.65 if cam_idx == current_view_cam_idx else 0.5
        
        # Draw frustum edges
        for corner in [corner1, corner2, corner3, corner4]:
            ax.plot([cam_center[0], corner[0]],
                   [cam_center[1], corner[1]],
                   [cam_center[2], corner[2]], 
                   color=cam_color, linewidth=line_width, alpha=alpha)
        
        # Draw frustum base rectangle
        corners = [corner1, corner2, corner3, corner4, corner1]
        for i in range(len(corners) - 1):
            ax.plot([corners[i][0], corners[i+1][0]],
                   [corners[i][1], corners[i+1][1]],
                   [corners[i][2], corners[i+1][2]], 
                   color=cam_color, linewidth=line_width, alpha=alpha)
        
        # Draw camera center
        marker_size = 35 if cam_idx == current_view_cam_idx else 20
        ax.scatter(cam_center[0], cam_center[1], cam_center[2], 
                  c=cam_color, s=marker_size, marker='o', alpha=0.8, depthshade=False,
                  edgecolors='black', linewidth=0.8)
        
        # Add camera label
        label_pos = cam_center + np.array([axis_length * 0.3, 0, axis_length * 0.2])
        marker = '*' if cam_idx == current_view_cam_idx else ''
        ax.text(label_pos[0], label_pos[1], label_pos[2], 
               f'Cam{cam_idx+1}{marker}', fontsize=5, color='black', weight='bold',
               bbox=dict(boxstyle="round,pad=0.05", facecolor=cam_color, alpha=0.6))
        
        # Collect bounds
        coords_to_check = [cam_center, corner1, corner2, corner3, corner4, far_center]
        for coord in coords_to_check:
            all_x_coords.append(coord[0])
            all_y_coords.append(coord[1])
            all_z_coords.append(coord[2])
    
    if all_x_coords:
        return (min(all_x_coords), max(all_x_coords),
                min(all_y_coords), max(all_y_coords),
                min(all_z_coords), max(all_z_coords))
    else:
        return None, None, None, None, None, None


def _create_view_image(points_sample, colors_sample, camera_centers, camera_poses, cam_idx, 
                      azim_angle, elev_angle, view_name, show_camera_axes=True, show_all_cameras=True,
                      ref_cam_idx=0, camera_view=False):
    """
    Create a single view image of the point cloud
    
    Args:
        points_sample: Sampled point cloud
        colors_sample: Sampled colors
        camera_centers: Camera center positions
        camera_poses: Camera pose matrices (4x4, camera-to-world)
        cam_idx: Current camera index
        azim_angle: Azimuth angle
        elev_angle: Elevation angle
        view_name: View name
        show_camera_axes: Whether to show camera axes
        show_all_cameras: Whether to show all cameras
        ref_cam_idx: Reference camera index for rotation
        camera_view: Whether to use camera view mode
        
    Returns:
        str: base64 encoded PNG image
    """
    # Get view camera pose (camera-to-world matrix)
    if camera_view:
        safe_ref_idx = max(0, min(ref_cam_idx, len(camera_poses) - 1))
        view_cam_pose = camera_poses[safe_ref_idx]
        logger.info(f"Using camera view mode: observing from camera {safe_ref_idx + 1}")
    else:
        view_cam_pose = camera_poses[cam_idx]
    
    # Extract rotation and translation
    R_cw = view_cam_pose[:3, :3]
    t_cw = view_cam_pose[:3, 3]
    
    # Compute world-to-camera transform
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    
    # Transform points to camera frame
    points_cam = (R_wc @ points_sample.T).T + t_wc
    
    # Flip Y and Z for visualization (OpenCV to standard coordinates)
    flip_transform = np.diag([1, -1, -1])
    points_cam = (flip_transform @ points_cam.T).T
    
    # Apply rotation if needed
    if abs(azim_angle) > 1e-6 or abs(elev_angle) > 1e-6:
        # Get reference camera position
        safe_ref_idx = max(0, min(ref_cam_idx, len(camera_poses) - 1))
        first_cam_world_pos = camera_poses[safe_ref_idx][:3, 3]
        first_cam_R_cw = camera_poses[safe_ref_idx][:3, :3]
        
        # Transform to current view
        first_cam_in_current = (R_wc @ first_cam_world_pos.T).T + t_wc
        first_cam_center = (flip_transform @ first_cam_in_current.T).T
        
        # Get reference camera axes
        first_cam_R_in_current = R_wc @ first_cam_R_cw
        first_cam_R_in_current = flip_transform @ first_cam_R_in_current
        
        first_cam_x_axis = first_cam_R_in_current[:, 0]
        first_cam_y_axis = first_cam_R_in_current[:, 1]
        
        # Center points on reference camera
        points_centered = points_cam - first_cam_center
        
        # Apply azimuth rotation (around Y axis)
        if abs(azim_angle) > 1e-6:
            R_azim = R.from_rotvec(np.radians(azim_angle) * first_cam_y_axis).as_matrix()
            points_centered = (R_azim @ points_centered.T).T
        
        # Apply elevation rotation (around X axis)
        if abs(elev_angle) > 1e-6:
            R_elev = R.from_rotvec(np.radians(elev_angle) * first_cam_x_axis).as_matrix()
            points_centered = (R_elev @ points_centered.T).T
        
        # Move points back
        points_cam = points_centered + first_cam_center
        
        # Apply same rotation to cameras
        R_azim_full = R.from_rotvec(np.radians(azim_angle) * first_cam_y_axis).as_matrix() if abs(azim_angle) > 1e-6 else np.eye(3)
        R_elev_full = R.from_rotvec(np.radians(elev_angle) * first_cam_x_axis).as_matrix() if abs(elev_angle) > 1e-6 else np.eye(3)
        R_rel = R_elev_full @ R_azim_full
        
        # Rotate camera positions and poses
        rotated_camera_centers = []
        rotated_camera_poses = []
        for i, (cam_center, cam_pose) in enumerate(zip(camera_centers, camera_poses)):
            cam_center_in_current = (R_wc @ cam_center.T).T + t_wc
            cam_center_flipped = (flip_transform @ cam_center_in_current.T).T
            
            cam_center_centered = cam_center_flipped - first_cam_center
            cam_center_rotated = (R_rel @ cam_center_centered.T).T
            cam_center_final = cam_center_rotated + first_cam_center
            
            cam_R = cam_pose[:3, :3]
            cam_R_in_current = R_wc @ cam_R
            cam_R_flipped = flip_transform @ cam_R_in_current
            cam_R_rotated = R_rel @ cam_R_flipped
            
            rotated_pose = np.eye(4)
            rotated_pose[:3, :3] = cam_R_rotated
            rotated_pose[:3, 3] = cam_center_final
            
            rotated_camera_centers.append(cam_center_final)
            rotated_camera_poses.append(rotated_pose)
        
        camera_centers = np.array(rotated_camera_centers)
        camera_poses = np.array(rotated_camera_poses)
    
    # Apply view filtering in camera view mode
    if camera_view:
        point_directions = points_cam / (np.linalg.norm(points_cam, axis=1, keepdims=True) + 1e-8)
        camera_forward = np.array([0, 0, -1])
        cos_angles = point_directions @ camera_forward
        
        fov_angle_threshold = np.cos(np.radians(110))
        fov_mask = cos_angles > fov_angle_threshold
        
        if np.sum(fov_mask) > 0:
            points_cam = points_cam[fov_mask]
            colors_sample = colors_sample[fov_mask]
    
    # Compute point cloud range for scaling
    lower_percentile = np.percentile(points_cam, 7, axis=0)
    upper_percentile = np.percentile(points_cam, 93, axis=0)
    
    x_range = upper_percentile[0] - lower_percentile[0]
    y_range = upper_percentile[1] - lower_percentile[1]
    z_range = upper_percentile[2] - lower_percentile[2]
    max_range = max(x_range, y_range, z_range)
    
    # Compute point size
    if max_range > 0:
        base_point_size = max(0.03, min(0.15, 40.0 / max_range))
        if camera_view:
            point_size = base_point_size * 2.5
            alpha = 0.9
        else:
            point_size = base_point_size
            alpha = 0.8
    else:
        point_size = 1.0
        alpha = 0.8
        if camera_view:
            point_size = 2.5
            alpha = 0.9
    
    # Create figure
    fig = plt.figure(figsize=(12, 10), dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False
    
    # Plot point cloud
    colors_normalized = np.clip(colors_sample, 0, 1)
    scatter = ax.scatter(
        points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
        c=colors_normalized,
        s=point_size,
        alpha=alpha, 
        edgecolors='none',
        depthshade=True,
        linewidth=0
    )
    
    # Draw cameras
    show_cameras_in_view = show_all_cameras and not camera_view
    
    if show_cameras_in_view:
        x_min_cam, x_max_cam, y_min_cam, y_max_cam, z_min_cam, z_max_cam = _draw_cameras_visualization(
            ax, camera_centers, camera_poses, cam_idx, max_range, show_cameras=True
        )
        
        if x_min_cam is not None:
            x_min = min(points_cam[:, 0].min(), x_min_cam)
            x_max = max(points_cam[:, 0].max(), x_max_cam)
            y_min = min(points_cam[:, 1].min(), y_min_cam)
            y_max = max(points_cam[:, 1].max(), y_max_cam)
            z_min = min(points_cam[:, 2].min(), z_min_cam)
            z_max = max(points_cam[:, 2].max(), z_max_cam)
        else:
            x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
            y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
            z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()
    else:
        x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
        y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
        z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()
    
    # Set view angle
    if camera_view:
        ax.view_init(elev=0.0, azim=-90.0)
        ax.dist = 7
    else:
        ax.view_init(elev=0.0, azim=-90.0)
        ax.dist = 10
    
    # Set axis limits
    margin_factor = 0.02
    x_margin = (x_max - x_min) * margin_factor
    y_margin = (y_max - y_min) * margin_factor
    z_margin = (z_max - z_min) * margin_factor
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)
    
    # Style axes
    ax.set_xlabel('X', fontsize=14, fontweight='bold', color='red')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold', color='green')
    ax.set_zlabel('Z', fontsize=14, fontweight='bold', color='blue')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.set_facecolor('lightgray')
    
    # Save to buffer
    buf = io.BytesIO()
    try:
        fig.canvas.draw()
        plt.savefig(buf, format='png', dpi=500, bbox_inches='tight', 
                   pad_inches=0.05, facecolor='white', edgecolor='none',
                   transparent=False)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        img_b64 = ""
    finally:
        plt.close(fig)
        buf.close()
    
    return img_b64


def generate_custom_angle_views(predictions, azimuth_angle, elevation_angle, 
                                points_filtered=None, colors_filtered=None,
                                rotation_reference_camera=1, camera_view=False):
    """
    Generate view images at custom angles
    """
    points_sample, colors_sample, camera_centers, camera_poses = _prepare_points_and_cameras(
        predictions, points_filtered, colors_filtered
    )
    
    # Use reference camera (1-based to 0-based)
    cam_idx = max(0, min(int(rotation_reference_camera) - 1, len(camera_centers) - 1))
    
    view_images = []
    view_name = f"custom_azim_{azimuth_angle}_elev_{elevation_angle}"
    
    # Compensate for coordinate flip
    adjusted_elevation = elevation_angle + 100.0
    
    img_b64 = _create_view_image(
        points_sample, colors_sample, camera_centers, camera_poses,
        cam_idx, azimuth_angle, adjusted_elevation, view_name, 
        show_camera_axes=False, show_all_cameras=True,
        ref_cam_idx=cam_idx, camera_view=camera_view
    )
    
    view_images.append({
        "camera": cam_idx + 1,
        "view": view_name,
        "azimuth_angle": azimuth_angle,
        "elevation_angle": elevation_angle,
        "image": img_b64
    })
    
    return view_images


def generate_camera_views(predictions, max_views_per_camera=7, 
                         points_filtered=None, colors_filtered=None,
                         rotation_reference_camera=1, camera_view=False):
    """
    Generate multiple view images
    """
    points_sample, colors_sample, camera_centers, camera_poses = _prepare_points_and_cameras(
        predictions, points_filtered, colors_filtered
    )
    
    # View angles
    view_angles = [
        (0, 0, "camera_front"),
        (-45, 0, "camera_left_45"),
        (45, 0, "camera_right_45"),
        (0, -45, "camera_front_down"),
        (0, 45, "camera_front_up"),
    ]
    
    view_images = []
    
    for cam_idx in range(min(4, len(camera_centers))):
        limited_view_angles = view_angles[:max_views_per_camera]
        
        for azim_offset, elev_offset, view_name in limited_view_angles:
            adjusted_elev = elev_offset + 100.0
            
            img_b64 = _create_view_image(
                points_sample, colors_sample, camera_centers, camera_poses,
                cam_idx, azim_offset, adjusted_elev, view_name, 
                show_camera_axes=False, show_all_cameras=True,
                ref_cam_idx=cam_idx, camera_view=camera_view
            )
            
            view_images.append({
                "camera": cam_idx + 1,
                "view": view_name,
                "image": img_b64
            })
    
    return view_images


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(next(model.parameters()).device) if model is not None else None
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/test', methods=['GET'])
def test():
    """Test endpoint with synthetic data"""
    global model
    
    try:
        logger.info("Creating test images...")
        test_images = []
        img_size = 280  # Must be divisible by 14
        
        for i in range(3):
            test_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            for j in range(img_size):
                test_image[:, j, 0] = (i * 80 + j) % 256
                test_image[:, j, 1] = (i * 60 + j) % 256
                test_image[:, j, 2] = (i * 40 + j) % 256
            test_images.append(test_image)
        
        logger.info("Running test inference...")
        predictions = run_inference(test_images, apply_mask=True)
        
        logger.info("Test inference successful")
        return jsonify({
            "success": True,
            "message": "Test inference successful",
            "points_count": len(predictions["world_points"]),
            "num_cameras": len(predictions["camera_poses"])
        })
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/infer', methods=['POST'])
def infer():
    """Main inference endpoint"""
    global model
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'images' not in data:
            return jsonify({"error": "Missing 'images' field"}), 400
        
        # Get parameters
        # conf_percentile: confidence percentile threshold (0-100, default 10)
        # This filters out the bottom N% of points by confidence
        # Note: MapAnything uses percentile (not fixed threshold like Pi3) because
        # MapAnything's conf values are in range [1, +∞), not [0, 1]
        conf_percentile = data.get('conf_percentile', 30)
        # Also accept conf_threshold for backward compatibility (convert to percentile)
        if 'conf_threshold' in data and 'conf_percentile' not in data:
            # conf_threshold was typically 0.08, map to percentile ~10
            conf_percentile = 30
        generate_views = data.get('generate_views', True)
        max_views_per_camera = data.get('max_views_per_camera', 7)
        remove_outliers_flag = data.get('remove_outliers', True)
        azimuth_angle = data.get('azimuth_angle', None)
        elevation_angle = data.get('elevation_angle', None)
        rotation_reference_camera = data.get('rotation_reference_camera', 1)
        camera_view = data.get('camera_view', False)
        image_names = data.get('image_names', [])
        apply_mask = data.get('apply_mask', True)
        
        # Decode images
        images = []
        for img_b64 in data['images']:
            image_bytes = base64.b64decode(img_b64)
            image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Ensure size is divisible by patch size
            h, w = image_bgr.shape[:2]
            patch_size = HIGH_LEVEL_CONFIG["patch_size"]
            new_h = ((h + patch_size - 1) // patch_size) * patch_size
            new_w = ((w + patch_size - 1) // patch_size) * patch_size
            
            if new_h != h or new_w != w:
                image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            images.append(image_bgr)
        
        logger.info(f"Received {len(images)} images for inference")
        
        # Run inference
        logger.info("Running MapAnything inference...")
        predictions = run_inference(images, apply_mask=apply_mask, conf_percentile=conf_percentile)
        
        # Get points and colors
        points_filtered = predictions["world_points"]
        colors_filtered = predictions["colors"]
        
        # Remove outliers if enabled (same as Pi3)
        if remove_outliers_flag:
            logger.info("Removing outliers (Mahalanobis method)...")
            points_filtered, colors_filtered, _ = remove_outliers_mahalanobis(
                points_filtered, colors_filtered, threshold_std=2.5  # Same as Pi3
            )
        
        # Generate filename
        import hashlib
        if image_names and len(image_names) > 0:
            scene_id = os.path.splitext(os.path.basename(image_names[0]))[0]
            safe_name = "".join(c for c in scene_id if c.isalnum() or c in ('-', '_'))
            ply_filename = f"mapanything_{safe_name}.ply"
        else:
            first_img_bytes = data['images'][0].encode('utf-8')
            img_hash = hashlib.md5(first_img_bytes).hexdigest()[:8]
            ply_filename = f"mapanything_{img_hash}.ply"
        
        ply_path = f"outputs/{ply_filename}"
        os.makedirs("outputs", exist_ok=True)
        
        # Write PLY file
        ply_b64 = write_ply(points_filtered, colors_filtered, ply_path)
        
        # Extract camera pose information
        camera_poses_raw = predictions["camera_poses"]
        camera_poses_list = []
        
        if len(camera_poses_raw) > 0:
            R_ref = camera_poses_raw[0][:3, :3]
            
            for i, pose in enumerate(camera_poses_raw):
                R_cw = pose[:3, :3]
                t_cw = pose[:3, 3]
                
                # Compute relative rotation
                R_relative = R_cw @ R_ref.T
                rotation_relative = R.from_matrix(R_relative)
                
                try:
                    euler_yx = rotation_relative.as_euler('yx', degrees=True)
                    azimuth_from_cam1 = euler_yx[0]
                    elevation_from_cam1 = euler_yx[1]
                except:
                    euler_xyz_rel = rotation_relative.as_euler('xyz', degrees=True)
                    azimuth_from_cam1 = euler_xyz_rel[1]
                    elevation_from_cam1 = euler_xyz_rel[0]
                
                camera_poses_list.append({
                    "camera_id": i + 1,
                    "position": t_cw.tolist(),
                    "azimuth_angle": float(azimuth_from_cam1),
                    "elevation_angle": float(elevation_from_cam1)
                })
        
        # Prepare predictions dict for view generation
        predictions["world_points"] = points_filtered
        predictions["colors"] = colors_filtered
        
        response_data = {
            "success": True,
            "ply_file": ply_b64,
            "ply_filename": ply_filename,
            "points_count": len(points_filtered),
            "camera_poses": camera_poses_list,
            "camera_views": []
        }
        
        # Generate view images
        if generate_views:
            try:
                if azimuth_angle is not None and elevation_angle is not None:
                    view_mode = "camera view" if camera_view else "global view"
                    logger.info(f"Generating custom angle views: azim={azimuth_angle}°, elev={elevation_angle}°, mode={view_mode}")
                    view_images = generate_custom_angle_views(
                        predictions, azimuth_angle, elevation_angle,
                        points_filtered, colors_filtered,
                        rotation_reference_camera=rotation_reference_camera,
                        camera_view=camera_view
                    )
                else:
                    logger.info(f"Generating default views, ref_camera={rotation_reference_camera}")
                    view_images = generate_camera_views(
                        predictions, max_views_per_camera,
                        points_filtered, colors_filtered,
                        rotation_reference_camera=rotation_reference_camera,
                        camera_view=camera_view
                    )
                response_data["camera_views"] = view_images
            except Exception as e:
                logger.warning(f"Failed to generate view images: {e}")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("MapAnything inference completed")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MapAnything Inference Server')
    parser.add_argument('--port', type=int, default=20022,
                        help='Port to run the server on (default: 20022)')
    
    args = parser.parse_args()
    
    logger.info("Starting MapAnything server...")
    logger.info(f"Server port: {args.port}")
    
    # Load model
    if not load_model():
        logger.error("Failed to load model, exiting")
        exit(1)
    
    logger.info("Model loaded, starting server...")
    app.run(host='0.0.0.0', port=args.port, debug=False)
