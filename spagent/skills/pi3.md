---
name: pi3
title: Pi3 - 3D Reconstruction
summary: >
  3D reconstruction from images. Generate novel viewpoints by specifying
  azimuth/elevation angles to understand spatial structure from any direction.
tool_name: pi3_tool
---

## Pi3 - 3D Reconstruction Tool

Reconstruct a 3D point cloud from input images and render it from arbitrary viewpoints.

### When to Use
- Understanding 3D spatial relationships between objects
- Viewing a scene from angles not available in the original images
- Analyzing relative positions, orientations, and distances in 3D space
- Questions about what is behind, above, below, or beside objects

### Key Concepts
- The input images are at (azimuth=0, elevation=0). Do NOT request (0,0) again.
- azimuth: left-right rotation (-180 to 180). Negative=left, positive=right.
- elevation: up-down rotation (-90 to 90). Negative=down, positive=up.
- rotation_reference_camera: which camera to rotate around (1-based index)
- camera_view: True=first-person from that camera, False=bird's-eye view

### Recommended Angles
- Left: azimuth=-45 or -90
- Right: azimuth=45 or 90
- Top: elevation=30 to 60 (great for relative positioning)
- Back: azimuth=180 or +/-135
- Diagonal: combine azimuth and elevation, e.g. (45, 30)

### Parameters (Inference Mode)

All 5 parameters are available during inference:

- **image_path** (list, required): The list of the path to the input images for 3D reconstruction.
- **azimuth_angle** (number, optional): Azimuth angle (left-right rotation) in degrees for custom viewpoint generation. Range: -180 to 180. Default is 0 (front view). Negative values rotate left, positive values rotate right.
- **elevation_angle** (number, optional): Elevation angle (up-down rotation) in degrees for custom viewpoint generation. Range: -90 to 90. Default is 0 (horizontal). Negative values look down, positive values look up.
- **rotation_reference_camera** (integer, optional): Reference camera index (1-based) to define rotation center and axes when generating viewpoints. When you have multiple input images, try DIFFERENT values (1, 2, 3, etc.) to rotate around different camera positions for better analysis. Default is 1 (uses the first input camera).
- **camera_view** (boolean, optional): Whether to use first-person camera view mode. When True, generates point cloud visualization from the selected camera's first-person perspective (as if you are standing at that camera position looking at the scene). When False (default), uses global bird's-eye view. Combine with rotation_reference_camera to view from different camera positions.

### Parameters (Training Mode)

Only 3 parameters are available during training (rotation_reference_camera and camera_view are NOT supported):

- **image_path** (list, required): The list of the path to the input images for 3D reconstruction.
- **azimuth_angle** (number, optional): Azimuth angle (left-right rotation) in degrees for custom viewpoint generation. Range: -180 to 180. Default is 0 (front view). Negative values rotate left, positive values rotate right.
- **elevation_angle** (number, optional): Elevation angle (up-down rotation) in degrees for custom viewpoint generation. Range: -90 to 90. Default is 0 (horizontal). Negative values look down, positive values look up.

### Call Format (Inference)
<tool_call>
{"name": "pi3_tool", "arguments": {"image_path": "<image_path>", "azimuth_angle": "<azimuth_angle>", "elevation_angle": "<elevation_angle>", "rotation_reference_camera": "<rotation_reference_camera>", "camera_view": "<camera_view>"}}
</tool_call>

### Call Format (Training)
<tool_call>
{"name": "pi3_tool", "arguments": {"image_path": "<image_path>", "azimuth_angle": "<azimuth_angle>", "elevation_angle": "<elevation_angle>"}}
</tool_call>

### Output
Returns a rendered 3D point cloud image from the specified viewpoint.
