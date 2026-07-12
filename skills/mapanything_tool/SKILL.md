---
name: mapanything_tool
description: This tool is suitable for motion and spatial reasoning tasks that involve camera movement, object rotation, or directional motion analysis.
category: 3d_reconstruction
group: 3d
runtime: server
catalog_key: mapanything
---

# mapanything_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

This tool is suitable for motion and spatial reasoning tasks that involve camera movement, object rotation, or directional motion analysis. It performs 3D reconstruction from images using MapAnything to generate point clouds and visualizations from CUSTOM viewing angles.

**Important Note**: The 0° azimuth angle and 0° elevation angle corresponds to the first input image viewpoint (cam1). Do not use this angle.

**Angle Parameters**:
- **azimuth_angle** (-180° to 180°, integer only): Controls left-right rotation.
- **elevation_angle** (-90° to 90°, integer only): Controls up-down rotation.
By convention, (azimuth=0, elevation=0) corresponds EXACTLY to the first input image viewpoint (cam1). All rotations are defined in the INPUT CAMERA coordinate frame: azimuth rotates left/right around the camera's vertical axis; elevation rotates up/down around the camera's right axis.

**rotation_reference_camera** (must be output, 1-based): This parameter is used to rotate around a specific input image's camera. By picking an image you pick its camera (e.g., set rotation_reference_camera=3 for the third image's viewpoint; defaults to 1).

**camera_view** (must be output, boolean): This parameter is used to generate first-person perspective from the selected camera position (as if standing at that camera looking at the scene), instead of the default global bird's-eye view. This is especially useful for understanding what each camera can see and analyzing spatial relationships from specific viewpoints. Combine with rotation_reference_camera to experience the scene from different camera positions.

Note that default camera_view is false. You must output camera_view = true if you want to set ego-view. If you want to set global-view, you must output camera_view = false.

**Usage Strategy**: You can call this tool MULTIPLE times with DIFFERENT angles and different camera views to analyze the 3D structure comprehensively. The MLLM is encouraged to autonomously explore angles (coarse-to-fine) until sufficient evidence is gathered. The generated visualization uses cone-shaped markers to indicate camera positions, numbered from 1 (cam1, cam2, etc.).

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | list | yes | — | The list of the path to the input images for 3D reconstruction. |
| `azimuth_angle` | number | no | — | Azimuth angle (left-right rotation) in degrees for custom viewpoint generation. Range: -180 to 180. Default is 0 (front view). Negative values rotate left, positive values rotate right. |
| `elevation_angle` | number | no | — | Elevation angle (up-down rotation) in degrees for custom viewpoint generation. Range: -90 to 90. Default is 0 (horizontal). Negative values look down, positive values look up. |
| `rotation_reference_camera` | integer | no | — | Reference camera index (1-based) to define rotation center and axes when generating viewpoints. When you have multiple input images, try DIFFERENT values (1, 2, 3, etc.) to rotate around different camera positions for better analysis. Default is 1 (uses the first input camera). |
| `camera_view` | boolean | no | — | Whether to use first-person camera view mode. When True, generates point cloud visualization from the selected camera's first-person perspective (as if you are standing at that camera position looking at the scene). When False (default), uses global bird's-eye view. Combine with rotation_reference_camera to view from different camera positions. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `3d_reconstruction`

Raw payload — any ONE of these carrier groups must be present:
- `ply_filename`
- `points`

Common optional fields: `points_count`, `view_count`, `camera_views`, `camera_poses`, `mesh_path`, `scale_info`

Default render projection: `ply_filename`, `points_count`

## Invocation

```bash
python -m spagent.skills.run mapanything_tool --args '{"image_path": ["assets/dog.jpeg"]}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20033`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:20033` (health check: `GET /health`)
- Launch: `python spagent/external_experts/mapanything/mapanything_server.py --port 20033`
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: facebook/map-anything auto-downloads from HF on first launch.
