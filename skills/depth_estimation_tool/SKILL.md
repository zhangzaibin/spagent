---
name: depth_estimation_tool
description: Generate a monocular depth map for one input image to analyze relative depth, near/far ordering, and 3D layout cues.
category: depth
group: 2d_perception
runtime: server
catalog_key: depth
---

# depth_estimation_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Generate a monocular depth map for one input image to analyze relative depth, near/far ordering, and 3D layout cues.

When to use: spatial relationship questions (closer/farther), occlusion reasoning, scene layout, or when depth ordering helps answer the question.
When NOT to use: object naming/counting (prefer detection), pixel masks (prefer segmentation), or novel camera viewpoints (prefer pi3/pi3x).
Example: call with image_path='scene.jpg' to compare which object is nearer to the camera.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | The path to the input image for depth estimation. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `depth`

Raw payload — any ONE of these carrier groups must be present:
- `depth_data`
- `depth_path`

Common optional fields: `shape`, `value_range`, `confidence_map`, `normal_map`

Default render projection: description + visualization images only.

## Invocation

```bash
python -m spagent.skills.run depth_estimation_tool --args '{"image_path": "assets/dog.jpeg"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20019`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:20019` (health check: `GET /health`)
- Launch: `python spagent/external_experts/Depth_AnythingV2/depth_server.py --checkpoint_path checkpoints/depth_anything/depth_anything_v2_vitb.pth --port 20019`
- Checkpoint: `checkpoints/depth_anything/depth_anything_v2_vitb.pth`
- Mock available: yes (`--use-mock`, no GPU/server needed)

## Guidance (curated)

# Curated guidance for depth (harvested from PR #157; facts above are generated)

### When to use
- Comparing relative distances of objects in the scene
- Understanding spatial layout and depth ordering
- Analyzing occlusion relationships between objects
- Determining which object is closer to or farther from the camera

### Reading the output
The depth map reveals 3D spatial structure: brighter regions are closer to
the camera, darker regions are farther away.
