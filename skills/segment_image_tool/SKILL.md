---
name: segment_image_tool
description: Segment objects or regions in an image using SAM2.
category: segmentation
group: 2d_perception
runtime: server
catalog_key: segmentation
---

# segment_image_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Segment objects or regions in an image using SAM2. Supports point, box, or automatic segmentation guided by the user's request.

When to use: you need precise pixel-level masks, object boundaries, or region isolation.
When NOT to use: only bounding boxes are needed (prefer detect_objects_tool or yolo26_tool), or the task is purely depth/viewpoint reasoning.
Example: segment the red cup after detecting its approximate location with a box prompt.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | The path to the input image for segmentation. |
| `point_coords` | array | no | — | Optional list of point coordinates [[x1,y1], [x2,y2], ...] |
| `point_labels` | array | no | — | Optional list of point labels (1 for foreground, 0 for background) |
| `box` | array | no | — | Optional bounding box coordinates [x1,y1,x2,y2] |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `segmentation`

Raw payload — any ONE of these carrier groups must be present:
- `masks`
- `mask_path`
- `polygon`
- `rle`

Common optional fields: `area`, `bbox`, `class_name`, `shape`, `image_width`, `image_height`

Default render projection: description + visualization images only.

## Invocation

```bash
python -m spagent.skills.run segment_image_tool --args '{"image_path": "assets/dog.jpeg"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20020`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:20020` (health check: `GET /health`)
- Launch: `python spagent/external_experts/SAM2/sam2_server.py --checkpoint_path checkpoints/sam2/sam2.1_b.pt --port 20020`
- Checkpoint: `checkpoints/sam2/sam2.1_b.pt`
- Mock available: yes (`--use-mock`, no GPU/server needed)

## Guidance (curated)

# Curated guidance for segmentation (harvested from PR #157; facts above are generated)

### When to use
- Isolating specific objects or regions in a scene
- Counting distinct objects by their boundaries
- Analyzing object shapes, sizes, and spatial extent
- Comparing areas occupied by different objects

### Tips
- Guide the mask with `point_coords` + `point_labels` (1=foreground,
  0=background) or constrain it with a `box`; omit all three for automatic
  segmentation.
