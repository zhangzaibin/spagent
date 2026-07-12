---
name: supervision_tool
description: Run YOLO-based object detection or instance segmentation with Supervision visualization.
category: detection
group: 2d_perception
runtime: server
catalog_key: supervision
---

# supervision_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Run YOLO-based object detection or instance segmentation with Supervision visualization.

When to use: standard detection/segmentation with annotated output images; task='image_det' for boxes, task='image_seg' for masks.
When NOT to use: open-vocabulary text queries (prefer detect_objects_tool) or custom class lists via YOLO-E (prefer yoloe_detection_tool).
Example: task='image_det' on image_path='room.jpg' to list detected objects with boxes.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | The path to the input image for processing. |
| `task` | string (image_det \\| image_seg) | yes | — | The task type: 'image_det' for object detection or 'image_seg' for segmentation. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

**Dual-behavior tool:** the output category is resolved per call from the `task` argument — `task='image_det'` returns the detection contract, `task='image_seg'` returns the segmentation contract.

### Category `detection`

Raw payload — any ONE of these carrier groups must be present:
- `detections`
- `boxes` + `labels`

Common optional fields: `confidence`, `box_format`, `image_width`, `image_height`, `class_id`, `crop_paths`, `masks`

Default render projection: `labels`, `boxes`, `confidence`

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
python -m spagent.skills.run supervision_tool --args '{"image_path": "assets/dog.jpeg", "task": "image_det"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:8000`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:8000` (health check: `GET /health`)
- Launch: `python spagent/external_experts/supervision/supervision_server.py`
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: YOLO/Supervision annotation server on port 8000.

## Guidance (curated)

# Curated guidance for supervision (harvested from PR #157; facts above are generated)

### When to use
- Quick object detection with bounding boxes (task='image_det')
- Instance segmentation with pixel-level masks (task='image_seg')
- General scene understanding when you need to identify all objects

### Tips
- This tool uses a fixed YOLO vocabulary; for open-vocabulary text queries
  prefer detect_objects_tool, for custom class lists prefer
  yoloe_detection_tool.
