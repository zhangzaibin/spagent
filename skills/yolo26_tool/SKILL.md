---
name: yolo26_tool
description: Run fast local YOLO26 object detection on one image.
category: detection
group: 2d_perception
runtime: local
catalog_key: yolo26
---

# yolo26_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Run fast local YOLO26 object detection on one image. Returns bounding boxes, class labels, and confidence scores; can save an annotated visualization.

When to use: quick detection of common object classes without writing a text prompt.
When NOT to use: rare/custom object names (prefer detect_objects_tool or yoloe_detection_tool), segmentation masks, or 3D reasoning.
Example: image_path='kitchen.jpg', conf=0.25 to list visible appliances and furniture.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Path to the input image. |
| `conf` | number | no | 0.25 | Confidence threshold for detections. |
| `save_annotated` | boolean | no | True | Whether to save an annotated visualization image. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `detection`

Raw payload — any ONE of these carrier groups must be present:
- `detections`
- `boxes` + `labels`

Common optional fields: `confidence`, `box_format`, `image_width`, `image_height`, `class_id`, `crop_paths`, `masks`

Default render projection: `labels`, `boxes`, `confidence`

## Invocation

```bash
python -m spagent.skills.run yolo26_tool --args '{"image_path": "assets/dog.jpeg"}'
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false.

## Runtime requirements

- Runtime class: **local**
- Checkpoint: `checkpoints/yolo26/yolo26n.pt`
- Requires: ultralytics
- Mock available: no
- Notes: Runs in-process (CPU or GPU); weights auto-download on first use. No mock mode.
