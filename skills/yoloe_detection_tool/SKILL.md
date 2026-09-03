---
name: yoloe_detection_tool
description: Detect objects with YOLO-E using user-specified custom class names.
category: detection
group: 2d_perception
runtime: server
catalog_key: yoloe
---

# yoloe_detection_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Detect objects with YOLO-E using user-specified custom class names. Returns bounding boxes (detection only, not segmentation).

When to use: you have a predefined class list and need accurate localization for those classes.
When NOT to use: free-form text queries (prefer detect_objects_tool), masks, or video generation.
Example: class_names=['helmet', 'vest'] on a construction-site image.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | The path to the input image for YOLO-E processing. |
| `task` | string (image \\| video) | yes | — | The processing task type: 'image' for single image object detection, or 'video' for video frame processing. |
| `class_names` | array | yes | — | List of object class names to detect (e.g., ['person', 'car', 'dog', 'cat']). YOLO-E can detect custom objects based on text descriptions. |

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
python -m spagent.skills.run yoloe_detection_tool --args '{"image_path": "assets/dog.jpeg", "task": "image", "class_names": ["dog"]}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:8000`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:8000` (health check: `GET /health`)
- Launch: `python spagent/external_experts/supervision/sv_yoloe_server.py`
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: YOLO-E server on port 8000 (weights per spagent/external_experts/supervision/download_weights.py).

## Guidance (curated)

# Curated guidance for yoloe (harvested from PR #157; facts above are generated)

### When to use
- Detecting specific object categories you define (custom vocabulary via
  `class_names`, e.g. ['person', 'car', 'dog'])
- High-speed detection on images or video frames

### Tips
- Detection only (bounding boxes) — it does not return segmentation masks.
