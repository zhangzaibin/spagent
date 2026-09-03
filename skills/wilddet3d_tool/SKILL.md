---
name: wilddet3d_tool
description: WildDet3D: promptable 3D object detection from a single RGB image.
category: detection
group: 3d
runtime: mock-only
catalog_key: wilddet3d
---

# wilddet3d_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

WildDet3D: promptable 3D object detection from a single RGB image. Given a text prompt (e.g. 'chair', 'car', 'object'), detects and localizes objects in both 2D and 3D. Returns an annotated image with bounding boxes and 3D location estimates. Useful for object localization, spatial understanding, and scene analysis tasks.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Path to the input RGB image. |
| `prompt_text` | string | no | — | Text prompt describing the object(s) to detect. Examples: 'chair', 'person', 'car', 'object' (detects all objects). Ignored when input_boxes or input_points are provided. Default: 'object'. |
| `input_boxes` | array | no | — | Optional 2D bounding box prompt [x1, y1, x2, y2] in pixel coordinates. Use when you already know the approximate region of interest. Takes priority over prompt_text. |
| `input_points` | array | no | — | Optional point prompts [[x, y, label], ...] in pixel coordinates. label=1 for foreground (object of interest), label=0 for background. Takes priority over prompt_text. |

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
python -m spagent.skills.run wilddet3d_tool --args '{"image_path": "assets/dog.jpeg"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.

## Runtime requirements

- Runtime class: **mock-only**
- Requires: WILDDET3D_ROOT env var (upstream WildDet3D repo checkout)
- Requires: optional WILDDET3D_CHECKPOINT env var
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: The real model is not vendored in this repo; set WILDDET3D_ROOT to enable it, otherwise only --use-mock works.
