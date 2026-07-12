---
name: qwenvl_detection_tool
description: Detect objects in an image using Qwen VL 2.5.
category: detection
group: 2d_perception
runtime: cloud-API
catalog_key: qwenvl
---

# qwenvl_detection_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Detect objects in an image using Qwen VL 2.5. Supports referring detection (locate objects matching a text description) and reasoning detection (detect objects relevant to a reasoning question). Returns bounding boxes with normalized coordinates [0,1].

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Path or URL of the input image. |
| `text_prompt` | string | yes | — | For ref_detection: object description to locate (e.g. 'red car'). For reasoning_detection: a reasoning question about the scene. |
| `task` | string (ref_detection \\| reasoning_detection) | no | ref_detection | Detection mode. Default is 'ref_detection'. |

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
python -m spagent.skills.run qwenvl_detection_tool --args '{"image_path": "assets/dog.jpeg", "text_prompt": "dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.

## Runtime requirements

- Runtime class: **cloud-API**
- Requires: DashScope API key (constructor arg `api_key`)
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: Third-party VLM detection API; without a key only --use-mock works.
