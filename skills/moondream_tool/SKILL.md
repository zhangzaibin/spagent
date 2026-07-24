---
name: moondream_tool
description: Lightweight vision-language tasks on one image: captioning, visual Q&A, object detection, and pointing via Moondream.
category: point_grounding
group: vlm
runtime: cloud-API
catalog_key: moondream
---

# moondream_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Lightweight vision-language tasks on one image: captioning, visual Q&A, object detection, and pointing via Moondream.

When to use: quick language-driven understanding, short answers about image content, or simple grounding without heavy 3D reconstruction.
When NOT to use: precise metric 3D pose (prefer orient_anything_v2_tool), dense depth maps, or multi-view reconstruction.
Example: task='vqa' with a question like 'What color is the car on the left?'.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | The path to the input image. |
| `task` | string (point) | yes | — | The task to perform: point (locate objects - supports both single objects like 'car' and multiple objects like 'car, person, tree') |
| `object_name` | string | yes | — | Name of the object(s) to locate. Can be a single object like 'car' or multiple objects separated by commas like 'car, person, tree' |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `point_grounding`

Raw payload — any ONE of these carrier groups must be present:
- `points`
- `points_by_image`

Common optional fields: `confidence`, `labels`, `image_width`, `image_height`, `raw_text`

Default render projection: `points`

## Invocation

```bash
python -m spagent.skills.run moondream_tool --args '{"image_path": "assets/dog.jpeg", "task": "point", "object_name": "dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20024`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **cloud-API**
- Server: `http://127.0.0.1:20024` (health check: `GET /health`)
- Requires: Moondream provider API key
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: Served via spagent/external_experts/moondream/md_server.py (port 20024), which itself needs a provider API key; without one only --use-mock works.

## Guidance (curated)

# Curated guidance for moondream (harvested from PR #157; facts above are generated)

### When to use
- Finding the exact position of named objects in an image (pointing)
- Locating multiple objects at once (comma-separated names,
  e.g. 'car, person, tree')
- Getting pixel coordinates for spatial reasoning
