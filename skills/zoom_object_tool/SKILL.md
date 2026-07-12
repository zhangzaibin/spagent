---
name: zoom_object_tool
description: Zoom into a specific object by detecting it and returning a cropped close-up image.
category: detection
group: 2d_perception
runtime: server
catalog_key: zoom
---

# zoom_object_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Zoom into a specific object by detecting it and returning a cropped close-up image. Use this tool when you need to closely examine an object's fine-grained attributes.

WHEN TO USE (attribute inspection):
- "What COLOR is the X?" → zoom into X to see its color clearly
- "What does the X say / show?" → zoom in to read text or markings
- "What PATTERN / MATERIAL / SHAPE is the X?" → zoom in for texture detail
- Any question requiring magnified inspection of a specific object

WHEN NOT TO USE:
- WHERE/HOW MANY questions → use localize_object_tool instead
- Mood/emotion, abstract scene-level concepts
- When you need pixel masks (prefer segment_image_tool)

Key feature: the top detected regions are cropped with surrounding context and returned as high-resolution close-up images for analysis.

text_prompt rules:
- Specify AT MOST 2 object names separated by '.' (e.g. 'scarf' or 'helmet . person')
- Name the object you want to zoom into, not the whole scene
- If nothing found: retry with a synonym (e.g. 'motorbike' for 'motorcycle', 'bag' for 'handbag')

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Path to the input image. |
| `text_prompt` | string | yes | — | Object(s) to detect. Specify AT MOST 2 names separated by '.' — e.g. 'car' or 'car . truck'. Do NOT list more than 2 names; make multiple tool calls instead. |
| `box_threshold` | number | no | 0.35 | Confidence threshold for box detection. |
| `text_threshold` | number | no | 0.25 | Confidence threshold for text matching. |

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
python -m spagent.skills.run zoom_object_tool --args '{"image_path": "assets/dog.jpeg", "text_prompt": "dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20022`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:20022` (health check: `GET /health`)
- Launch: `python spagent/external_experts/GroundingDINO/grounding_dino_server.py --checkpoint_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth --port 20022`
- Checkpoint: `checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth`
- Mock available: yes (`--use-mock`, no GPU/server needed)
