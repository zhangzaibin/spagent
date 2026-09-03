---
name: localize_object_tool
description: Locate objects in an image by detecting them and drawing bounding boxes on the full scene.
category: detection
group: 2d_perception
runtime: server
catalog_key: localize
---

# localize_object_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Locate objects in an image by detecting them and drawing bounding boxes on the full scene. Use this tool when you need to understand WHERE objects are or HOW MANY there are.

WHEN TO USE (spatial / counting questions):
- "How many X are in the image?" → count instances
- "Is X to the left/right/above/below Y?" → understand layout
- "Which X is closest to Y?" → relative positioning
- "Where is the X?" → find its location in the scene
- Understanding the overall arrangement of multiple objects

WHEN NOT TO USE:
- COLOR / TEXTURE / MATERIAL / TEXT questions → use zoom_object_tool
- Mood/emotion, abstract scene-level concepts
- When you need pixel masks (prefer segment_image_tool)

Key feature: returns the full image with labeled bounding boxes, plus a text summary of each detection's normalized position (e.g. 'center at x=0.42, y=0.61, covering 18%×22% of image').

text_prompt rules:
- Specify AT MOST 2 object names separated by '.' (e.g. 'person' or 'car . truck')
- If nothing found: retry with a synonym

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
python -m spagent.skills.run localize_object_tool --args '{"image_path": "assets/dog.jpeg", "text_prompt": "dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20022`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:20022` (health check: `GET /health`)
- Launch: `python spagent/external_experts/GroundingDINO/grounding_dino_server.py --checkpoint_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth --port 20022`
- Checkpoint: `checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth`
- Mock available: yes (`--use-mock`, no GPU/server needed)
