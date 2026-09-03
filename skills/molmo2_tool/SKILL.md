---
name: molmo2_tool
description: Molmo2 point-grounding tool.
category: point_grounding
group: vlm
runtime: server
catalog_key: molmo2
---

# molmo2_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Molmo2 point-grounding tool. Given a natural-language instruction, it locates the described object or region in the image and returns an annotated overlay image showing the exact position with a marked point. Always use a short reasoning sentence as the prompt, e.g. 'Point to the object the robot should grasp next.' or 'Point to the item that does not belong with the others.'

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Path to the input image. |
| `prompt` | string | yes | — | A short reasoning sentence describing what to point to. Do NOT just name an object — phrase it as a task, e.g. 'Point to the object the robot should grasp next.' or 'Point to the ripest fruit on the table.' or 'Point to the item that is out of place.' This lets the model apply scene understanding before pointing. |
| `save_annotated` | boolean | no | True | Save a JPEG overlay with the marked point(s). Default true. |
| `max_new_tokens` | integer | no | 200 |  |

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
python -m spagent.skills.run molmo2_tool --args '{"image_path": "assets/dog.jpeg", "prompt": "a dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20025`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:20025` (health check: `GET /health`)
- Launch: `python spagent/external_experts/Molmo2/molmo2_server.py --port 20025`
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: HF weights download on first launch.
