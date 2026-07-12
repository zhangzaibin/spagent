---
name: image_generation_sana_tool
description: Generate an image from a text prompt using Sana.
category: image_generation
group: generation
runtime: server
catalog_key: sana
---

# image_generation_sana_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Generate an image from a text prompt using Sana. Use this when you need to visualize a hypothetical scene, target state, plan outcome, or imagined world state. The generated image is synthetic and should be treated as a visualization rather than direct evidence from the original observation.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | Text prompt describing the image to generate. Be explicit about the scene, objects, layout, and desired appearance. |
| `size` | string (512x512 \\| 1024x1024) | no | 1024x1024 | Output image size. Default is '1024x1024'. |
| `num_inference_steps` | integer | no | 2 | Number of diffusion inference steps. Default is 2 for Sana-Sprint. |
| `guidance_scale` | number | no | 4.5 | Classifier-free guidance scale. Default is 4.5. |
| `seed` | integer | no | 42 | Random seed for reproducibility. Default is 42. |
| `negative_prompt` | string | no | — | Optional negative prompt describing what should be avoided. |
| `n` | integer | no | 1 | Number of images to generate. Default is 1. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `image_generation`

Raw payload — any ONE of these carrier groups must be present:
- `output_path`
- `image_paths`

Common optional fields: `seed`, `model`, `size`, `file_size_bytes`

Default render projection: `output_path`

## Invocation

```bash
python -m spagent.skills.run image_generation_sana_tool --args '{"prompt": "a dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:30000`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:30000` (health check: `GET /health`)
- Launch: `python spagent/external_experts/Sana/sana_server.py --port 30000`
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: If port 30000 is taken, launch on another port and pass --server-url to spagent.skills.run.
