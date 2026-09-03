---
name: video_generation_veo_tool
description: Generate a video from a text prompt (and optionally a reference image) using Google Veo.
category: video_generation
group: generation
runtime: cloud-API
catalog_key: veo
---

# video_generation_veo_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Generate a video from a text prompt (and optionally a reference image) using Google Veo. Returns the path to the generated .mp4 video file. Use this when the task requires creating a video visualization, animation, or video content from a description.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | Text description of the video to generate. |
| `image_path` | string | no | — | Optional path to a reference image for image-to-video generation. |
| `duration` | integer | no | 8 | Video duration in seconds (5 or 8). Default is 8. |
| `aspect_ratio` | string (16:9 \\| 9:16) | no | 16:9 | Aspect ratio: '16:9' (landscape) or '9:16' (portrait). Default is '16:9'. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `video_generation`

Raw payload — any ONE of these carrier groups must be present:
- `output_path`

Common optional fields: `duration`, `resolution`, `fps`, `codec`, `result_dir`, `frame_paths`

Default render projection: `output_path`

## Invocation

```bash
python -m spagent.skills.run video_generation_veo_tool --args '{"prompt": "a dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.

## Runtime requirements

- Runtime class: **cloud-API**
- Requires: Google Veo API key (constructor arg `api_key`)
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: Without a key only --use-mock works.
