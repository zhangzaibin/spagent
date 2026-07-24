---
name: video_generation_vace_tool
description: Generate a video from one reference image and a text prompt via the local VACE first-frame pipeline; returns the path to the generated .mp4.
category: video_generation
group: generation
runtime: server
catalog_key: vace
---

# video_generation_vace_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Generate a video from one reference image and a text prompt via the local VACE first-frame pipeline; returns the path to the generated .mp4. Hard rule: output exactly one <tool_call> for this tool per assistant turn—never multiple tool_use / <tool_call> blocks for this tool in the same response. This tool is very slow and GPU-heavy (minutes per call); a second call in one turn is forbidden. When several frames or views exist, pick the single most critical image before that one call; do not sweep many frames. The prompt should describe the desired motion or outcome for that first frame. Use for controlled motion, camera change, or short temporal animation—not for bulk frame analysis.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Path to the first-frame reference image (one file). If several views or frames exist, pick the most important frame and pass its path here. |
| `prompt` | string | yes | — | Motion prompt describing how the generated video should move. What you want the video to do (motion, viewpoint change, etc.); the rollout follows this instruction. |
| `base` | string | no | wan | VACE base model backend. Default: 'wan'. |
| `task` | string | no | frameref | VACE task name. Default: 'frameref'. |
| `mode` | string | no | firstframe | VACE mode for first-frame generation. Default: 'firstframe'. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `video_generation`

Raw payload — any ONE of these carrier groups must be present:
- `output_path`

Common optional fields: `duration`, `resolution`, `fps`, `codec`, `result_dir`, `frame_paths`

Default render projection: `output_path`

## Invocation

```bash
python -m spagent.skills.run video_generation_vace_tool --args '{"image_path": "assets/dog.jpeg", "prompt": "a dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20034`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:20034` (health check: `GET /health`)
- Launch: `python spagent/external_experts/vace/vace_server.py --port 20034`
- Checkpoint: `spagent/external_experts/vace/models/Wan2.1-VACE-1.3B`
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: Heavy Wan2.1-VACE stack (large download, big GPU); see spagent/external_experts/vace/README.md.
