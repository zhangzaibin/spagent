---
name: flowseek_tool
description: FlowSeek: optical flow estimation between two images.
category: optical_flow
group: 2d_perception
runtime: local
catalog_key: flowseek
---

# flowseek_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

FlowSeek: optical flow estimation between two images. Given a pair of images (e.g. consecutive video frames or a before/after pair), estimates the per-pixel motion field and returns a colorized visualization. The M variant uses ResNet-34 + ViT-B (higher accuracy); the T variant uses ResNet-18 + ViT-S (faster). Useful for motion analysis, video understanding, and dynamic scene tasks.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image1_path` | string | yes | — | Path to the first (source) image. |
| `image2_path` | string | yes | — | Path to the second (target) image. |
| `output_path` | string | no | — | Optional path to save the colorized flow image. Auto-generated under outputs/ if not specified. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `optical_flow`

Raw payload — any ONE of these carrier groups must be present:
- `flow_path`
- `flow_array`

Common optional fields: `flow_shape`, `flow_magnitude_mean`, `motion_boundaries`, `confidence_map`

Default render projection: `flow_magnitude_mean`, `flow_path`

## Invocation

```bash
python -m spagent.skills.run flowseek_tool --args '{"image1_path": "assets/dog.jpeg", "image2_path": "assets/dog.jpeg"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.

## Runtime requirements

- Runtime class: **local**
- Requires: FLOWSEEK_CHECKPOINT env var (FlowSeek weights)
- Requires: FLOWSEEK_DAV2_CHECKPOINT env var (Depth-AnythingV2 weights)
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: Runs in-process on GPU when both checkpoints are set; otherwise use --use-mock.
