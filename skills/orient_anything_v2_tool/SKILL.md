---
name: orient_anything_v2_tool
description: Estimates the 3D orientation of objects in images using Orient Anything V2 Given a single image, returns azimuth (0-360°), elevation (-90~90°), in-plane rotation (-180~180°), and symmetry_alpha (0/1/2/4 indicating rotational symmetry order).
category: orientation
group: 3d
runtime: server
catalog_key: orient_anything_v2
---

# orient_anything_v2_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

Estimates the 3D orientation of objects in images using Orient Anything V2 Given a single image, returns azimuth (0-360°), elevation (-90~90°), in-plane rotation (-180~180°), and symmetry_alpha (0/1/2/4 indicating rotational symmetry order). Given two images of the same object from different viewpoints, also returns rel_azimuth, rel_elevation, rel_rotation — the relative pose of the second image with respect to the first. Useful for robotic grasping, AR/VR scene understanding, and spatial reasoning.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Absolute path to the input image. |
| `object_category` | string | yes | — | The semantic category of the object to estimate orientation for (e.g. 'chair', 'car', 'bottle', 'laptop'). Providing an accurate category significantly improves estimation quality. |
| `task` | string (orientation \\| symmetry \\| relative_rotation) | no | orientation | 'orientation': return azimuth/elevation/rotation of the object and its symmetry_alpha. 'symmetry': same as orientation — symmetry_alpha is always returned. 'relative_rotation': estimate relative pose between two views (requires image_path2); also returns the absolute pose of the first. |
| `image_path2` | string | no | — | Path to the second image. Required only when task='relative_rotation'. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `orientation`

Raw payload — any ONE of these carrier groups must be present:
- `azimuth` + `elevation` + `rotation`
- `rotation_matrix`
- `quaternion`

Common optional fields: `symmetry_order`, `confidence`

Default render projection: `azimuth`, `elevation`, `rotation`

## Invocation

```bash
python -m spagent.skills.run orient_anything_v2_tool --args '{"image_path": "assets/dog.jpeg", "object_category": "dog"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.
Use `--server-url URL` to override the default backend (`http://127.0.0.1:20034`), and `--output-dir DIR` to redirect artifacts when the tool supports it.

## Runtime requirements

- Runtime class: **server**
- Server: `http://127.0.0.1:20034` (health check: `GET /health`)
- Launch: `python spagent/external_experts/OrientAnythingV2/oa_v2_server.py --checkpoint_path checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt --repo_path third_party/orient_anything_v2 --port 20034`
- Checkpoint: `checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt`
- Requires: third_party/orient_anything_v2 (upstream repo checkout)
- Mock available: yes (`--use-mock`, no GPU/server needed)
