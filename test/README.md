# Test Directory

Unit tests and integration smoke tests for SPAgent.

---

## Files

| File | Purpose |
|---|---|
| `test_tool.py` | Directly invoke external expert tools (Pi3, Pi3X, SAM2, GroundingDINO, etc.) without going through the LLM agent |
| `test_pi3_llm.py` | End-to-end agent test: video → frame extraction → Pi3 tool → LLM answer |
| `test_prompt.py` | Verify system prompt construction for all configurations (3D spatial / general vision / custom) |
| `test_crop_tool.py` | CropTool tests for box, multi-box, mask, and polygon crops |

---

## Quick Start

```bash
# Run from project root
cd /path/to/spagent

# Test system prompts (no server needed)
python test/test_prompt.py

# Test a specific prompt case
python test/test_prompt.py --case general
python test/test_prompt.py --case 3d
python test/test_prompt.py --case constants

# Test Pi3 tool directly (requires Pi3 server running)
python test/test_tool.py --tool pi3 --image assets/dog.jpeg --azimuth 45 --elevation -30

# Test Pi3X tool
python test/test_tool.py --tool pi3x --image assets/dog.jpeg --azimuth 45 --elevation -30

# Test GroundingDINO tool directly
python test/test_tool.py --tool grounding_dino --image assets/dog.jpeg --text_prompt "dog"

# Test CropTool
python -m pytest -q test/test_crop_tool.py

# Test SAM2 tool directly
python test/test_tool.py --tool sam --image assets/dog.jpeg
```

---

## test_prompt.py

Tests that system prompts are built correctly across all usage patterns.
**No external server or API key required** — uses mock tools.

### Test Cases

| `--case` | What it checks |
|---|---|
| `3d` | `create_system_prompt()` with no args → contains azimuth/elevation instructions |
| `general` | `create_system_prompt(workflow=GENERAL_VISION_WORKFLOW)` → no 3D content |
| `spagent_none` | `SPAgent(system_prompt=None)` internal path → falls back to 3D prompt |
| `spagent_general` | `SPAgent(system_prompt=GENERAL_VISION_SYSTEM_PROMPT)` → `{tools_json}` placeholder replaced correctly |
| `custom` | Custom string without `{tools_json}` placeholder → tools block appended automatically |
| `constants` | Print `SPATIAL_3D_WORKFLOW` and `GENERAL_VISION_WORKFLOW` constants side-by-side |

---

## test_tool.py

Directly calls individual tools bypassing the LLM. Useful for checking connectivity
and output format before running full evaluations.

### Supported Tools

| `--tool` | Tool class | Default server |
|---|---|---|
| `pi3` | `Pi3Tool` | `http://localhost:20030` |
| `pi3x` | `Pi3XTool` | `http://localhost:20031` |
| `sam` | `SegmentationTool` | `http://localhost:20020` |
| `grounding_dino` | `ObjectDetectionTool` | `http://localhost:20022` |

### Key Options

```
--image       One or more image paths
--azimuth     Horizontal angle for Pi3/Pi3X (-180 ~ 180)
--elevation   Vertical angle for Pi3/Pi3X (-90 ~ 90)
--camera_view Enable first-person camera perspective (Pi3/Pi3X)
--server_url  Override the default server URL
--text_prompt Object description for GroundingDINO
```

---

## test_pi3_llm.py

Full end-to-end smoke test: extracts frames from a video, passes them to an
`SPAgent` configured with `Pi3Tool`, and prints the final answer.

Requires:
- Pi3 server running at `http://10.7.8.94:20030`
- A valid video file at `dataset/VLM4D_videos/synthetic_synth_216.mp4`
- `OPENAI_API_KEY` set in environment

---

## Recent Updates

### System Prompt Parameterisation (`spagent/core/prompts.py`, `spagent/core/spagent.py`)

The system prompt is now fully configurable at the `SPAgent` level.

**New exports from `spagent.core.prompts`:**

| Symbol | Type | Description |
|---|---|---|
| `SPATIAL_3D_WORKFLOW` | `str` | Original 3D workflow instructions (Pi3, azimuth, elevation, camera view) |
| `GENERAL_VISION_WORKFLOW` | `str` | Generic vision workflow — no 3D-specific content |
| `SPATIAL_3D_SYSTEM_PROMPT` | `str` | Full 3D system prompt template with `{tools_json}` placeholder |
| `GENERAL_VISION_SYSTEM_PROMPT` | `str` | Full general vision prompt template with `{tools_json}` placeholder |

`create_system_prompt(tools, workflow=None)` now accepts an optional `workflow`
argument. Passing `None` (default) keeps the original 3D behaviour.

**`SPAgent.__init__` new parameter:**

```python
SPAgent(
    model=...,
    tools=...,
    system_prompt=None,   # ← new
)
```

- `None` (default) → build prompt with `create_system_prompt()` (3D spatial, backward-compatible)
- A template string → use it directly; `{tools_json}` placeholder is replaced with the
  live tool schema JSON; if no placeholder, the tools block is appended automatically

**`evaluate_tool_config` new parameter** (`spagent_evaluation.py`):

```python
evaluate_tool_config(
    ...,
    system_prompt=None,   # ← new, passed through to SPAgent
)
```

**Usage examples:**

```python
from spagent.core.prompts import GENERAL_VISION_SYSTEM_PROMPT, SPATIAL_3D_SYSTEM_PROMPT

# General vision (DinoSAM, depth, etc.)
evaluate_tool_config(..., system_prompt=GENERAL_VISION_SYSTEM_PROMPT)

# 3D spatial (Pi3X) — same as before, no change needed
evaluate_tool_config(..., system_prompt=None)

# Fully custom prompt
MY_PROMPT = "You are a specialist.\n<tools>\n{tools_json}\n</tools>\n..."
SPAgent(model=model, tools=tools, system_prompt=MY_PROMPT)
```

### DinoSAM Evaluation Script (`examples/evaluation/evaluate_dinosam.py`)

- Switched tool config from Pi3X (port 20031) to SAM2 + GroundingDINO:
  - `SegmentationTool` → `http://localhost:20020`
  - `ObjectDetectionTool` → `http://localhost:20022`
- Now passes `GENERAL_VISION_SYSTEM_PROMPT` to `evaluate_tool_config`
- Removed Pi3-specific statistics printing and imports
