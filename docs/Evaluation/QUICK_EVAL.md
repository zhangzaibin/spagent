# Quick Eval Script

> **中文版本**: [QUICK_EVAL_ZH.md](QUICK_EVAL_ZH.md) | **English Version**: This document

`scripts/quick_eval.py` is the fast evaluation entry point for SPAgent. It bypasses vlmeval's `infer_data_job` pipeline entirely — instead iterating dataset rows → `SPAgent.solve_problem` → writing xlsx → calling `dataset.evaluate()` for scoring.

---

## Quick Start

```bash
# Minimal run: 5 samples, no tools
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --datasets VStarBench \
    --limit 5

# No-tools baseline across 5 benchmarks
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
    --limit 50

# GroundingDINO combo (zoom for attributes, localize for spatial/counting)
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom localize \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
    --limit 50 \
    --detection-url http://localhost:20022

# Perception combo: zoom + segmentation + depth
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom segmentation depth \
    --datasets MMStar \
    --limit 50 \
    --detection-url    http://localhost:20022 \
    --segmentation-url http://localhost:20020 \
    --depth-url        http://localhost:20019

# Spatial understanding combo (perception + 3D reconstruction)
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom localize segmentation depth pi3x \
    --datasets VStarBench BLINK \
    --limit 50
```

---

## Arguments

### Core

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gpt-4.1-mini` | LLM model name for inference (OpenAI / LiteLLM compatible) |
| `--tools` | _(none)_ | Tools to enable, space-separated, any combination (see tool list below) |
| `--datasets` | `VStarBench` | Dataset names to evaluate, space-separated, multiple supported |
| `--limit` | _(per-dataset defaults)_ | Max samples per dataset; omit to use built-in defaults |

### Inference Control

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-iterations` | `3` | Max agent iterations per question (including tool calls) |
| `--temperature` | `0.0` | LLM sampling temperature |
| `--seed` | `42` | Random seed for reproducibility |

### Scoring & Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--judge-model` | `gpt-4o-mini` | Model used as judge for scoring |
| `--nproc` | `4` | Number of parallel processes for scoring |
| `--no-score` | _(off)_ | Skip scoring; run inference only |
| `--work-dir` | `outputs/vlmeval_runs` | Directory for xlsx predictions and score summaries |
| `--trace-dir` | `outputs/spagent_traces` | Directory for per-sample inference traces (JSON) |

---

## Tool List

`--tools` accepts any combination of the following names:

| Tool Name | Class | Description | URL Argument | Default Address |
|-----------|-------|-------------|--------------|-----------------|
| `zoom` | ZoomObjectTool | GroundingDINO crop close-up for attribute inspection (color/texture/text) | `--detection-url` | `http://localhost:20022` |
| `localize` | LocalizeObjectTool | GroundingDINO bbox annotation for spatial/counting | `--detection-url` | `http://localhost:20022` |
| `detection` | ZoomObjectTool | Backward-compatible alias of `zoom` | `--detection-url` | `http://localhost:20022` |
| `segmentation` | SegmentationTool | Instance segmentation (SAM2) | `--segmentation-url` | `http://localhost:20020` |
| `depth` | DepthEstimationTool | Monocular depth estimation | `--depth-url` | `http://localhost:20019` |
| `pi3x` | Pi3XTool | 3D scene reconstruction (Pi3X) | `--pi3x-url` | `http://localhost:20031` |
| `pi3` | Pi3Tool | 3D scene reconstruction (Pi3) | `--pi3-url` | `http://localhost:20030` |
| `vggt` | VGGTTool | Visual geometry reconstruction (VGGT) | `--vggt-url` | `http://localhost:20032` |
| `mapanything` | MapAnythingTool | Dense 3D reconstruction (MapAnything) | `--mapanything-url` | `http://localhost:20033` |
| `yoloe` | YOLOETool | YOLO-E object detection | `--yoloe-url` | `http://0.0.0.0:8000` |
| `supervision` | SupervisionTool | Visual supervision annotations | `--supervision-url` | `http://0.0.0.0:8000` |
| `moondream` | MoondreamTool | Lightweight vision-language model | `--moondream-url` | `http://localhost:20024` |
| `molmo2` | Molmo2Tool | Molmo2 point grounding | `--molmo2-url` | `http://localhost:20025` |
| `orient` | OrientAnythingV2Tool | Orientation estimation (Orient Anything V2) | `--orient-url` | `http://localhost:20034` |
| `vace` | VaceTool | Local video generation (Wan2.1-VACE) | `--vace-url` | `http://localhost:20034` |
| `sana` | SanaTool | Image generation (Sana) | `--sana-url` | `http://127.0.0.1:30000` |
| `qwenvl` | QwenVLTool | Qwen-VL vision-language model | _(API Key)_ | — |
| `veo` | VeoTool | Video generation (Veo) | _(API Key)_ | — |
| `sora` | SoraTool | Video generation (Sora) | _(API Key)_ | — |
| `wan` | WanTool | Video generation (Wan) | _(API Key)_ | — |

> **Note**: `zoom`, `localize`, and `detection` all share the single GroundingDINO server via `--detection-url`. `orient` and `vace` both default to port `20034` and are not meant to run at the same time. Tools with a `server_url` require the corresponding service to be running before evaluation (see [TOOL_USING.md](../Tool/TOOL_USING.md)). Tools using API Keys (`qwenvl`, `veo`, `sora`, `wan`) require the relevant credentials set in environment variables.

---

## Supported Datasets

Built-in per-dataset sample limits (overridable with `--limit`):

| Dataset | Default Limit |
|---------|---------------|
| MMStar | 200 |
| VStarBench | _(full)_ |
| BLINK | 200 (balanced across categories) |
| MMMU_DEV_VAL | 150 |
| MathVista_MINI | 200 |
| MMBench_dev_en | 200 |
| RealWorldQA | 200 |
| ScienceQA_VAL | 200 |
| HRBench4K | 200 |
| HRBench8K | 200 |
| MathVerse_MINI | 200 |
| WeMath | 200 |
| LogicVista | 200 |
| MMMU_Pro_10c | 150 |
| DynaMath | 200 |

---

## Output Directory Structure

Results are grouped by a `{model}_{tools}` tag, so different tool combinations never overwrite each other:

```
outputs/
├── vlmeval_runs/
│   ├── gpt_4_1_mini_no_tools/
│   │   ├── MMStar/
│   │   │   └── gpt_4_1_mini_no_tools_MMStar.xlsx     # predictions
│   │   └── VStarBench/
│   │       └── gpt_4_1_mini_no_tools_VStarBench.xlsx
│   ├── gpt_4_1_mini_zoom_localize_general/            # zoom+localize combo
│   │   └── ...
│   └── gpt_4_1_mini_no_tools_quick_summary.json       # per-dataset score summary
└── spagent_traces/
    └── gpt_4_1_mini_zoom_localize_general/
        └── MMStar/
            ├── 00000.json   # full inference trace per sample
            ├── 00001.json
            └── ...
```

### Trace File Format

Each `.json` trace file contains:

```json
{
  "index": 0,
  "dataset": "MMStar",
  "question": "...",
  "image_paths": ["..."],
  "answer": "A",
  "used_tools": ["zoom_object_tool_iter1"],
  "tool_calls": [...],
  "tool_results": {...},
  "iterations": 2,
  "elapsed_s": 3.14,
  "error": null
}
```

---

## Resume / Checkpoint

The script automatically resumes interrupted runs: if an xlsx file already exists, it loads existing predictions and skips completed samples.

```bash
# If a 50-sample run was interrupted, re-run the same command to continue from where it stopped
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom localize \
    --datasets MMStar \
    --limit 50
```

---

## Common Scenarios

### Ablation Study (adding tools incrementally)

```bash
# Step 1: No-tools baseline
python scripts/quick_eval.py --model gpt-4.1-mini --datasets MMStar --limit 100

# Step 2: Zoom only
python scripts/quick_eval.py --model gpt-4.1-mini --tools zoom --datasets MMStar --limit 100

# Step 3: Zoom + localize
python scripts/quick_eval.py --model gpt-4.1-mini --tools zoom localize --datasets MMStar --limit 100

# Step 4: Full perception suite
python scripts/quick_eval.py --model gpt-4.1-mini --tools zoom localize segmentation depth --datasets MMStar --limit 100
```

### Inference Only (skip scoring)

```bash
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom depth \
    --datasets MMStar VStarBench \
    --no-score
```

### All 15 Benchmarks

```bash
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom localize segmentation depth \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
              MMBench_dev_en RealWorldQA ScienceQA_VAL \
              HRBench4K HRBench8K \
              MathVerse_MINI WeMath LogicVista MMMU_Pro_10c DynaMath \
    --detection-url    http://localhost:20022 \
    --segmentation-url http://localhost:20020 \
    --depth-url        http://localhost:20019
```

---

## Shell Script Shortcuts

`scripts/` ships ready-to-run wrappers around `quick_eval.py`. They share defaults
through `scripts/_eval_common.sh` (localhost tool URLs, `MODEL`, `MAX_ITER`, dirs).
Every knob is an environment variable, so you override on the command line:

```bash
MODEL=gpt-4.1 DATASETS="MindCube MMStar" LIMIT=100 bash scripts/eval_all_tools.sh
```

| Script | Tools | Default datasets |
|--------|-------|------------------|
| `eval_no_tools.sh` | _(none)_ — baseline | all 17 benchmarks |
| `eval_all_tools.sh` | `pi3x zoom localize segmentation depth molmo2 vace` | all 17 benchmarks |
| `eval_detection_only.sh` | `zoom localize` | MMStar |
| `eval_detection_pi3x.sh` | `zoom localize pi3x` | MMStar |
| `eval_molmo2_only.sh` | `molmo2` | MMStar |
| `eval_pi3x_only.sh` | `pi3x` (spatial prompt) | MindCube + VSIBench |

**Common environment variables** (see `scripts/_eval_common.sh` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `gpt-4.1-mini` | Model name |
| `MODEL_BACKEND` | `auto` | `auto` / `gpt` / `qwen` / `qwen-vllm` |
| `DATASETS` | _(per-script)_ | Space-separated dataset list |
| `LIMIT` | `50` (empty = no limit) | Samples per dataset |
| `MAX_ITER` | `3` (`1` for `eval_no_tools.sh`) | Max tool-call iterations |
| `TOOLS` | _(per-script)_ | Override the tool set |
| `DETECTION_URL` | `http://localhost:20022` | GroundingDINO server |
| `PI3X_URL` | `http://localhost:20031` | Pi3X server |
| `SEGMENTATION_URL` | `http://localhost:20020` | SAM2 server |
| `DEPTH_URL` | `http://localhost:20019` | Depth Anything server |
| `MOLMO2_URL` | `http://localhost:20025` | Molmo2 server |
| `VACE_URL` | `http://localhost:20034` | VACE server |

> Defaults point at `localhost`. For a remote deployment, override the URL, e.g.
> `DETECTION_URL=http://10.7.8.94:20022 bash scripts/eval_detection_only.sh`.

---

## Troubleshooting the judge / re-scoring

`quick_eval.py` scores predictions with a VLMEvalKit judge model (default
`gpt-4o-mini`). If the judge API is unreachable (SSL / proxy errors in the log),
VLMEvalKit silently falls back to **exact string matching**, which under-counts
correct answers (e.g. a real ~0.52 can show up as 0.32).

The judge result is **cached** — simply re-running will *not* re-call the judge.
To force a clean re-score, delete the cache, fix the judge endpoint, then re-run
(inference is skipped because predictions load from the existing `.xlsx`):

```bash
RUN=outputs/vlmeval_runs/gpt_4_1_zoom_localize_general/MMStar
rm -f "$RUN"/*_gpt-4o-mini_result.pkl \
      "$RUN"/*_gpt-4o-mini_result.xlsx \
      "$RUN"/*_acc.csv

# Clear any broken proxy and point the judge at a reachable endpoint
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export OPENAI_API_BASE="https://your-endpoint/v1/chat/completions"
export OPENAI_API_KEY="your_key"

DATASETS=MMStar LIMIT=50 bash scripts/eval_detection_only.sh
```

A correct judge run prints `Scoring (judge=gpt-4o-mini) ...` **without** the
`will use exact matching for evaluation` warning.
