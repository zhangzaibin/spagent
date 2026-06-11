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

# Perception tool combo (detection + segmentation + depth)
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools detection segmentation depth \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
    --limit 50 \
    --detection-url    http://localhost:20022 \
    --segmentation-url http://localhost:20020 \
    --depth-url        http://localhost:20019

# Custom combo: detection + depth only
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools detection depth \
    --datasets MMStar \
    --limit 50

# Spatial understanding combo (perception + 3D reconstruction)
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools detection segmentation depth pi3x \
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
| `detection` | ObjectDetectionTool | Object detection (DINO/YOLO) | `--detection-url` | `http://localhost:20022` |
| `segmentation` | SegmentationTool | Instance segmentation (SAM2) | `--segmentation-url` | `http://localhost:20020` |
| `depth` | DepthEstimationTool | Monocular depth estimation | `--depth-url` | `http://localhost:20019` |
| `pi3x` | Pi3XTool | 3D scene reconstruction (Pi3X) | `--pi3x-url` | `http://localhost:20031` |
| `pi3` | Pi3Tool | 3D scene reconstruction (Pi3) | `--pi3-url` | `http://localhost:20030` |
| `vggt` | VGGTTool | Visual geometry reconstruction (VGGT) | `--vggt-url` | `http://localhost:20022` |
| `mapanything` | MapAnythingTool | Scene mapping (MapAnything) | `--mapanything-url` | `http://localhost:20022` |
| `yoloe` | YOLOETool | YOLO-E object detection | `--yoloe-url` | `http://0.0.0.0:8000` |
| `supervision` | SupervisionTool | Visual supervision annotations | `--supervision-url` | `http://0.0.0.0:8000` |
| `moondream` | MoondreamTool | Lightweight vision-language model | `--moondream-url` | `http://localhost:20024` |
| `molmo2` | Molmo2Tool | Molmo2 vision-language model | `--molmo2-url` | `http://localhost:20025` |
| `orient` | OrientAnythingV2Tool | Orientation estimation (Orient Anything V2) | `--orient-url` | `http://localhost:20034` |
| `vace` | VaceTool | Video/image content editing | `--vace-url` | `http://localhost:20034` |
| `sana` | SanaTool | Image generation (Sana) | `--sana-url` | `http://127.0.0.1:30000` |
| `qwenvl` | QwenVLTool | Qwen-VL vision-language model | _(API Key)_ | — |
| `veo` | VeoTool | Video generation (Veo) | _(API Key)_ | — |
| `sora` | SoraTool | Video generation (Sora) | _(API Key)_ | — |
| `wan` | WanTool | Video generation (Wan) | _(API Key)_ | — |

> **Note**: Tools with a `server_url` require the corresponding service to be running and listening on the specified port before evaluation. Tools using API Keys (`qwenvl`, `veo`, `sora`, `wan`) require the relevant credentials set in environment variables.

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
│   ├── gpt_4_1_mini_detection_depth/                  # detection+depth combo
│   │   └── ...
│   └── gpt_4_1_mini_no_tools_quick_summary.json       # per-dataset score summary
└── spagent_traces/
    └── gpt_4_1_mini_detection_depth/
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
  "used_tools": ["detection_iter1"],
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
    --tools detection \
    --datasets MMStar \
    --limit 50
```

---

## Common Scenarios

### Ablation Study (adding tools incrementally)

```bash
# Step 1: No-tools baseline
python scripts/quick_eval.py --model gpt-4.1-mini --datasets MMStar --limit 100

# Step 2: Detection only
python scripts/quick_eval.py --model gpt-4.1-mini --tools detection --datasets MMStar --limit 100

# Step 3: Detection + segmentation
python scripts/quick_eval.py --model gpt-4.1-mini --tools detection segmentation --datasets MMStar --limit 100

# Step 4: Full perception suite
python scripts/quick_eval.py --model gpt-4.1-mini --tools detection segmentation depth --datasets MMStar --limit 100
```

### Inference Only (skip scoring)

```bash
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools detection depth \
    --datasets MMStar VStarBench \
    --no-score
```

### All 15 Benchmarks

```bash
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools detection segmentation depth \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
              MMBench_dev_en RealWorldQA ScienceQA_VAL \
              HRBench4K HRBench8K \
              MathVerse_MINI WeMath LogicVista MMMU_Pro_10c DynaMath \
    --detection-url    http://localhost:20022 \
    --segmentation-url http://localhost:20020 \
    --depth-url        http://localhost:20019
```
