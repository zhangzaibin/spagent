# Reinforcement Learning (GRPO) Training

SPAgent supports **GRPO** (Group Relative Policy Optimization) reinforcement
learning of the underlying VLM policy with multi-turn tool calling, powered by
[ms-swift](https://github.com/modelscope/ms-swift).

During training the policy learns to interleave `<think>`, `<tool_call>` and
`<answer>` steps, calling spatial experts (e.g. Pi3X novel-view rendering) and
being rewarded on both **answer accuracy** and **output format**.

This document covers the full pipeline, including the **offline Pi3X cache**
that makes RL training self-contained — no GPU expert servers or network calls
are needed inside the training loop.

- [Overview](#overview)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [1. Prepare the dataset](#1-prepare-the-dataset)
- [2. Pre-compute the Pi3X point-cloud cache](#2-pre-compute-the-pi3x-point-cloud-cache)
- [3. Smoke-test the offline tool](#3-smoke-test-the-offline-tool)
- [4. Launch GRPO training](#4-launch-grpo-training)
- [Reward functions](#reward-functions)
- [Multi-turn tool-calling scheduler](#multi-turn-tool-calling-scheduler)
- [System prompts](#system-prompts)
- [Post-training](#post-training)
- [File reference](#file-reference)

---

## Overview

| Concept | Description |
|---------|-------------|
| **Algorithm** | GRPO via `swift rlhf --rlhf_type grpo` |
| **Policy model** | A VLM (default: `Qwen3-VL-8B-Instruct`) |
| **Interaction** | Multi-turn: the model can call tools for up to `--max_turns` turns before answering |
| **Reward** | `external_r1v_acc` (answer correctness) + `external_multiturn_format` (format) + `external_angle_penalty` (penalizes wasted (0°, 0°) tool calls), equally weighted |
| **Expert tool** | `Pi3XOfflineTool` — renders novel viewpoints from a **pre-computed** point-cloud cache |

The key design choice for RL is that the Pi3X expert runs **offline**: instead of
querying a live Flask server per rollout, point clouds are computed **once** ahead
of time and cached to disk. Rollouts then render new viewpoints from the cache
with pure NumPy/matplotlib, which is fast, deterministic, and avoids GPU/server
contention during training.

## Pipeline at a glance

```
raw JSONL (placeholder paths)
        │  scripts/build_rl_dataset.py
        ▼
crossviewQA_train_rl_fixed.jsonl  (real local image paths)
        │  train/precompute_pi3x_cache.py   (Pi3X model, run once)
        ▼
dataset/pi3x_cache/<scene_id>.npz  (points + colors + camera poses)
        │  train/train_grpo.sh   (swift rlhf + Pi3XOfflineTool)
        ▼
output/grpo_*/checkpoint-*        (trained policy)
```

## 1. Prepare the dataset

The raw training JSONL ships with placeholder image paths such as
`/<replace_with_your_own_data_path>/mindcube/data/...`. Rewrite them to your
local dataset location with `scripts/build_rl_dataset.py`:

```bash
python scripts/build_rl_dataset.py \
    --input  "dataset/crossviewQA_train_rl (1).jsonl" \
    --output  dataset/crossviewQA_train_rl_fixed.jsonl \
    --mindcube-root dataset/mindcube \
    --verify        # warn about any missing images (use --skip-missing to drop them)
```

The output `crossviewQA_train_rl_fixed.jsonl` is what training consumes.

## 2. Pre-compute the Pi3X point-cloud cache

Run Pi3X inference **once** for every unique scene in the dataset and store the
filtered point cloud + camera poses as compressed `.npz` files. Training later
loads these without any model or network calls.

First download the Pi3X checkpoint (skip system proxies if needed):

```bash
mkdir -p checkpoints/pi3x
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    wget https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors \
    -O checkpoints/pi3x/model.safetensors
```

Then build the cache:

```bash
python train/precompute_pi3x_cache.py \
    --dataset dataset/crossviewQA_train_rl_fixed.jsonl \
    --cache-dir dataset/pi3x_cache \
    --checkpoint checkpoints/pi3x/model.safetensors \
    --gpu 0
```

Notes:
- The script is **idempotent** — scenes that are already cached are skipped, so
  it is safe to re-run after a crash.
- Use `--dry-run` to preview which scenes would be processed without running
  inference.
- Pass `--dataset` multiple times to cache several JSONL files in one go.

Each cache file (`<cache_dir>/<scene_id>.npz`) contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `points` | `(N, 3)` float32 | filtered 3-D world points |
| `colors` | `(N, 3)` float32 | RGB in `[0, 1]` |
| `camera_poses` | `(M, 4, 4)` float32 | camera-to-world matrices |
| `image_paths` | object array | original image paths for the scene |

## 3. Smoke-test the offline tool

Before launching a full run, verify the offline rendering pipeline end-to-end.
This creates a synthetic cache and exercises `Pi3XOfflineTool.call()` — **no Pi3X
model or network access required**:

```bash
python train/smoke_test_offline.py
```

It checks novel azimuth/elevation rendering, PNG output, and graceful failure on
a missing cache. All tests should print `PASSED`.

## 4. Launch GRPO training

Point `PI3X_CACHE_DIR` at your cache directory and run the training script:

```bash
cd train
bash train_grpo.sh
```

`train_grpo.sh` wires everything together — the important flags are:

```bash
MAX_PIXELS=262144 NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model <path-to-Qwen3-VL-8B-Instruct> \
    --external_plugins spagent/plugin/plugin.py \
    --multi_turn_scheduler spagent_tool_call_scheduler \
    --max_turns 3 \
    --reward_funcs external_r1v_acc external_multiturn_format external_angle_penalty \
    --reward_weights 1.0 1.0 1.0 \
    --tuner_type full \
    --dataset spagent/dataset/crossviewQA_train_rl_fixed.jsonl \
    --system spagent/train/system_prompt/system_prompt_grpo.txt \
    --max_completion_length 2048 \
    --num_generations 8 \
    --temperature 0.6 \
    --learning_rate 1e-6 \
    --deepspeed zero2 \
    --output_dir spagent/output/grpo_1111
```

The `Pi3XOfflineTool` is auto-registered by the scheduler and reads its cache
from `PI3X_CACHE_DIR` (exported at the top of `train_grpo.sh`, default
`spagent/dataset/pi3x_cache`).

Training logs (including sampled completions when `--log_completions true`) are
written to `--output_dir`; TensorBoard event files live under
`<output_dir>/*/runs/`.

## Reward functions

Rewards are implemented in [`plugin/plugin.py`](../plugin/plugin.py) and
registered into swift's `orms` registry. The default recipe uses three, equally
weighted:

| `--reward_funcs` name | Class | What it rewards |
|-----------------------|-------|-----------------|
| `external_r1v_acc` | accuracy reward | correctness of the final `<answer>` against the ground truth |
| `external_multiturn_format` | multi-turn format reward | valid `<think>` / `<tool_call>` / `<answer>` structure across turns |
| `external_angle_penalty` | `ZeroAngleToolCallPenalty` | penalizes `pi3_tool`/`pi3x_tool` calls that request a "wasted" `(azimuth=0, elevation=0)` view (or omit the angles, which default to 0) — such calls just re-render the original viewpoint and add no new geometric information. `-0.5` per wasted call, floored at `-1.0`, across the whole trajectory |

Additional reward functions are available in the same file (e.g.
`external_multiturn_format` progressive variant, tool-use format/length/correctness
rewards, code-execution rewards). To use a different combination, change
`--reward_funcs` and `--reward_weights`.

## Multi-turn tool-calling scheduler

The `spagent_tool_call_scheduler` (`SPAgentToolCallingScheduler` in
`plugin/plugin.py`) drives the multi-turn rollout:

- Parses the model's `<tool_call>` blocks, executes the requested tool, and feeds
  the result (text + rendered image) back as the next turn's observation.
- Auto-registers `Pi3XOfflineTool` (plus other available mock tools) so the
  policy's tool vocabulary matches inference.
- Stops after `--max_turns` turns or when the model emits a final `<answer>`.

## System prompts

Training modes use different prompts in `train/system_prompt/`:

| File | Mode |
|------|------|
| `system_prompt_grpo.txt` | Standard training with tool calling |
| `system_prompt_grpo_all_angles.txt` | Training with all angle combinations |
| `system_prompt_grpo_wotool.txt` | Baseline without tools |

## Post-training

```bash
# Merge LoRA adapters into the base model (if training with a LoRA tuner)
swift export \
    --adapters output/grpo_xxx/checkpoint-xxx \
    --merge_lora true
```

The merged/full checkpoint can then be served or evaluated like any other
SPAgent model (see [Quick Eval](Evaluation/QUICK_EVAL.md)).

> **Important**: when evaluating a GRPO-trained checkpoint, pass `--rl-trained`
> to `quick_eval.py` (or the eval wrapper scripts). This exposes the `pi3x_tool`
> schema exactly as it appeared in `train/system_prompt/system_prompt_grpo.txt`
> (angles **required**, no "default 0"). Without it, the default eval schema makes
> the angles optional and advertises 0 as the default, causing the model to emit
> the meaningless `(azimuth=0, elevation=0)` view it never used in training. See
> [Evaluating RL-Trained Models](Evaluation/QUICK_EVAL.md#evaluating-rl-trained-models).

## File reference

| Path | Role |
|------|------|
| `scripts/build_rl_dataset.py` | Rewrite placeholder image paths in the raw RL JSONL |
| `train/precompute_pi3x_cache.py` | Offline Pi3X inference → `.npz` point-cloud cache |
| `spagent/tools/pi3x_offline_tool.py` | `Pi3XOfflineTool`: renders cached point clouds during rollouts |
| `spagent/external_experts/Pi3/pi3x_render.py` | NumPy/matplotlib novel-view renderer used by the offline tool |
| `train/smoke_test_offline.py` | Offline-pipeline smoke test (no model/network) |
| `train/train_grpo.sh` | GRPO launch script |
| `plugin/plugin.py` | Reward functions + `spagent_tool_call_scheduler` |
| `train/system_prompt/` | Training system prompts |

## Related documentation

- [Advanced Examples](Examples/ADVANCED_EXAMPLES.md) — `step()` API, AgentMemory, generation tools
- [Tool Reference](Tool/TOOL_USING.md) — expert tool APIs and deployment
- [Evaluation Guide](Evaluation/EVALUATION.md)
