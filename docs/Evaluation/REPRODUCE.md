# Reproducing SPAgent Evaluation Results

> **中文版本**: [REPRODUCE_ZH.md](REPRODUCE_ZH.md) | **English Version**: This document

This is the single end-to-end recipe for reproducing SPAgent benchmark numbers.
It ties together environment setup, tool-server deployment, dataset preparation,
and the evaluation scripts. For deeper detail on any step, follow the links.

| Step | What | Reference |
|------|------|-----------|
| 1 | Install the environment | [readme.md → Installation](../../readme.md#-installation--setup) |
| 2 | Configure API keys | this doc, §2 |
| 3 | Start the tool servers you need | [TOOL_USING.md](../Tool/TOOL_USING.md) |
| 4 | Prepare datasets | [EVALUATION.md](EVALUATION.md) |
| 5 | Run an eval script | [QUICK_EVAL.md](QUICK_EVAL.md) |
| 6 | Read the scores | this doc, §6 |

---

## 1. Environment

```bash
conda create -n spagent python=3.11 -y
conda activate spagent
pip install -r requirements.txt
```

## 2. API keys

`quick_eval.py` auto-loads a `.env` file from the project root. Create one with the
keys for the model(s) and judge you plan to use:

```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # or your gateway

# Optional, depending on backend / tools:
DASHSCOPE_API_KEY=...      # Qwen models (--model-backend qwen)
GOOGLE_API_KEY=...         # Veo tool
```

The judge model (`--judge-model`, default `gpt-4o-mini`) uses the same
`OPENAI_*` variables.

## 3. Start tool servers

Only start the servers for the tools you intend to evaluate. Ports below match
[TOOL_USING.md](../Tool/TOOL_USING.md); the eval scripts default to `localhost`
on these ports.

| Tool | Port | Start command |
|------|------|---------------|
| Depth Anything V2 | 20019 | `python spagent/external_experts/Depth_AnythingV2/depth_server.py --checkpoint_path checkpoints/depth_anything/depth_anything_v2_vitb.pth --port 20019` |
| SAM2 | 20020 | `python spagent/external_experts/SAM2/sam2_server.py --checkpoint_path checkpoints/sam2/sam2.1_b.pt --port 20020` |
| GroundingDINO (zoom/localize) | 20022 | `python spagent/external_experts/GroundingDINO/grounding_dino_server.py --checkpoint_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth --port 20022` |
| Molmo2 | 20025 | `python spagent/external_experts/Molmo2/molmo2_server.py --checkpoint allenai/Molmo2-4B --port 20025` |
| Pi3X | 20031 | `python spagent/external_experts/Pi3/pi3x_server.py --checkpoint_path checkpoints/pi3x/model.safetensors --port 20031` |
| VACE | 20034 | `python spagent/external_experts/vace/vace_server.py --checkpoint_path checkpoints/Wan2.1-VACE-1.3B --port 20034` |

Health check (most servers expose `/health`):

```bash
curl -s http://localhost:20022/health
```

> If a server runs on a remote machine, override the URL when launching the eval
> script, e.g. `DETECTION_URL=http://10.7.8.94:20022 bash scripts/eval_detection_only.sh`.

## 4. Prepare datasets

- **VLMEvalKit benchmarks** (MMStar, VStarBench, BLINK, …): downloaded automatically
  on first use, no manual step needed.
- **Local datasets** (MindCube, VSIBench): prepare once —

```bash
python spagent/utils/download_mindcube.py     # → dataset/MindCube_data.jsonl
python spagent/utils/download_vsibench.py      # → dataset/VSI_Bench.jsonl
```

See [EVALUATION.md](EVALUATION.md) for every dataset.

## 5. Run an evaluation

Smoke test (no servers required):

```bash
bash scripts/eval_no_tools.sh        # baseline, all benchmarks, 50 samples each
# or a single quick check:
python scripts/quick_eval.py --model gpt-4.1-mini --datasets MMStar --limit 5
```

Full tool stack (requires the servers from §3):

```bash
MODEL=gpt-4.1 DATASETS="MindCube MMStar VStarBench BLINK" LIMIT=200 \
  bash scripts/eval_all_tools.sh
```

Targeted runs:

```bash
bash scripts/eval_detection_only.sh   # GroundingDINO zoom + localize
bash scripts/eval_detection_pi3x.sh   # GroundingDINO + Pi3X
bash scripts/eval_pi3x_only.sh        # Pi3X on MindCube + VSIBench
bash scripts/eval_molmo2_only.sh      # Molmo2 point grounding
```

See [QUICK_EVAL.md](QUICK_EVAL.md) for the full list of scripts and env vars.

## 6. Read the results

Outputs are grouped under `outputs/vlmeval_runs/<model_tag>/`:

- `<tag>_quick_summary.json` — per-dataset score summary (start here)
- `<dataset>/<tag>_<dataset>.xlsx` — predictions (auto-resumed on re-run)
- `<dataset>/<tag>_<dataset>_results.json` — per-sample results + accuracy breakdown
- `<dataset>/<tag>_<dataset>_errors.jsonl` — wrong-answer records for analysis

Per-sample inference traces (prompts, tool calls, tool outputs) are under
`outputs/spagent_traces/<model_tag>/<dataset>/NNNNN.json`.

---

## Reference results

> **TODO**: fill in once a reference run is completed. Numbers below are placeholders.

Settings: `--limit 200`, `--max-iterations 3`, `--temperature 0.0`, `--seed 42`,
judge = `gpt-4o-mini`.

| Model | Tools | MindCube | VSIBench | MMStar | VStarBench | BLINK |
|-------|-------|----------|----------|--------|------------|-------|
| gpt-4.1-mini | none (baseline) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| gpt-4.1-mini | zoom + localize | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| gpt-4.1-mini | full stack | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| gpt-4.1 | full stack | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
