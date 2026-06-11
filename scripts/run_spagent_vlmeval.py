"""
run_spagent_vlmeval.py
======================
Evaluate SPAgent (with or without tools) on VLMEvalKit benchmarks.

Usage
-----
# Smoke-test: 10 samples on VStarBench, no tools
python scripts/run_spagent_vlmeval.py \\
    --model gpt-4.1-mini \\
    --config no_tools \\
    --datasets VStarBench \\
    --limit 10 \\
    --trace-dir outputs/spagent_traces \\
    --work-dir  outputs/vlmeval_runs

# Full 5-benchmark empirical study
python scripts/run_spagent_vlmeval.py \\
    --model gpt-4.1-mini \\
    --config perception \\
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \\
    --trace-dir outputs/spagent_traces \\
    --work-dir  outputs/vlmeval_runs

Tool config can also be specified as an individual server URL override, e.g.:
    --detection-url http://localhost:20022
    --segmentation-url http://localhost:20020
    --depth-url http://localhost:20019

All 15 mentor benchmarks (add any subset to --datasets):
  MMStar  VStarBench  BLINK  MMMU_DEV_VAL  MathVista_MINI
  MMBench_dev_en  RealWorldQA  ScienceQA_VAL
  HRBench4K  HRBench8K
  MathVerse_MINI  WeMath  LogicVista  MMMU_Pro_10c  DynaMath
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── VLMEvalKit ────────────────────────────────────────────────────────────────
os.environ.setdefault("LMUData", str(Path.home() / "LMUData"))

# Suppress LiteLLM warning about network connectivity
warnings.filterwarnings("ignore", message=".*LiteLLM.*")

# ── SPAgent imports ───────────────────────────────────────────────────────────
from spagent.core import SPAgent, GENERAL_VISION_CONTINUATION_HINT
from spagent.models import GPTModel
from spagent.eval.spagent_vlmeval_model import SPAgentVLMEvalModel

# Tool imports (all optional; missing tool = server not running)
_tool_import_errors: List[str] = []
try:
    from spagent.tools import ObjectDetectionTool, SegmentationTool, DepthEstimationTool
except ImportError as e:
    _tool_import_errors.append(str(e))
try:
    from spagent.tools import Pi3XTool, VGGTTool
except ImportError:
    Pi3XTool = VGGTTool = None  # type: ignore[assignment]
try:
    from spagent.tools.catalog import build_all_tools, build_tools
except ImportError as e:
    build_all_tools = None  # type: ignore[assignment,misc]
    build_tools = None  # type: ignore[assignment,misc]
    _tool_import_errors.append(str(e))

# ── VLMEvalKit dataset helpers ────────────────────────────────────────────────
from vlmeval.dataset import build_dataset, DATASET_TYPE
from vlmeval.dataset.image_mcq import ImageMCQDataset, HRBenchDataset, MMMUDataset, WeMath
from vlmeval.dataset.image_vqa import ImageVQADataset, MathVista, MathVerse, LogicVista


# ── Default benchmark limits ──────────────────────────────────────────────────
DEFAULT_LIMIT: Dict[str, Optional[int]] = {
    "MMStar":        200,
    "VStarBench":    None,   # only 191 total
    "BLINK":         200,
    "MMMU_DEV_VAL":  150,
    "MathVista_MINI": 200,
    # extras from mentor list
    "MMBench_dev_en":  200,
    "RealWorldQA":     200,
    "ScienceQA_VAL":   200,
    "HRBench4K":       200,
    "HRBench8K":       200,
    "MathVerse_MINI":  200,
    "WeMath":          200,
    "LogicVista":      200,
    "MMMU_Pro_10c":    150,
    "DynaMath":        200,
}


# ── Tool configuration factory ────────────────────────────────────────────────

def make_tools(config: str, args) -> List[Any]:
    """Return a list of tool instances for the given config name."""
    if config == "no_tools":
        return []

    det_url = getattr(args, "detection_url", "http://127.0.0.1:20022")
    seg_url = getattr(args, "segmentation_url", "http://127.0.0.1:20020")
    dep_url = getattr(args, "depth_url", "http://127.0.0.1:20019")
    pi3x_url = getattr(args, "pi3x_url", "http://127.0.0.1:20031")

    if config == "perception":
        tools = []
        try:
            tools.append(ObjectDetectionTool(use_mock=False, server_url=det_url))
            tools.append(SegmentationTool(use_mock=False, server_url=seg_url))
            tools.append(DepthEstimationTool(use_mock=False, server_url=dep_url))
        except Exception as exc:
            print(f"[WARN] Could not instantiate perception tools: {exc}")
        return tools

    if config == "spatial":
        tools = make_tools("perception", args)
        try:
            if Pi3XTool is not None:
                tools.append(Pi3XTool(use_mock=False, server_url=pi3x_url))
        except Exception as exc:
            print(f"[WARN] Could not instantiate Pi3XTool: {exc}")
        return tools

    if config == "all_tools":
        if build_tools is None:
            raise RuntimeError("build_tools unavailable; check spagent.tools.catalog import errors")
        overrides = {
            "detection": {"server_url": det_url},
            "segmentation": {"server_url": seg_url},
            "depth": {"server_url": dep_url},
            "pi3x": {"server_url": pi3x_url},
        }
        use_mock = getattr(args, "use_mock", False)
        tool_keys = getattr(args, "tools", None)
        tools, skipped = build_tools(
            tool_keys=tool_keys,
            use_mock=use_mock,
            overrides=overrides,
            skip_unavailable=True,
        )
        if skipped:
            print(f"[WARN] Skipped unavailable all-tools entries: {skipped}")
        return tools

    raise ValueError(
        f"Unknown tool config: {config!r}. Choose from: no_tools, perception, spatial, all_tools"
    )


# ── Dataset loader with stratified sub-sampling ───────────────────────────────

def load_dataset_subset(dataset_name: str, limit: Optional[int]):
    """
    Load a VLMEvalKit dataset and optionally sub-sample it.

    For BLINK (14 sub-tasks), stratified sampling is used so every sub-task
    is represented (using the 'category' column).
    """
    ds = build_dataset(dataset_name)

    if limit is None or len(ds.data) <= limit:
        return ds

    df = ds.data
    if dataset_name == "BLINK" and "category" in df.columns:
        # Stratified: take ceil(limit / n_cats) from each category
        cats = df["category"].unique()
        per_cat = max(1, limit // len(cats))
        sampled = (
            df.groupby("category", group_keys=False)
            .apply(lambda g: g.head(per_cat))
            .reset_index(drop=True)
            .head(limit)
        )
        ds.data = sampled
    else:
        ds.data = df.head(limit)

    return ds


# ── Run inference using VLMEvalKit's infer_data_job ──────────────────────────

def run_inference(
    vlm_model: SPAgentVLMEvalModel,
    dataset_name: str,
    dataset_obj,
    work_dir: Path,
    model_tag: str,
    nproc: int = 1,
) -> Path:
    """
    Drive VLMEvalKit's inference loop and return the path to the prediction xlsx.
    """
    from vlmeval.inference import infer_data_job

    ds_work_dir = work_dir / model_tag / dataset_name
    ds_work_dir.mkdir(parents=True, exist_ok=True)

    # infer_data_job writes {model_name}_{dataset}.xlsx into work_dir
    infer_data_job(
        model=vlm_model,
        work_dir=str(ds_work_dir),
        model_name=model_tag,
        dataset=dataset_obj,
        nproc=nproc,
        verbose=False,
    )

    pred_file = ds_work_dir / f"{model_tag}_{dataset_name}.xlsx"
    return pred_file


# ── Run evaluation / scoring ──────────────────────────────────────────────────

def run_evaluate(
    dataset_obj,
    pred_file: Path,
    judge_model: str,
    nproc: int = 4,
) -> Dict[str, Any]:
    """Call VLMEvalKit's dataset.evaluate() and return the score dict."""
    if not pred_file.exists():
        print(f"  [WARN] pred file not found: {pred_file}")
        return {}

    judge_kwargs = {"model": judge_model, "nproc": nproc, "verbose": False}
    try:
        result = dataset_obj.evaluate(str(pred_file), **judge_kwargs)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        print(f"  [WARN] evaluate() failed: {exc}")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SPAgent on VLMEvalKit benchmarks for empirical study."
    )
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="OpenAI model name for SPAgent backbone (default: gpt-4.1-mini)")
    parser.add_argument("--config", default="no_tools",
                        choices=["no_tools", "perception", "spatial", "all_tools"],
                        help="Tool configuration to run")
    parser.add_argument("--datasets", nargs="+",
                        default=["MMStar", "VStarBench", "BLINK", "MMMU_DEV_VAL", "MathVista_MINI"],
                        help="Benchmark names to evaluate (space-separated)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override max samples per benchmark (None = use defaults)")
    parser.add_argument("--trace-dir", default="outputs/spagent_traces",
                        help="Root directory for per-sample trace JSON files")
    parser.add_argument("--work-dir", default="outputs/vlmeval_runs",
                        help="Root directory for VLMEvalKit output xlsx files")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="GPT model used for VLMEvalKit judge scoring")
    parser.add_argument("--nproc", type=int, default=1,
                        help="Number of parallel inference workers (default: 1)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max tool-call iterations per sample (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    # Tool server URL overrides
    parser.add_argument("--detection-url",  default="http://127.0.0.1:20022")
    parser.add_argument("--segmentation-url", default="http://127.0.0.1:20020")
    parser.add_argument("--depth-url",       default="http://127.0.0.1:20019")
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock tool clients (recommended for all_tools smoke tests)")
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help=(
            "For --config all_tools: catalog keys or tool names to load "
            "(e.g. depth segmentation pi3x). Default: all catalog tools."
        ),
    )

    parser.add_argument("--pi3x-url",        default="http://127.0.0.1:20031")

    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    work_dir  = Path(args.work_dir)

    # model_tag identifies this run in file names and trace sub-dirs
    model_safe = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
    model_tag  = f"{model_safe}_{args.config}"

    print(f"\n{'='*70}")
    print(f"  SPAgent VLMEvalKit Empirical Study")
    print(f"  Model:    {args.model}")
    print(f"  Config:   {args.config}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Limit:    {args.limit or 'defaults'}")
    print(f"  Tag:      {model_tag}")
    print(f"{'='*70}\n")

    # Build SPAgent (shared across all datasets for this config)
    gpt_model = GPTModel(
        model_name=args.model,
        temperature=args.temperature,
        seed=args.seed,
    )
    tools = make_tools(args.config, args)
    print(f"Tools loaded: {[t.name for t in tools] if tools else ['(none)']}")

    if args.config == "all_tools":
        agent = SPAgent(
            model=gpt_model,
            tools=tools,
            max_workers=4,
            workflow_mode="all_tools",
        )
    else:
        system_prompt = (
            "You are a helpful multimodal assistant. Analyze the provided image(s) "
            "and answer the question accurately. Use available tools when they can "
            "help you see finer details, detect objects, or measure spatial relationships."
        )
        agent = SPAgent(
            model=gpt_model,
            tools=tools,
            max_workers=4,
            system_prompt=system_prompt,
            continuation_hint=GENERAL_VISION_CONTINUATION_HINT,
        )

    summary: Dict[str, Any] = {}

    for dataset_name in args.datasets:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {dataset_name}")
        limit = args.limit if args.limit is not None else DEFAULT_LIMIT.get(dataset_name)
        print(f"  Samples: {limit or 'all'}")

        # 1. Load dataset
        try:
            dataset_obj = load_dataset_subset(dataset_name, limit)
        except Exception as exc:
            print(f"  [ERROR] Failed to load {dataset_name}: {exc}")
            summary[dataset_name] = {"error": str(exc)}
            continue
        print(f"  Loaded {len(dataset_obj.data)} rows")

        # 2. Build VLM wrapper (fresh counter per dataset)
        vlm_model = SPAgentVLMEvalModel(
            agent=agent,
            trace_dir=str(trace_dir),
            dataset_tag=model_tag,
            max_iterations=args.max_iterations,
        )

        # 3. Run inference
        print(f"  Running inference ...")
        try:
            pred_file = run_inference(
                vlm_model, dataset_name, dataset_obj,
                work_dir, model_tag, nproc=args.nproc,
            )
            print(f"  Predictions saved to: {pred_file}")
        except Exception as exc:
            print(f"  [ERROR] Inference failed: {exc}")
            summary[dataset_name] = {"error": str(exc)}
            continue

        # 4. Evaluate / score
        print(f"  Evaluating (judge: {args.judge_model}) ...")
        scores = run_evaluate(dataset_obj, pred_file, args.judge_model, nproc=args.nproc)
        summary[dataset_name] = {
            "pred_file": str(pred_file),
            "scores": scores,
            "n_samples": len(dataset_obj.data),
        }
        if scores:
            print(f"  Scores: {scores}")
        else:
            print(f"  (No scores returned; check pred file manually)")

    # Save run summary
    summary_path = work_dir / f"{model_tag}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved to: {summary_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
