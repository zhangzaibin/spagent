"""
analyze_failures.py
===================
Join VLMEvalKit prediction xlsx files with SPAgent trace JSONs,
classify every sample into one of 5 failure buckets, and produce:

  outputs/failure_analysis/<model_tag>_buckets.csv
  outputs/failure_analysis/<model_tag>_summary.md

Failure Buckets
---------------
correct         - answer is right (regardless of tool use)
no_tool_used    - answer is wrong AND the agent made zero tool calls
tool_failed     - a tool was called but at least one returned success=False
wrong_tool      - tools ran fine but the called tools are irrelevant for this dataset type
tool_didnt_help - tools ran fine and were plausibly relevant but answer is still wrong

Usage
-----
# Analyze one model/config run
python scripts/analyze_failures.py \\
    --model-tag  gpt_4_1_mini_perception \\
    --work-dir   outputs/vlmeval_runs \\
    --trace-dir  outputs/spagent_traces \\
    --datasets   MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \\
    --output-dir outputs/failure_analysis

# Compare two configs
python scripts/analyze_failures.py \\
    --model-tag  gpt_4_1_mini_no_tools gpt_4_1_mini_perception \\
    --work-dir   outputs/vlmeval_runs \\
    --trace-dir  outputs/spagent_traces \\
    --output-dir outputs/failure_analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ── Tool relevance heuristic ──────────────────────────────────────────────────

# Tools that are almost never useful for pure-knowledge benchmarks
_IRRELEVANT_TOOL_MAP: Dict[str, List[str]] = {
    # Dataset -> tools that are considered "wrong" to call there
    "MMMU_DEV_VAL":  ["pi3_tool", "pi3x_tool", "vggt_tool", "segmentation_tool"],
    "MMMU_Pro_10c":  ["pi3_tool", "pi3x_tool", "vggt_tool"],
    "ScienceQA_VAL": ["pi3_tool", "pi3x_tool", "vggt_tool"],
    "MathVista_MINI": ["pi3_tool", "pi3x_tool", "vggt_tool", "segmentation_tool"],
    "MathVerse_MINI": ["pi3_tool", "pi3x_tool"],
}


def _tool_mismatch(used_tools: List[str], tool_call_names: List[str], dataset_name: str) -> bool:
    """
    Return True if ALL tool calls in this sample are from the 'irrelevant' list
    for this dataset type (heuristic for 'wrong_tool' bucket).
    """
    irrelevant = _IRRELEVANT_TOOL_MAP.get(dataset_name, [])
    if not irrelevant or not tool_call_names:
        return False
    return all(t in irrelevant for t in tool_call_names)


# ── Per-sample classifier ─────────────────────────────────────────────────────

def classify_sample(row: pd.Series, trace: Optional[Dict[str, Any]], dataset_name: str) -> str:
    """
    Assign one of the 5 bucket labels to a sample.

    Parameters
    ----------
    row:         A row from the VLMEvalKit xlsx (must have 'hit' column: 1=correct, 0=wrong).
    trace:       The matching trace JSON dict (may be None if trace file is missing).
    dataset_name: Benchmark name string.
    """
    hit = int(row.get("hit", 0)) if pd.notna(row.get("hit")) else 0

    if hit == 1:
        return "correct"

    if trace is None:
        return "no_trace"   # trace file missing; can't classify further

    used_tools: List[str] = trace.get("used_tools", [])
    tool_calls: List[Dict] = trace.get("tool_calls", [])
    tool_results: Dict[str, Any] = trace.get("tool_results", {})
    tool_call_names: List[str] = [tc.get("name", "") for tc in tool_calls]

    if not tool_call_names:
        return "no_tool_used"

    # Check if any tool reported failure
    any_failed = any(
        not v.get("success", True)
        for v in tool_results.values()
        if isinstance(v, dict)
    )
    if any_failed:
        return "tool_failed"

    if _tool_mismatch(used_tools, tool_call_names, dataset_name):
        return "wrong_tool"

    return "tool_didnt_help"


# ── Trace loader ──────────────────────────────────────────────────────────────

def load_traces(trace_dir: Path, model_tag: str, dataset_name: str) -> Dict[int, Dict]:
    """
    Load all trace JSONs for a (model_tag, dataset_name) combination.
    Returns a dict keyed by sample index (the 'index' field in the trace).
    """
    trace_folder = trace_dir / model_tag / dataset_name
    traces: Dict[int, Dict] = {}
    if not trace_folder.exists():
        return traces
    for p in sorted(trace_folder.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                t = json.load(f)
            idx = t.get("index", int(p.stem))
            traces[idx] = t
        except Exception:
            pass
    return traces


# ── xlsx loader ───────────────────────────────────────────────────────────────

def load_pred_xlsx(work_dir: Path, model_tag: str, dataset_name: str) -> Optional[pd.DataFrame]:
    """Load VLMEvalKit prediction xlsx. Returns None if not found."""
    candidates = [
        work_dir / model_tag / dataset_name / f"{model_tag}_{dataset_name}.xlsx",
        work_dir / model_tag / f"{model_tag}_{dataset_name}.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_excel(p)
    return None


# ── Analyse one (model_tag, dataset) pair ─────────────────────────────────────

def analyse_dataset(
    model_tag: str,
    dataset_name: str,
    work_dir: Path,
    trace_dir: Path,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per sample, columns:
      dataset, index, category, gt, pred, hit, bucket,
      used_tools, tool_calls_summary, question, image_paths
    """
    df = load_pred_xlsx(work_dir, model_tag, dataset_name)
    if df is None:
        print(f"  [SKIP] No xlsx found for {model_tag} / {dataset_name}")
        return pd.DataFrame()

    traces = load_traces(trace_dir, model_tag, dataset_name)
    print(f"  {dataset_name}: {len(df)} xlsx rows, {len(traces)} trace files")

    rows = []
    for position, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict()
        # VLMEvalKit uses 0-based position as sample index within the run
        trace = traces.get(position)

        bucket = classify_sample(pd.Series(row_dict), trace, dataset_name)

        # Summarise tool calls for the CSV (compact string)
        tool_calls_summary = ""
        if trace:
            tcs = trace.get("tool_calls", [])
            tool_calls_summary = "; ".join(
                f"{tc.get('name','?')}({list(tc.get('arguments',{}).keys())})"
                for tc in tcs
            ) if tcs else ""

        rows.append({
            "model_tag":         model_tag,
            "dataset":           dataset_name,
            "index":             position,
            "category":          row_dict.get("category", ""),
            "gt":                row_dict.get("answer", ""),
            "pred":              row_dict.get("prediction", ""),
            "hit":               int(row_dict.get("hit", 0)) if pd.notna(row_dict.get("hit")) else 0,
            "bucket":            bucket,
            "used_tools":        json.dumps(trace.get("used_tools", []) if trace else []),
            "tool_calls_summary": tool_calls_summary,
            "question":          row_dict.get("question", trace.get("question", "") if trace else ""),
            "image_paths":       json.dumps(trace.get("image_paths", []) if trace else []),
        })

    return pd.DataFrame(rows)


# ── Summary markdown ──────────────────────────────────────────────────────────

_BUCKET_ORDER = ["correct", "no_tool_used", "tool_failed", "wrong_tool",
                 "tool_didnt_help", "no_trace"]

_BUCKET_DESC = {
    "correct":        "Answer correct",
    "no_tool_used":   "Wrong & no tool called",
    "tool_failed":    "Tool error (success=False)",
    "wrong_tool":     "Wrong tool called (heuristic)",
    "tool_didnt_help": "Tool ran OK but didn't fix answer",
    "no_trace":       "Trace file missing",
}


def _format_example(row: pd.Series, max_q_chars: int = 200) -> str:
    q = str(row.get("question", ""))[:max_q_chars]
    gt = row.get("gt", "")
    pred = row.get("pred", "")
    tools = row.get("tool_calls_summary", "")
    imgs = row.get("image_paths", "[]")
    return (
        f"- **Q**: {q}\n"
        f"  **GT**: `{gt}` | **Pred**: `{pred}`\n"
        f"  **Tools**: {tools or '(none)'}\n"
        f"  **Images**: {imgs}\n"
    )


def build_summary_md(all_df: pd.DataFrame, model_tags: List[str]) -> str:
    lines = ["# SPAgent Failure Analysis\n"]
    lines.append(f"Model tags: {', '.join(model_tags)}\n")

    for model_tag in model_tags:
        mdf = all_df[all_df["model_tag"] == model_tag]
        if mdf.empty:
            continue
        lines.append(f"\n## Config: `{model_tag}`\n")

        for dataset_name in sorted(mdf["dataset"].unique()):
            ddf = mdf[mdf["dataset"] == dataset_name]
            n_total = len(ddf)
            n_correct = (ddf["hit"] == 1).sum()
            acc = n_correct / n_total if n_total else 0.0

            lines.append(f"\n### {dataset_name}  ({n_correct}/{n_total} = {acc:.1%})\n")
            lines.append("| Bucket | Count | % |\n|---|---|---|\n")

            bucket_counts = ddf["bucket"].value_counts()
            for bkt in _BUCKET_ORDER:
                cnt = bucket_counts.get(bkt, 0)
                pct = cnt / n_total * 100 if n_total else 0
                lines.append(f"| {_BUCKET_DESC[bkt]} | {cnt} | {pct:.1f}% |\n")

            # 3 example rows per non-correct bucket
            for bkt in _BUCKET_ORDER[1:]:
                examples = ddf[ddf["bucket"] == bkt].head(3)
                if examples.empty:
                    continue
                lines.append(f"\n**{_BUCKET_DESC[bkt]}** – examples:\n\n")
                for _, ex in examples.iterrows():
                    lines.append(_format_example(ex))

    return "".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Failure attribution for SPAgent evaluation runs.")
    parser.add_argument("--model-tag", nargs="+", required=True,
                        help="One or more model tags (e.g. gpt_4_1_mini_perception)")
    parser.add_argument("--work-dir",  default="outputs/vlmeval_runs",
                        help="Root dir of VLMEvalKit xlsx outputs")
    parser.add_argument("--trace-dir", default="outputs/spagent_traces",
                        help="Root dir of per-sample trace JSONs")
    parser.add_argument("--datasets",  nargs="+",
                        default=["MMStar", "VStarBench", "BLINK", "MMMU_DEV_VAL", "MathVista_MINI"],
                        help="Benchmark names to analyse")
    parser.add_argument("--output-dir", default="outputs/failure_analysis",
                        help="Where to write CSV and summary.md")
    args = parser.parse_args()

    work_dir  = Path(args.work_dir)
    trace_dir = Path(args.trace_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []

    for model_tag in args.model_tag:
        print(f"\n{'='*60}")
        print(f"Analysing: {model_tag}")
        for dataset_name in args.datasets:
            df = analyse_dataset(model_tag, dataset_name, work_dir, trace_dir)
            if not df.empty:
                all_frames.append(df)

    if not all_frames:
        print("No data found. Check --work-dir and --trace-dir paths.")
        return

    all_df = pd.concat(all_frames, ignore_index=True)

    # ── Per-model-tag CSV ──────────────────────────────────────────────────────
    for model_tag in args.model_tag:
        mdf = all_df[all_df["model_tag"] == model_tag]
        if mdf.empty:
            continue
        csv_path = out_dir / f"{model_tag}_buckets.csv"
        mdf.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\nWrote: {csv_path}")

    # ── Combined summary markdown ──────────────────────────────────────────────
    md = build_summary_md(all_df, args.model_tag)
    md_tags = "_vs_".join(args.model_tag)
    md_path = out_dir / f"{md_tags}_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {md_path}")

    # ── Quick console summary ──────────────────────────────────────────────────
    print("\n=== Quick Summary ===")
    for model_tag in args.model_tag:
        mdf = all_df[all_df["model_tag"] == model_tag]
        if mdf.empty:
            continue
        print(f"\n  Config: {model_tag}")
        pivot = (
            mdf.groupby(["dataset", "bucket"])
            .size()
            .unstack(fill_value=0)
        )
        # Reorder columns
        cols = [c for c in _BUCKET_ORDER if c in pivot.columns]
        print(pivot[cols].to_string())

    print()


if __name__ == "__main__":
    main()
