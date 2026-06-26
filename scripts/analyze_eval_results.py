#!/usr/bin/env python3
"""
Eval Comparison: No Tools vs With Tools
========================================
Reads two quick_summary.json files from vlmeval_runs, generates
comparison charts, and prints a summary table.

Usage
-----
    python scripts/analyze_eval_results.py [--output-dir outputs]

All output PNGs are written to --output-dir.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Default paths ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.join(_SCRIPT_DIR, "..")

DEFAULT_NO_TOOLS_JSON = os.path.join(
    _ROOT, "outputs/vlmeval_runs/gpt_4_1_mini_no_tools_quick_summary.json"
)
DEFAULT_TOOLS_JSON = os.path.join(
    _ROOT,
    "outputs/vlmeval_runs/"
    "gpt_4_1_mini_pi3x_detection_segmentation_depth_molmo2_vace_quick_summary.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "outputs")

# ── Colour palette ────────────────────────────────────────────────────────────
C_NOTOOL = "#5B8DB8"   # steel blue
C_TOOLS  = "#E07B54"   # warm orange
C_POS    = "#4CAF50"   # green  (improvement)
C_NEG    = "#F44336"   # red    (regression)
C_ZERO   = "#9E9E9E"   # grey   (no change)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    with open(path) as f:
        content = f.read()
    # json module cannot parse bare NaN → replace with null
    content = content.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(content)


def _safe_float(v):
    """Return float or None for null/NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _to_pct(v):
    """Normalise a score to percentage (0-100)."""
    if v is None:
        return None
    return v * 100 if v <= 1.0 else v


def extract_overall(scores: dict) -> float | None:
    """
    Extract a single Overall accuracy (%) from a scores dict.
    Handles the various formats used across datasets.
    """
    # WeMath: {'accuracy': 0.42, ...}
    if "accuracy" in scores and "Overall" not in scores:
        v = _safe_float(scores["accuracy"])
        return _to_pct(v)

    # MathVista / LogicVista: {'Task&Skill': [...], 'acc': [76.0, ...]}
    if "acc" in scores and "Task&Skill" in scores:
        skills = scores["Task&Skill"]
        acc    = scores["acc"]
        if "Overall" in skills:
            idx = skills.index("Overall")
            v = _safe_float(acc[idx])
            return _to_pct(v)

    # Most datasets: {'Overall': [0.84]} or {'Overall': 0.84}
    if "Overall" in scores:
        val = scores["Overall"]
        if isinstance(val, list):
            vals = [_safe_float(x) for x in val if _safe_float(x) is not None]
            if not vals:
                return None
            return _to_pct(sum(vals) / len(vals))
        v = _safe_float(val)
        return _to_pct(v)

    return None


def extract_mathverse_splits(scores: dict) -> dict[str, float]:
    """Return {split_name: pct} dict."""
    splits = scores.get("split", [])
    overalls = scores.get("Overall", [])
    return {s: _to_pct(_safe_float(o))
            for s, o in zip(splits, overalls)
            if _safe_float(o) is not None}


def extract_task_skill(scores: dict) -> dict[str, float]:
    """Return {task: acc_pct} for MathVista / LogicVista."""
    tasks  = scores.get("Task&Skill", [])
    accs   = scores.get("acc", [])
    result = {}
    for t, a in zip(tasks, accs):
        v = _safe_float(a)
        if v is not None:
            result[t] = _to_pct(v)
    return result


# ── Build summary table ───────────────────────────────────────────────────────

DATASET_LABELS = {
    "BLINK":         "BLINK",
    "LogicVista":    "LogicVista",
    "MathVerse_MINI":"MathVerse",
    "MathVista_MINI":"MathVista",
    "MindCube":      "MindCube",
    "MMMU_DEV_VAL":  "MMMU-DEV",
    "MMMU_Pro_10c":  "MMMU-Pro",
    "MMStar":        "MMStar",
    "RealWorldQA":   "RealWorldQA",
    "ScienceQA_VAL": "ScienceQA",
    "VStarBench":    "VStarBench",
    "WeMath":        "WeMath",
}


def build_overall_table(no_tools: dict, tools: dict) -> list[dict]:
    rows = []
    for key, label in DATASET_LABELS.items():
        nt_entry = no_tools.get(key, {})
        t_entry  = tools.get(key, {})
        if "error" in nt_entry or "error" in t_entry:
            continue
        nt_score = extract_overall(nt_entry.get("scores", {}))
        t_score  = extract_overall(t_entry.get("scores", {}))
        if nt_score is None or t_score is None:
            continue
        n = nt_entry.get("n_samples", t_entry.get("n_samples", "?"))
        rows.append({
            "key":      key,
            "label":    label,
            "n":        n,
            "no_tools": round(nt_score, 2),
            "tools":    round(t_score, 2),
            "delta":    round(t_score - nt_score, 2),
        })
    return rows


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _fig_ax(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 1 – Overall grouped bar ────────────────────────────────────────────

def plot_overall_comparison(rows: list[dict], out_dir: str):
    labels = [r["label"]    for r in rows]
    nt     = [r["no_tools"] for r in rows]
    to_    = [r["tools"]    for r in rows]

    x   = np.arange(len(labels))
    w   = 0.36
    fig, ax = _fig_ax(14, 6)

    b1 = ax.bar(x - w/2, nt,  w, color=C_NOTOOL, label="No Tools",   zorder=3)
    b2 = ax.bar(x + w/2, to_, w, color=C_TOOLS,  label="With Tools", zorder=3)

    # value labels
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy: No Tools vs With Tools\n(gpt-4.1-mini, 50 samples each)", fontsize=13)
    ax.legend(handles=[
        mpatches.Patch(color=C_NOTOOL, label="No Tools"),
        mpatches.Patch(color=C_TOOLS,  label="With Tools"),
    ])
    ax.set_ylim(0, max(max(nt), max(to_)) * 1.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    _save(fig, os.path.join(out_dir, "eval_comparison_overall.png"))


# ── Chart 2 – Delta bar ───────────────────────────────────────────────────────

def plot_delta(rows: list[dict], out_dir: str):
    rows_sorted = sorted(rows, key=lambda r: r["delta"], reverse=True)
    labels = [r["label"] for r in rows_sorted]
    deltas = [r["delta"] for r in rows_sorted]
    colors = [C_POS if d > 0 else (C_NEG if d < 0 else C_ZERO) for d in deltas]

    fig, ax = _fig_ax(10, 5)
    bars = ax.barh(labels, deltas, color=colors, zorder=3, height=0.55)
    ax.axvline(0, color="black", linewidth=0.8)

    for bar, d in zip(bars, deltas):
        x_pos = d + (0.1 if d >= 0 else -0.1)
        ha = "left" if d >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{d:+.1f}%", va="center", ha=ha, fontsize=9)

    ax.set_xlabel("Δ Accuracy (With Tools − No Tools, pp)")
    ax.set_title("Per-Dataset Improvement with Tools\n(positive = tools help)", fontsize=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    _save(fig, os.path.join(out_dir, "eval_comparison_delta.png"))


# ── Chart 3 – Radar chart ────────────────────────────────────────────────────

def plot_radar(rows: list[dict], out_dir: str):
    labels = [r["label"]    for r in rows]
    nt     = [r["no_tools"] for r in rows]
    to_    = [r["tools"]    for r in rows]
    N      = len(labels)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    # close the polygon
    nt_c  = nt  + [nt[0]]
    to_c  = to_ + [to_[0]]
    ang_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(ang_c, nt_c,  "o-", color=C_NOTOOL, linewidth=2, label="No Tools")
    ax.fill(ang_c, nt_c,  alpha=0.15, color=C_NOTOOL)
    ax.plot(ang_c, to_c,  "s-", color=C_TOOLS,  linewidth=2, label="With Tools")
    ax.fill(ang_c, to_c,  alpha=0.15, color=C_TOOLS)

    ax.set_thetagrids(np.degrees(angles), labels, fontsize=10)
    ax.set_title("Radar: No Tools vs With Tools\n(gpt-4.1-mini)", y=1.12, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))

    _save(fig, os.path.join(out_dir, "eval_comparison_radar.png"))


# ── Chart 4 – MathVista task breakdown ───────────────────────────────────────

def plot_mathvista(no_tools: dict, tools: dict, out_dir: str):
    nt_s = no_tools.get("MathVista_MINI", {}).get("scores", {})
    t_s  =    tools.get("MathVista_MINI", {}).get("scores", {})
    nt_d = extract_task_skill(nt_s)
    t_d  = extract_task_skill(t_s)

    keys = [k for k in nt_d if k != "Overall" and k in t_d]
    if not keys:
        return

    nt_v = [nt_d[k] for k in keys]
    t_v  = [ t_d[k] for k in keys]
    short = [k.replace(" ", "\n") for k in keys]

    x = np.arange(len(keys))
    w = 0.38
    fig, ax = _fig_ax(14, 5)
    ax.bar(x - w/2, nt_v, w, color=C_NOTOOL, label="No Tools",   zorder=3)
    ax.bar(x + w/2, t_v,  w, color=C_TOOLS,  label="With Tools", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MathVista_MINI – Task & Skill Breakdown", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    _save(fig, os.path.join(out_dir, "eval_comparison_mathvista.png"))


# ── Chart 5 – MathVerse split breakdown ──────────────────────────────────────

def plot_mathverse(no_tools: dict, tools: dict, out_dir: str):
    nt_s = no_tools.get("MathVerse_MINI", {}).get("scores", {})
    t_s  =    tools.get("MathVerse_MINI", {}).get("scores", {})
    nt_d = extract_mathverse_splits(nt_s)
    t_d  = extract_mathverse_splits(t_s)

    keys = sorted(set(nt_d) & set(t_d))
    if not keys:
        return

    nt_v = [nt_d[k] for k in keys]
    t_v  = [ t_d[k] for k in keys]

    x = np.arange(len(keys))
    w = 0.38
    fig, ax = _fig_ax(10, 5)
    ax.bar(x - w/2, nt_v, w, color=C_NOTOOL, label="No Tools",   zorder=3)
    ax.bar(x + w/2, t_v,  w, color=C_TOOLS,  label="With Tools", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MathVerse_MINI – Split Breakdown", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    _save(fig, os.path.join(out_dir, "eval_comparison_mathverse.png"))


# ── Chart 6 – LogicVista task breakdown ──────────────────────────────────────

def plot_logicvista(no_tools: dict, tools: dict, out_dir: str):
    nt_s = no_tools.get("LogicVista", {}).get("scores", {})
    t_s  =    tools.get("LogicVista", {}).get("scores", {})
    nt_d = extract_task_skill(nt_s)
    t_d  = extract_task_skill(t_s)

    keys = [k for k in nt_d if k != "Overall" and k in t_d]
    if not keys:
        return

    nt_v = [nt_d[k] for k in keys]
    t_v  = [ t_d[k] for k in keys]

    x = np.arange(len(keys))
    w = 0.38
    fig, ax = _fig_ax(8, 5)
    ax.bar(x - w/2, nt_v, w, color=C_NOTOOL, label="No Tools",   zorder=3)
    ax.bar(x + w/2, t_v,  w, color=C_TOOLS,  label="With Tools", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=11)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("LogicVista – Task Breakdown", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    _save(fig, os.path.join(out_dir, "eval_comparison_logicvista.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare eval results: no-tools vs tools")
    parser.add_argument("--no-tools-json", default=DEFAULT_NO_TOOLS_JSON)
    parser.add_argument("--tools-json",    default=DEFAULT_TOOLS_JSON)
    parser.add_argument("--output-dir",    default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.no_tools_json}")
    no_tools = _load(args.no_tools_json)
    print(f"Loading: {args.tools_json}")
    tools    = _load(args.tools_json)

    rows = build_overall_table(no_tools, tools)

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Dataset':<14} {'N':>4}  {'No Tools':>10}  {'With Tools':>10}  {'Delta':>8}")
    print("-" * 65)
    for r in rows:
        sign = "▲" if r["delta"] > 0 else ("▼" if r["delta"] < 0 else " ")
        print(f"{r['label']:<14} {r['n']:>4}  {r['no_tools']:>9.2f}%  "
              f"{r['tools']:>9.2f}%  {sign}{abs(r['delta']):>5.2f}pp")
    print("=" * 65)

    # avg
    avg_nt = sum(r["no_tools"] for r in rows) / len(rows)
    avg_t  = sum(r["tools"]    for r in rows) / len(rows)
    print(f"{'Average':<14} {'':>4}  {avg_nt:>9.2f}%  {avg_t:>9.2f}%  "
          f"{'▲' if avg_t>avg_nt else '▼'}{abs(avg_t-avg_nt):>5.2f}pp\n")

    # ── Generate charts ─────────────────────────────────────────────────────
    print("Generating charts …")
    plot_overall_comparison(rows, args.output_dir)
    plot_delta(rows, args.output_dir)
    plot_radar(rows, args.output_dir)
    plot_mathvista(no_tools, tools, args.output_dir)
    plot_mathverse(no_tools, tools, args.output_dir)
    plot_logicvista(no_tools, tools, args.output_dir)

    print("\nDone. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
