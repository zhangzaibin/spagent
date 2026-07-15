#!/usr/bin/env python3
"""
Tool Usage Habit Visualization
==============================
Scans a spagent trace directory (JSON per sample) and visualizes how the model
uses the `pi3x_tool`: how often it calls tools at all, how many calls per
sample, and the distribution of tool arguments (azimuth / elevation angles,
reference camera, camera-view / image-path flags).

Usage
-----
    python scripts/visualize_tool_usage.py \
        [--traces-dir outputs/spagent_traces_full_qwen8brl90step00pen1] \
        [--output-dir outputs] \
        [--prefix tool_usage]

Run it with an environment that has matplotlib, e.g.:
    /home/zzb/anaconda3/envs/spagent/bin/python scripts/visualize_tool_usage.py
"""

import argparse
import glob
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Default paths ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

DEFAULT_TRACES_DIR = os.path.join(
    _ROOT, "outputs", "spagent_traces_full_qwen8brl90step00pen1"
)
DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "outputs")

# ── Colour palette ────────────────────────────────────────────────────────────
C_PRIMARY = "#4C72B0"
C_ACCENT = "#DD8452"
C_GREEN = "#55A868"
C_GREY = "#B0B0B0"


# ── Data collection ───────────────────────────────────────────────────────────
def collect_stats(traces_dir):
    """Walk every *.json trace and aggregate tool-usage statistics."""
    files = sorted(glob.glob(os.path.join(traces_dir, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"No JSON traces found under: {traces_dir}")

    stats = {
        "n_samples": 0,
        "n_used_tool": 0,          # samples with >=1 tool_call
        "tool_names": Counter(),   # tool name -> #calls
        "calls_per_sample": Counter(),
        "iterations": Counter(),
        "azimuth": Counter(),
        "elevation": Counter(),
        "refcam": Counter(),
        "has_image_path": Counter(),
        "has_camera_view": Counter(),
        "correct_with_tool": 0,
        "total_with_tool": 0,
        "correct_no_tool": 0,
        "total_no_tool": 0,
        "elapsed": [],
    }

    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        stats["n_samples"] += 1
        tcs = d.get("tool_calls") or []
        stats["calls_per_sample"][len(tcs)] += 1
        if tcs:
            stats["n_used_tool"] += 1
        it = d.get("iterations")
        if it is not None:
            stats["iterations"][it] += 1
        el = d.get("elapsed_s")
        if isinstance(el, (int, float)):
            stats["elapsed"].append(el)

        # accuracy split (uses <answer>X</answer> style ground_truth match)
        gt = (d.get("ground_truth") or "").strip().upper()
        ans = d.get("answer") or ""
        pred = _extract_choice(ans)
        used = bool(tcs)
        if gt:
            correct = int(pred == gt)
            if used:
                stats["total_with_tool"] += 1
                stats["correct_with_tool"] += correct
            else:
                stats["total_no_tool"] += 1
                stats["correct_no_tool"] += correct

        for tc in tcs:
            stats["tool_names"][tc.get("name", "unknown")] += 1
            a = tc.get("arguments") or {}
            stats["azimuth"][a.get("azimuth_angle")] += 1
            stats["elevation"][a.get("elevation_angle")] += 1
            stats["refcam"][a.get("rotation_reference_camera")] += 1
            stats["has_image_path"][a.get("has_image_path")] += 1
            stats["has_camera_view"][a.get("has_camera_view")] += 1

    return stats, files


def _extract_choice(text):
    """Best-effort extraction of a single-letter answer from the response."""
    import re

    m = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"answer\s+is[:\s]*([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    return ""


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _sorted_num_items(counter):
    """Sort a counter by numeric key, pushing None to the end."""
    return sorted(counter.items(), key=lambda kv: (kv[0] is None, kv[0]))


def _bar(ax, labels, values, color, title, xlabel, ylabel="# tool calls", rotate=0):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="center" if rotate == 0 else "right")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if values:
        ax.set_ylim(0, max(values) * 1.18)  # headroom so labels don't hit the title
    total = sum(values)
    for b, v in zip(bars, values):
        if v > 0:
            pct = f"\n{100*v/total:.0f}%" if total else ""
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v}{pct}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    return bars


def make_dashboard(stats, out_path, title):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.28)

    n = stats["n_samples"]
    used = stats["n_used_tool"]
    no_tool = n - used

    # 1. Tool usage rate (pie)
    ax = fig.add_subplot(gs[0, 0])
    ax.pie(
        [used, no_tool],
        labels=[f"Used tool\n{used}", f"No tool\n{no_tool}"],
        autopct="%1.1f%%",
        colors=[C_PRIMARY, C_GREY],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    ax.set_title(f"Tool-Use Rate (n={n})", fontsize=12, fontweight="bold")

    # 2. Calls per sample
    ax = fig.add_subplot(gs[0, 1])
    items = _sorted_num_items(stats["calls_per_sample"])
    _bar(
        ax,
        [str(k) for k, _ in items],
        [v for _, v in items],
        C_ACCENT,
        "# Tool Calls per Sample",
        "tool calls in one sample",
        ylabel="# samples",
    )

    # 3. Iterations per sample
    ax = fig.add_subplot(gs[0, 2])
    items = _sorted_num_items(stats["iterations"])
    _bar(
        ax,
        [str(k) for k, _ in items],
        [v for _, v in items],
        C_GREEN,
        "# Agent Iterations per Sample",
        "iterations",
        ylabel="# samples",
    )

    # 4. Azimuth angle distribution
    ax = fig.add_subplot(gs[1, 0])
    items = _sorted_num_items(stats["azimuth"])
    _bar(
        ax,
        ["None" if k is None else str(k) for k, _ in items],
        [v for _, v in items],
        C_PRIMARY,
        "Azimuth Angle",
        "azimuth (deg)",
        rotate=45,
    )

    # 5. Elevation angle distribution
    ax = fig.add_subplot(gs[1, 1])
    items = _sorted_num_items(stats["elevation"])
    _bar(
        ax,
        ["None" if k is None else str(k) for k, _ in items],
        [v for _, v in items],
        C_ACCENT,
        "Elevation Angle",
        "elevation (deg)",
        rotate=45,
    )

    # 6. Reference camera distribution
    ax = fig.add_subplot(gs[1, 2])
    items = _sorted_num_items(stats["refcam"])
    _bar(
        ax,
        ["None" if k is None else f"cam{k}" for k, _ in items],
        [v for _, v in items],
        C_GREEN,
        "Rotation Reference Camera",
        "reference camera",
    )

    # 7. Boolean flags (has_image_path / has_camera_view)
    ax = fig.add_subplot(gs[2, 0])
    flag_labels = ["has_image_path", "has_camera_view"]
    true_vals = [
        stats["has_image_path"].get(True, 0),
        stats["has_camera_view"].get(True, 0),
    ]
    false_vals = [
        stats["has_image_path"].get(False, 0),
        stats["has_camera_view"].get(False, 0),
    ]
    x = np.arange(len(flag_labels))
    w = 0.38
    ax.bar(x - w / 2, true_vals, w, label="True", color=C_PRIMARY)
    ax.bar(x + w / 2, false_vals, w, label="False", color=C_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(flag_labels)
    ax.set_title("Boolean Arguments", fontsize=12, fontweight="bold")
    ax.set_ylabel("# tool calls")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # 8. Elapsed time histogram
    ax = fig.add_subplot(gs[2, 1])
    if stats["elapsed"]:
        ax.hist(stats["elapsed"], bins=30, color=C_ACCENT, edgecolor="white")
        mean_el = float(np.mean(stats["elapsed"]))
        ax.axvline(mean_el, color=C_PRIMARY, linestyle="--", linewidth=1.5,
                   label=f"mean={mean_el:.1f}s")
        ax.legend()
    ax.set_title("Elapsed Time per Sample", fontsize=12, fontweight="bold")
    ax.set_xlabel("seconds")
    ax.set_ylabel("# samples")
    ax.grid(axis="y", alpha=0.3)

    # 9. Accuracy: tool vs no-tool
    ax = fig.add_subplot(gs[2, 2])
    acc_labels, acc_vals, acc_txt = [], [], []
    if stats["total_with_tool"]:
        acc_labels.append("with tool")
        a = stats["correct_with_tool"] / stats["total_with_tool"]
        acc_vals.append(100 * a)
        acc_txt.append(f"{stats['correct_with_tool']}/{stats['total_with_tool']}")
    if stats["total_no_tool"]:
        acc_labels.append("no tool")
        a = stats["correct_no_tool"] / stats["total_no_tool"]
        acc_vals.append(100 * a)
        acc_txt.append(f"{stats['correct_no_tool']}/{stats['total_no_tool']}")
    if acc_labels:
        bars = ax.bar(acc_labels, acc_vals, color=[C_PRIMARY, C_GREY][: len(acc_labels)])
        for b, v, t in zip(bars, acc_vals, acc_txt):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}%\n({t})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Accuracy: Tool vs No-Tool", fontsize=12, fontweight="bold")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Console summary ───────────────────────────────────────────────────────────
def print_summary(stats):
    n = stats["n_samples"]
    used = stats["n_used_tool"]
    total_calls = sum(stats["tool_names"].values())
    print("=" * 60)
    print(f"Samples scanned        : {n}")
    print(f"Samples using a tool   : {used}  ({100*used/n:.1f}%)")
    print(f"Total tool calls       : {total_calls}")
    if used:
        print(f"Avg calls / tool-user  : {total_calls/used:.2f}")
    print(f"Tool names             : {dict(stats['tool_names'])}")
    print(f"Azimuth dist           : {dict(_sorted_num_items(stats['azimuth']))}")
    print(f"Elevation dist         : {dict(_sorted_num_items(stats['elevation']))}")
    print(f"Ref camera dist        : {dict(_sorted_num_items(stats['refcam']))}")
    if stats["total_with_tool"]:
        print(f"Acc with tool          : "
              f"{100*stats['correct_with_tool']/stats['total_with_tool']:.1f}% "
              f"({stats['correct_with_tool']}/{stats['total_with_tool']})")
    if stats["total_no_tool"]:
        print(f"Acc no tool            : "
              f"{100*stats['correct_no_tool']/stats['total_no_tool']:.1f}% "
              f"({stats['correct_no_tool']}/{stats['total_no_tool']})")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-dir", default=DEFAULT_TRACES_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--prefix", default="tool_usage")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stats, files = collect_stats(args.traces_dir)
    print_summary(stats)

    run_name = os.path.basename(os.path.normpath(args.traces_dir))
    out_png = os.path.join(args.output_dir, f"{args.prefix}_dashboard.png")
    make_dashboard(stats, out_png, title=f"Tool Usage Habits — {run_name}")
    print(f"Saved dashboard -> {out_png}")

    # Also dump the raw aggregated stats as JSON for downstream use.
    out_json = os.path.join(args.output_dir, f"{args.prefix}_stats.json")
    serializable = {
        k: (dict(v) if isinstance(v, Counter) else v)
        for k, v in stats.items()
        if k != "elapsed"
    }
    serializable["elapsed_mean_s"] = (
        float(np.mean(stats["elapsed"])) if stats["elapsed"] else None
    )
    # Counter keys may be None / non-str -> stringify for JSON.
    for key in ["calls_per_sample", "iterations", "azimuth", "elevation",
                "refcam", "has_image_path", "has_camera_view"]:
        serializable[key] = {str(k): v for k, v in serializable[key].items()}
    json.dump(serializable, open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"Saved stats     -> {out_json}")


if __name__ == "__main__":
    main()
