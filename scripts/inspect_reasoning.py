#!/usr/bin/env python3
"""
Inspect SPAgent reasoning for pi3x tool calls
=============================================
Re-runs a handful of MindCube samples with the SAME config as
``scripts/eval/eval_pi3x_only.sh --prompt spatial`` (spatial_3d prompt,
pi3x_tool in *inference* mode, temperature 0.6, max 3 iterations) and dumps,
per iteration:

  * the EXACT prompt string sent to the model (system + user, or the
    continuation context on later iterations)
  * the RAW model response (with <think> ... </think> and <tool_call> ...)
  * the parsed tool-call arguments (azimuth / elevation / refcam / camera_view)

The point is to see *why* the model emits (azimuth=0, elevation=0) at
inference time even though the prompt forbids it.

Endpoint / auth come from the environment (OPENAI_BASE_URL / OPENAI_API_KEY),
so run this in the shell where those are already exported (e.g. the `eval`
tmux session).

Usage
-----
    python scripts/inspect_reasoning.py --num 3
    python scripts/inspect_reasoning.py --ids among_group375_q3_1_1 ...
    python scripts/inspect_reasoning.py --trace-files path/to/00132.json ...
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    for _c in [_ROOT / ".env", Path(".env")]:
        if _c.exists():
            load_dotenv(_c, override=False)
except ImportError:
    pass

from spagent.core import SPAgent
from spagent.models import GPTModel
from spagent.tools import Pi3XTool

DEFAULT_MODEL = "/18141169908/models/MrBean2024/spagent-grpo-v20-ckpt90"
DEFAULT_TRACE_DIR = str(
    _ROOT / "outputs" / "spagent_traces_full_qwen8brl90step00pen1"
)
DEFAULT_OUT_DIR = str(_ROOT / "outputs" / "reasoning_inspect")


def _iter_trace_files(trace_dir):
    return sorted(glob.glob(os.path.join(trace_dir, "**", "*.json"), recursive=True))


def _has_zero_zero(trace):
    for tc in (trace.get("tool_calls") or []):
        a = tc.get("arguments") or {}
        if a.get("azimuth_angle") == 0 and a.get("elevation_angle") == 0:
            return True
    return False


def select_samples(args):
    """Return a list of (id, question, image_paths, orig_trace) to re-run."""
    files = _iter_trace_files(args.trace_dir)
    by_id, by_path = {}, {}
    for f in files:
        try:
            t = json.load(open(f))
        except Exception:
            continue
        by_id[t.get("id")] = (f, t)
        by_path[os.path.abspath(f)] = (f, t)

    chosen = []
    if args.trace_files:
        for f in args.trace_files:
            key = os.path.abspath(f)
            if key in by_path:
                chosen.append(by_path[key])
            else:
                t = json.load(open(f))
                chosen.append((f, t))
    elif args.ids:
        for _id in args.ids:
            if _id in by_id:
                chosen.append(by_id[_id])
            else:
                print(f"[WARN] id not found in trace dir: {_id}")
    else:
        # Auto-pick: samples whose previous run emitted (0,0). Prefer a mix of
        # reference cameras so we see both refcam=1 (pure repeat) and refcam>1.
        zero_files = [(f, t) for f, t in by_id.values() if _has_zero_zero(t)]
        zero_files.sort(key=lambda ft: ft[1].get("index", 0))
        chosen = zero_files[: args.num]

    out = []
    for f, t in chosen:
        out.append({
            "id": t.get("id"),
            "question": t.get("question", ""),
            "image_paths": t.get("image_paths", []),
            "ground_truth": t.get("ground_truth"),
            "orig_tool_calls": t.get("tool_calls", []),
            "trace_file": f,
        })
    return out


def build_agent(args):
    model = GPTModel(model_name=args.model, temperature=args.temperature, seed=args.seed)
    tools = [Pi3XTool(use_mock=False, server_url=args.pi3x_url)]  # mode='inference'
    # Spatial branch: no system_prompt override -> built-in spatial_3d prompt.
    agent = SPAgent(model=model, tools=tools, max_workers=4)
    return agent


def install_capture(agent):
    """Monkeypatch _run_model_inference to record (prompt, response) per call."""
    captured = []
    orig = agent._run_model_inference

    def wrapped(images, prompt, **kw):
        resp = orig(images, prompt, **kw)
        captured.append({
            "num_images": len(images) if images else 0,
            "images": list(images) if images else [],
            "prompt": prompt,
            "response": resp,
        })
        return resp

    agent._run_model_inference = wrapped
    return captured


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iterations", type=int, default=3)
    ap.add_argument("--pi3x-url", default="http://localhost:20031")
    ap.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--num", type=int, default=3, help="auto-pick N (0,0) samples")
    ap.add_argument("--ids", nargs="*", help="specific sample ids to re-run")
    ap.add_argument("--trace-files", nargs="*", help="specific trace json files")
    args = ap.parse_args()

    if not os.getenv("OPENAI_BASE_URL"):
        print("[WARN] OPENAI_BASE_URL is not set — inference will likely fail.\n"
              "       Run this in the shell where the model endpoint is exported.")

    os.makedirs(args.out_dir, exist_ok=True)
    samples = select_samples(args)
    if not samples:
        print("No samples selected.")
        return
    print(f"Model     : {args.model}  (temperature={args.temperature})")
    print(f"Endpoint  : {os.getenv('OPENAI_BASE_URL', '(default openai.com)')}")
    print(f"Selected  : {len(samples)} samples -> {[s['id'] for s in samples]}\n")

    agent = build_agent(args)
    captured = install_capture(agent)

    combined = []
    for s in samples:
        print("=" * 78)
        print(f"SAMPLE {s['id']}  (gt={s['ground_truth']})")
        print(f"Q: {s['question'][:160]}...")
        print(f"images: {len(s['image_paths'])}")
        captured.clear()
        t0 = time.time()
        try:
            result = agent.solve_problem(
                image_path=s["image_paths"],
                question=s["question"],
                max_iterations=args.max_iterations,
            )
            answer = result.get("answer", "")
            new_tool_calls = result.get("tool_calls", [])
            err = None
        except Exception as exc:
            answer, new_tool_calls, err = "", [], str(exc)
            print(f"[ERROR] {exc}")
        elapsed = round(time.time() - t0, 2)

        # snapshot captured iterations (deep copy the list contents)
        iters = [dict(c) for c in captured]
        for i, c in enumerate(iters, 1):
            print(f"\n----- iteration {i}  ({c['num_images']} imgs) -----")
            print("[RESPONSE]")
            print(c["response"])
        print(f"\n[FINAL ANSWER] {answer[:300]}")
        print(f"[NEW tool_calls] {json.dumps(_args_only(new_tool_calls), ensure_ascii=False)}")
        print(f"[elapsed] {elapsed}s\n")

        record = {
            "id": s["id"],
            "ground_truth": s["ground_truth"],
            "question": s["question"],
            "image_paths": s["image_paths"],
            "orig_tool_calls": _args_only(s["orig_tool_calls"]),
            "rerun_tool_calls": _args_only(new_tool_calls),
            "final_answer": answer,
            "elapsed_s": elapsed,
            "error": err,
            "iterations": iters,
        }
        combined.append(record)
        out_path = os.path.join(args.out_dir, f"{_safe(s['id'])}.json")
        json.dump(record, open(out_path, "w"), indent=2, ensure_ascii=False)
        print(f"saved -> {out_path}")

    combined_path = os.path.join(args.out_dir, "combined.json")
    json.dump(combined, open(combined_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved combined -> {combined_path}")


def _args_only(tool_calls):
    return [{"name": tc.get("name"), "arguments": tc.get("arguments")} for tc in (tool_calls or [])]


def _safe(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))


if __name__ == "__main__":
    main()
