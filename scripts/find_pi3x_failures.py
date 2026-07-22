#!/usr/bin/env python3
"""Scan quick_eval trace files for samples where pi3x_tool was called but failed.

Detection principle (see spagent/core/spagent.py:326-327):
  - `tool_calls`  = every tool the agent *attempted*.
  - `used_tools`  = only tools whose result had success=True (suffixed `_iterN`).
So a sample that has "pi3x" in tool_calls but nothing "pi3x" in used_tools means
every pi3x reconstruction call failed (e.g. server 502 / "No result returned").

These failures are NOT saved as INFER_FAIL in predictions.json (the tool returns
{"success": False} instead of raising), so the sample keeps a degraded prediction
and would be skipped on resume. This script lists them, and can delete the matching
keys from predictions.json so a rerun redoes them.

Usage:
  # just report
  python scripts/find_pi3x_failures.py TRACES_DIR

  # report + cross-check / delete from a predictions.json
  python scripts/find_pi3x_failures.py TRACES_DIR --predictions PRED.json
  python scripts/find_pi3x_failures.py TRACES_DIR --predictions PRED.json --delete

  # also treat "partial" failures (some pi3x calls ok, some failed) as candidates
  python scripts/find_pi3x_failures.py TRACES_DIR --predictions PRED.json --include-partial
"""
import argparse
import json
import shutil
from pathlib import Path


def classify_trace(trace: dict) -> str:
    """Return one of: 'no_pi3x', 'ok', 'partial', 'full_fail', 'crash'."""
    # Agent-level crash (already stored as INFER_FAIL in predictions)
    if trace.get("error"):
        return "crash"

    tool_calls = trace.get("tool_calls", []) or []
    used_tools = trace.get("used_tools", []) or []

    pi3x_calls = sum(1 for tc in tool_calls if "pi3x" in str(tc.get("name", "")))
    if pi3x_calls == 0:
        return "no_pi3x"

    pi3x_success = sum(1 for u in used_tools if "pi3x" in str(u))
    if pi3x_success == 0:
        return "full_fail"
    if pi3x_success < pi3x_calls:
        return "partial"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces_dir", help="dir containing NNNNN.json trace files")
    ap.add_argument("--predictions", help="path to *_predictions.json to cross-check / edit")
    ap.add_argument("--include-partial", action="store_true",
                    help="also treat partial pi3x failures as rerun candidates")
    ap.add_argument("--delete", action="store_true",
                    help="delete failed sample ids from --predictions (makes a .bak)")
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir)
    trace_files = sorted(traces_dir.glob("[0-9]*.json"))
    if not trace_files:
        raise SystemExit(f"No NNNNN.json trace files found in {traces_dir}")

    buckets: dict[str, list[tuple[int, str]]] = {
        "full_fail": [], "partial": [], "crash": [],
    }
    total = 0
    for f in trace_files:
        try:
            trace = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] cannot parse {f.name}: {exc}")
            continue
        total += 1
        kind = classify_trace(trace)
        if kind in buckets:
            buckets[kind].append((trace.get("index", -1), str(trace.get("id", ""))))

    print(f"Scanned {total} traces in {traces_dir}")
    print(f"  full pi3x failures (called, 0 success) : {len(buckets['full_fail'])}")
    print(f"  partial pi3x failures (some failed)     : {len(buckets['partial'])}")
    print(f"  agent crashes (INFER_FAIL)              : {len(buckets['crash'])}")

    candidates = list(buckets["full_fail"])
    if args.include_partial:
        candidates += buckets["partial"]
    candidates.sort()

    print(f"\nRerun candidates ({len(candidates)}):")
    for idx, sid in candidates:
        print(f"  idx={idx:<5} id={sid}")

    if not args.predictions:
        print("\n(no --predictions given; nothing cross-checked or deleted)")
        return

    pred_path = Path(args.predictions)
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    cand_ids = [sid for _, sid in candidates]
    present = [sid for sid in cand_ids if sid in preds]
    print(f"\nOf {len(cand_ids)} candidate ids, {len(present)} are present in "
          f"{pred_path.name} (would be skipped on resume).")

    if not args.delete:
        print("Run again with --delete to remove them (a .bak backup will be made).")
        return

    if not present:
        print("Nothing to delete.")
        return

    backup = pred_path.with_suffix(pred_path.suffix + ".bak")
    shutil.copy(pred_path, backup)
    for sid in present:
        preds.pop(sid, None)
    pred_path.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    print(f"Backup -> {backup}")
    print(f"Deleted {len(present)} entries; {len(preds)} predictions remain.")
    print("Rerun the original eval command to redo them (ensure Pi3X server is up).")


if __name__ == "__main__":
    main()
