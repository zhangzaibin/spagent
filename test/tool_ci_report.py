"""Contract CI report for tool contributions (no-compute).

Runs every catalog tool in mock mode and checks it against the standardized
ToolResult contract (PR #230): build, parameter schema, mock call, contract
payload, render round-trip, box-convention sanity, and failure-path behavior.

Intended as the PR gate for new/changed tools:

    # full catalog, report only (exit 0 unless a *selected* tool fails)
    python test/tool_ci_report.py

    # gate only the tools touched by this PR (CI usage)
    python test/tool_ci_report.py --changed-from <base-sha>

    # gate specific tools
    python test/tool_ci_report.py --tools detection,zoom

A tool whose mock build/call needs unavailable heavy deps is reported as
DEP-SKIP and never fails the gate (mirrors verify_all_tools.py semantics).
Markdown report goes to $GITHUB_STEP_SUMMARY when set, and/or --markdown-out.
Zero/low VRAM; no servers required.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.disable(logging.WARNING)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spagent"))
os.chdir(str(REPO))

from tools.catalog import TOOL_CATALOG, build_tools            # noqa: E402
from core.tool_result import ToolResult, validate_payload      # noqa: E402
from core.render import render                                 # noqa: E402

IMG = "assets/dog.jpeg"

# Minimal mock-call kwargs per catalog key. A catalog entry with no row here
# is reported as NO-CALL-KW (a contribution gap), never a KeyError crash.
CALL_KW = {
    "depth":        dict(image_path=IMG),
    "segmentation": dict(image_path=IMG),
    "detection":    dict(image_path=IMG, text_prompt="dog"),
    "zoom":         dict(image_path=IMG, text_prompt="dog"),
    "localize":     dict(image_path=IMG, text_prompt="dog"),
    "supervision":  dict(image_path=IMG, task="image_det"),
    "yoloe":        dict(image_path=IMG, task="image", class_names=["dog"]),
    "yolo26":       dict(image_path=IMG),
    "face_detection": dict(image_path=IMG),
    "qwenvl":       dict(image_path=IMG, text_prompt="dog"),
    "moondream":    dict(image_path=IMG, task="point", object_name="dog"),
    "molmo2":       dict(image_path=IMG, prompt="Point to the dog"),
    "pi3":          dict(image_path=[IMG], azimuth_angle=30, elevation_angle=10),
    "pi3x":         dict(image_path=[IMG], azimuth_angle=30, elevation_angle=10),
    "vggt":         dict(image_path=[IMG], azimuth_angle=30, elevation_angle=10),
    "mapanything":  dict(image_path=[IMG], azimuth_angle=30, elevation_angle=10),
    "orient_anything_v2": dict(image_path=IMG, object_category="dog"),
    "sana":         dict(prompt="a dog"),
    "veo":          dict(prompt="a dog running"),
    "sora":         dict(prompt="a dog running"),
    "wan":          dict(prompt="a dog running"),
    "vace":         dict(image_path=IMG, prompt="dog walks"),
    "flowseek":     dict(image1_path=IMG, image2_path=IMG,
                         output_path="outputs/vflow.png"),
    "paddleocr_vl": dict(image_path=IMG),
    "wilddet3d":    dict(image_path=IMG, prompt_text="dog"),
    "countgd":      dict(image_path=IMG, text="dog"),
}

CHECKS = ["build", "schema", "call", "toolresult", "contract", "render", "boxes", "failpath", "docs"]

DOC_FILES = ("docs/Tool/TOOL_USING.md", "docs/Tool/EXTERNAL_EXPERTS.md")

# What each check verifies and how to fix a failure. Rendered as a legend in
# every report and quoted in the per-tool failure details.
CHECK_INFO = {
    "build": (
        "the tool constructs from the catalog with `use_mock=True` and zero VRAM",
        "register the tool in `spagent/tools/catalog.py`, accept `use_mock=True`, "
        "and provide a format-faithful mock client"),
    "schema": (
        "`tool.parameters` is a JSON schema dict with `type` and `properties` — "
        "this is the schema the VLM sees for tool-calling",
        "expose a `parameters` property returning "
        "`{'type': 'object', 'properties': {...}, 'required': [...]}`"),
    "call": (
        "a minimal mock call (kwargs from CALL_KW in test/tool_ci_report.py) "
        "returns a dict with `success: True`",
        "add/adjust the tool's CALL_KW row and make the mock path succeed on it"),
    "toolresult": (
        "the success result is a `ToolResult` envelope "
        "(`spagent/core/tool_result.py`), not a plain dict — plain dicts bypass "
        "contract validation and standardized rendering",
        "return `ToolResult(success=True, payload=<typed payload>, "
        "description=..., result=<raw>)`; keep legacy keys as extras"),
    "contract": (
        "the result satisfies its category contract (`validate_payload`): every "
        "category requires ONE OF its payload carriers "
        "(e.g. segmentation → masks/mask_path/polygon/rle; "
        "detection → boxes+labels; ocr → text)",
        "surface the required carrier field — the raw data usually already "
        "exists in the backend response and just isn't being returned"),
    "render": (
        "`render(result)` produces non-empty model-facing text under both the "
        "`default` and `all` presets — what the VLM would actually receive",
        "populate `description` and payload fields the projection can select"),
    "boxes": (
        "detection boxes project to sane pixels via `payload.to_xyxy_pixel()`: "
        "x2>x1 and y2>y1 — guards the normalized-cxcywh-under-an-xyxy-key bug "
        "that shipped for weeks (symptom: y2 < y1)",
        "emit boxes in the declared `box_format`; never relabel a convention "
        "without converting the values"),
    "failpath": (
        "calling with a nonexistent input image returns `success: False` "
        "without raising — agents feed tools bad paths routinely",
        "validate input paths at the top of `call()` (in mock mode too) and "
        "return an error dict instead of raising"),
    "docs": (
        "the tool is documented: its class name, tool name, or catalog key "
        "appears in docs/Tool/TOOL_USING.md or EXTERNAL_EXPERTS.md",
        "add the tool to the TOOL_USING.md tool table (and EXTERNAL_EXPERTS.md "
        "if it has a backend/server) — see other tools' rows for the format"),
}

PASS, FAIL, SKIP, NA = "✅", "❌", "⏭ dep", "—"


def _is_dep_error(exc: Exception) -> bool:
    return isinstance(exc, (ImportError, ModuleNotFoundError, FileNotFoundError, OSError))


_DOC_CACHE = None


def _doc_corpus() -> str:
    global _DOC_CACHE
    if _DOC_CACHE is None:
        parts = []
        for f in DOC_FILES:
            try:
                parts.append((REPO / f).read_text(encoding="utf-8"))
            except OSError:
                pass
        _DOC_CACHE = "\n".join(parts)
    return _DOC_CACHE


def _bad_image_kwargs(kw):
    """Clone kwargs with image path(s) pointing at a nonexistent file."""
    out = {}
    for k, v in kw.items():
        if "image" in k and "path" in k:
            out[k] = ["/nonexistent/ci_missing.jpg"] if isinstance(v, list) else "/nonexistent/ci_missing.jpg"
        else:
            out[k] = v
    return out


def check_tool(entry):
    """Run all checks for one catalog entry.

    Returns (results dict, notes list, fails dict) — ``fails`` maps a check
    name to the observed problem, used for the verbose failure-details section.
    """
    r = {c: NA for c in CHECKS}
    notes = []
    fails = {}
    key = entry.key

    def fail(check, msg, status=FAIL):
        r[check] = status
        notes.append(msg)
        if status == FAIL:
            fails[check] = msg

    # docs — static check, runs regardless of how the runtime checks go
    doc_text = _doc_corpus()
    if any(s in doc_text for s in (entry.cls.__name__, entry.tool_name, key)):
        r["docs"] = PASS
    else:
        fail("docs", f"neither `{entry.cls.__name__}`, `{entry.tool_name}`, nor "
                     f"`{key}` appears in {' or '.join(DOC_FILES)}")

    # build
    try:
        tools, errs = build_tools([key], use_mock=True)
    except Exception as e:
        fail("build", f"{type(e).__name__}: {e}"[:150],
             SKIP if _is_dep_error(e) else FAIL)
        return r, notes, fails
    if not tools:
        dep = any("No module" in e or "import" in e.lower() for e in errs)
        fail("build", "; ".join(errs)[:150], SKIP if dep else FAIL)
        return r, notes, fails
    tool = tools[0]
    r["build"] = PASS

    # schema
    try:
        p = tool.parameters
        if isinstance(p, dict) and p.get("type") and "properties" in p:
            r["schema"] = PASS
        else:
            fail("schema", f"`parameters` is {type(p).__name__} without "
                           "type/properties — the VLM cannot call this tool")
    except Exception as e:
        fail("schema", f"accessing `parameters` raised {type(e).__name__}: {e}"[:120])

    # call kwargs available?
    if key not in CALL_KW:
        fail("call", "no CALL_KW entry in test/tool_ci_report.py — the gate "
                     "cannot invoke this tool; add a minimal mock-call row")
        return r, notes, fails

    # mock call
    try:
        res = tool.call(**CALL_KW[key])
    except Exception as e:
        fail("call", f"mock call raised {type(e).__name__}: {e}"[:150],
             SKIP if _is_dep_error(e) else FAIL)
        return r, notes, fails
    if not isinstance(res, dict):
        fail("call", f"call returned {type(res).__name__}, expected a dict/ToolResult")
        return r, notes, fails
    if not res.get("success"):
        err = str(res.get("error"))[:150]
        dep = any(s in err.lower() for s in
                  ("not found", "no module named", "not installed",
                   "modulenotfounderror", "importerror"))
        fail("call", f"mock call returned success=False: {err}",
             SKIP if dep else FAIL)
        return r, notes, fails
    r["call"] = PASS

    # ToolResult migration (new tools must pass)
    if isinstance(res, ToolResult):
        r["toolresult"] = PASS
    else:
        fail("toolresult", "success result is a plain dict, not a ToolResult "
                           "envelope — it bypasses contract validation and "
                           "standardized rendering")

    # contract
    category = res.get("category") or entry.category
    try:
        ok, unmet = validate_payload(res, category)
        if ok:
            r["contract"] = PASS
        else:
            fail("contract", f"category `{category}` requires ONE OF these "
                             f"payload carriers, none present: {unmet}")
    except Exception as e:
        fail("contract", f"validate_payload raised {type(e).__name__}: {e}"[:120])

    # render round-trip, default + all presets
    try:
        for preset in (None, {"preset": "all"}):
            out = render(res, config=preset, tool_name=getattr(tool, "name", None))
            if not (out.text and out.text.strip()):
                fail("render", f"render() produced EMPTY text under preset="
                               f"{preset or 'default'} — the VLM would receive "
                               "nothing from this tool")
                break
        else:
            r["render"] = PASS
    except Exception as e:
        fail("render", f"render() raised {type(e).__name__}: {e}"[:120])

    # box-convention sanity (detection-style payloads only)
    boxes = res.get("boxes")
    payload = getattr(res, "payload", None)
    if boxes and payload is not None and hasattr(payload, "to_xyxy_pixel"):
        try:
            px = payload.to_xyxy_pixel()
            bad = [b for b in px if not (b[2] > b[0] and b[3] > b[1])]
            if bad:
                fail("boxes", f"pixel projection yields degenerate boxes "
                              f"(x2<=x1 or y2<=y1): {bad[:2]} — box_format "
                              "likely mislabels the actual convention")
            else:
                r["boxes"] = PASS
        except ValueError as e:
            fail("boxes", f"to_xyxy_pixel() failed: {e} — normalized boxes "
                          "need image_width/image_height in the payload")
    elif boxes:
        fail("boxes", "result has `boxes` but its payload lacks "
                      "to_xyxy_pixel() — use DetectionPayload")

    # failure path: bad image must yield success=False, never raise
    kw = CALL_KW[key]
    if any("image" in k and "path" in k for k in kw):
        try:
            bad_res = tool.call(**_bad_image_kwargs(kw))
            if isinstance(bad_res, dict) and not bad_res.get("success"):
                r["failpath"] = PASS
            else:
                fail("failpath", "call with a nonexistent image returned "
                                 f"success={bad_res.get('success')!r} — bad "
                                 "input paths must yield success=False")
        except Exception as e:
            fail("failpath", f"call with a nonexistent image RAISED "
                             f"{type(e).__name__} — must return an error dict "
                             "instead")

    return r, notes, fails


def changed_tool_keys(base):
    """Map files changed since `base` to catalog keys. core/ changes select all."""
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        capture_output=True, text=True, cwd=str(REPO), check=True,
    ).stdout.splitlines()
    if any(f.startswith("spagent/core/") for f in diff):
        return [e.key for e in TOOL_CATALOG], []
    stems = {Path(f).stem for f in diff if f.startswith("spagent/tools/") and f.endswith(".py")}
    keys = [e.key for e in TOOL_CATALOG
            if e.cls.__module__.rsplit(".", 1)[-1] in stems]
    orphans = stems - {e.cls.__module__.rsplit(".", 1)[-1] for e in TOOL_CATALOG} - {"catalog", "__init__"}
    return keys, sorted(orphans)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", help="comma-separated catalog keys to gate")
    ap.add_argument("--changed-from", help="git base sha: gate tools changed since it")
    ap.add_argument("--markdown-out", help="also write the markdown report here")
    args = ap.parse_args()

    gated = None
    orphans = []
    if args.tools:
        gated = [k.strip() for k in args.tools.split(",") if k.strip()]
        unknown = [k for k in gated if k not in {e.key for e in TOOL_CATALOG}]
        if unknown:
            print(f"❌ unknown tool key(s): {unknown}")
            return 1
    elif args.changed_from:
        gated, orphans = changed_tool_keys(args.changed_from)

    rows, gate_failures = [], []
    for entry in TOOL_CATALOG:
        results, notes, fails = check_tool(entry)
        rows.append((entry.key, entry.category, results, notes, fails))
        is_gated = gated is None or entry.key in gated
        if is_gated and fails:
            gate_failures.append(entry.key)

    # ---- report ----
    # Gated mode (--changed-from / --tools) shows ONLY the changed tools; the
    # rest of the catalog is still checked but collapsed into one fleet line.
    # Full-catalog table: run with no arguments.
    md = ["# Tool contract report (no-compute, mock mode)", ""]
    if gated is not None:
        md.append(f"**Changed tools in this PR** (only these gate the merge): "
                  f"`{', '.join(gated) or 'none'}`")
        md.append("")
    if orphans and not args.tools:
        md.append(f"⚠️ changed tool file(s) with **no catalog entry**: `{', '.join(orphans)}` "
                  "— new tools must be registered in `spagent/tools/catalog.py`.")
        md.append("")
        gate_failures.extend(f"unregistered:{o}" for o in orphans)

    shown = rows if gated is None else [r for r in rows if r[0] in gated]
    if shown:
        md.append("| tool | category | " + " | ".join(CHECKS) + " | notes |")
        md.append("|" + "---|" * (len(CHECKS) + 3))
        for key, cat, results, notes, _fails in shown:
            md.append(f"| {key} | {cat} | "
                      + " | ".join(results[c] for c in CHECKS)
                      + " | " + "; ".join(notes)[:160] + " |")
        md.append("")

    if gated is not None:
        rest = [r for r in rows if r[0] not in gated]
        broken = [k for k, _c, _r, _n, f in rest if f]
        skipped = [k for k, _c, r, _n, f in rest if not f and SKIP in r.values()]
        line = (f"Rest of the catalog ({len(rest)} unchanged tools, non-gating): "
                f"{len(rest) - len(broken) - len(skipped)} ✅")
        if skipped:
            line += f", {len(skipped)} ⏭ dep ({', '.join(skipped)})"
        if broken:
            line += f", {len(broken)} ❌ ({', '.join(broken)}) — pre-existing, not caused by this PR"
        md.append(line + ". Run `python test/tool_ci_report.py` for the full table.")
        md.append("")

    # Per-tool check report: every check as pass/fail with its success
    # criteria spelled out, and — on failure — what was observed and how to
    # fix it. Gated tools only (full-catalog mode would be 9 lines × 26 tools;
    # its failures are still listed, passes are left to the table).
    STATUS_WORD = {PASS: "✅ pass", FAIL: "❌ FAIL", SKIP: "⏭ dep-skipped",
                   NA: "— not applicable"}
    report_rows = [r for r in rows
                   if (gated is not None and r[0] in gated) or (gated is None and r[4])]
    if report_rows:
        md.append("## Check report" if gated is not None else "## Failure details")
        md.append("")
        for key, cat, results, _notes, fails in report_rows:
            n_fail = len(fails)
            verdict = "all checks passed" if not n_fail else f"{n_fail} check(s) failed"
            md.append(f"### `{key}` ({cat}) — {verdict}")
            md.append("")
            for check in CHECKS:
                status = results[check]
                if gated is None and status != FAIL:
                    continue  # full-catalog mode: failures only
                tests, fix = CHECK_INFO[check]
                md.append(f"- **{check}**: {STATUS_WORD[status]}")
                if status == NA:
                    continue
                md.append(f"  - *passes if:* {tests}")
                if check in fails:
                    md.append(f"  - *failed here:* {fails[check]}")
                    md.append(f"  - *fix:* {fix}")
                elif status == SKIP:
                    md.append("  - *skipped:* heavy dependency unavailable on this "
                              "runner — verified by the with-compute lane instead")
            md.append("")

    # legend: what every column verifies — full-catalog mode only; in gated
    # mode the Check report above already carries each check's criteria inline
    if gated is None:
        md.append("<details><summary>What each check tests</summary>")
        md.append("")
        for check in CHECKS:
            md.append(f"- **{check}** — {CHECK_INFO[check][0]}")
        md.append("")
        md.append("`⏭ dep` = unavailable heavy dependency on this runner: reported, "
                  "never gates (the with-compute lane covers it). `—` = not "
                  "applicable to this tool.")
        md.append("</details>")
        md.append("")

    if gate_failures:
        md.append(f"## ❌ gate failed: {', '.join(gate_failures)} — see Failure details above")
    else:
        md.append("## ✅ gate passed"
                  + ("" if gated is None else f" ({len(gated)} gated tool(s))"))
    report = "\n".join(md)

    print(report)
    for path in filter(None, [args.markdown_out, os.environ.get("GITHUB_STEP_SUMMARY")]):
        with open(path, "a") as f:
            f.write(report + "\n")

    return 1 if gate_failures else 0


if __name__ == "__main__":
    sys.exit(main())
