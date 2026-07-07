"""Unit tests for the parse/render module (zero VRAM).

Covers the spec's smoke-test assertions:
- default projection emits exactly the contract's default_projection fields
- "all" preset emits every populated field
- per-tool override beats per-category which beats preset
- legacy: a plain-dict result renders byte-identical to today's loop
  behavior (order, no dedup, .mp4 passthrough, failure -> no images)
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "spagent"))

from core.render import (  # noqa: E402
    RenderedOutput,
    legacy_projection,
    render,
    resolve_projection,
)
from core.tool_result import (  # noqa: E402
    BOX_CXCYWH_NORM,
    DETECTION,
    DetectionPayload,
    ToolResult,
)


def _touch(d, name):
    p = os.path.join(d, name)
    open(p, "w").close()
    return p


def _detection_result(**kw):
    payload = DetectionPayload(
        boxes=[[0.5, 0.5, 0.2, 0.2]], labels=["dog"],
        box_format=BOX_CXCYWH_NORM, confidence=[0.9],
        image_width=100, image_height=100,
    )
    return ToolResult(success=True, payload=payload,
                      description="found a dog", **kw)


# ---------------------------------------------------------------------------
# Legacy projection — must mirror the historical loop exactly
# ---------------------------------------------------------------------------

def test_legacy_matches_loop_semantics():
    d = tempfile.mkdtemp()
    out = _touch(d, "anno.png")
    crop = _touch(d, "crop.png")
    result = {
        "success": True,
        "description": "legacy tool output",
        "output_path": out,
        "vis_path": out,              # same file: loop appends it TWICE
        "crop_paths": [crop, os.path.join(d, "missing.png")],
    }
    r = legacy_projection(result)
    assert r.text == "legacy tool output"
    assert r.images == [out, out, crop]  # order + duplicate + existence-check


def test_legacy_mp4_passthrough_and_failure():
    d = tempfile.mkdtemp()
    mp4 = _touch(d, "video.mp4")
    ok = legacy_projection({"success": True, "output_path": mp4})
    assert ok.images == [mp4]  # .mp4 kept as-is; loop extracts frames
    fail = legacy_projection({"success": False, "error": "x",
                              "output_path": mp4, "description": "failed"})
    assert fail.images == [] and fail.text == "failed"


def test_plain_dict_routes_to_legacy():
    r = render({"success": True, "description": "d"}, config={"preset": "all"})
    assert isinstance(r, RenderedOutput) and r.text == "d"


# ---------------------------------------------------------------------------
# Standardized projections
# ---------------------------------------------------------------------------

def test_default_projection_exact_fields():
    res = _detection_result()
    r = render(res)  # no config -> default preset
    # draft description + exactly the contract's default fields (populated)
    lines = r.text.split("\n")
    assert lines[0] == "found a dog"
    keys = [ln.split(":")[0] for ln in lines[1:]]
    assert keys == ["labels", "boxes", "confidence"]


def test_all_preset_emits_every_populated_field():
    res = _detection_result()
    r = render(res, config={"preset": "all"})
    for key in ("detections", "boxes", "labels", "confidence",
                "box_format", "image_width", "image_height"):
        assert f"{key}:" in r.text, f"missing {key} in ALL projection"


def test_precedence_tool_beats_category_beats_preset():
    res = _detection_result()
    cfg = {
        "preset": "all",
        "categories": {DETECTION: {"fields": ["labels"]}},
        "tools": {"zoom_object_tool": {"fields": ["confidence"]}},
    }
    # category override beats preset
    r_cat = render(res, config=cfg, tool_name="other_tool")
    keys = [ln.split(":")[0] for ln in r_cat.text.split("\n")[1:]]
    assert keys == ["labels"]
    # tool override beats category
    r_tool = render(res, config=cfg, tool_name="zoom_object_tool")
    keys = [ln.split(":")[0] for ln in r_tool.text.split("\n")[1:]]
    assert keys == ["confidence"]


def test_coords_conversion_and_fallback():
    res = _detection_result()
    cfg = {"categories": {DETECTION: {"fields": ["boxes"], "coords": "pixel_xyxy"}}}
    r = render(res, config=cfg)
    assert "boxes: [[40, 40, 60, 60]]" in r.text  # cxcywh(0.5,0.5,.2,.2)@100px
    # without dims -> warning + native fallback (no crash)
    payload = DetectionPayload(boxes=[[0.5, 0.5, 0.2, 0.2]], labels=["a"])
    res2 = ToolResult(success=True, payload=payload, description="")
    r2 = render(res2, config=cfg)
    assert "0.5" in r2.text


def test_standardized_images_deduped_with_overlay():
    d = tempfile.mkdtemp()
    out = _touch(d, "a.png")
    ov = _touch(d, "ov.png")
    res = _detection_result(output_path=out, vis_path=out, overlay_path=ov)
    r = render(res)
    assert r.images == [out, ov]  # deduped; overlay consumed


def test_failure_standardized():
    res = ToolResult.fail("model exploded", category=DETECTION)
    r = render(res)
    assert r.images == [] and "model exploded" in r.text


def test_resolve_projection_shape():
    p = resolve_projection(None, DETECTION, None)
    assert p == {"preset": "default", "fields": None, "coords": None}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL {name}: {e}")
    print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
