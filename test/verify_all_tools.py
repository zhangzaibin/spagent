"""All-catalog verification: every tool builds (mock), returns ToolResult,
and satisfies its category contract. Zero/low VRAM.

Tools whose mock call needs unavailable deps are reported, not failed.
"""
import sys, os, logging, traceback
logging.disable(logging.WARNING)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "spagent"))
os.chdir(str(__import__("pathlib").Path(__file__).parent.parent))

from tools.catalog import TOOL_CATALOG, build_tools
from core.tool_result import ToolResult, validate_payload

IMG = "assets/dog.jpeg"

# per-key call kwargs for a minimal mock invocation
CALL_KW = {
    "depth":        dict(image_path=IMG),
    "segmentation": dict(image_path=IMG),
    "detection":    dict(image_path=IMG, text_prompt="dog"),
    "zoom":         dict(image_path=IMG, text_prompt="dog"),
    "localize":     dict(image_path=IMG, text_prompt="dog"),
    "supervision":  dict(image_path=IMG, task="image_det"),
    "yoloe":        dict(image_path=IMG, task="image", class_names=["dog"]),
    "yolo26":       dict(image_path=IMG),          # real local model (tiny)
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
}

rows = []
for entry in TOOL_CATALOG:
    key = entry.key
    status = notes = ""
    try:
        tools, errs = build_tools([key], use_mock=True)
        if not tools:
            rows.append((key, entry.category, "BUILD-FAIL", "; ".join(errs)[:60]))
            continue
        tool = tools[0]
        res = tool.call(**CALL_KW[key])
        is_tr = isinstance(res, ToolResult)
        if not res.get("success"):
            rows.append((key, entry.category, "CALL-FAIL", str(res.get("error"))[:60]))
            continue
        ok, unmet = validate_payload(res, res.get("category") or entry.category)
        status = ("ToolResult" if is_tr else "dict") + (" ✓contract" if ok else " ✗CONTRACT")
        if not is_tr:
            notes = "not migrated"
        if not ok:
            notes = f"unmet={unmet}"
    except Exception as e:
        status = "ERROR"
        notes = f"{type(e).__name__}: {e}"[:70]
    rows.append((key, entry.category, status, notes))

print(f"{'key':<20} {'category':<18} {'status':<24} notes")
print("-" * 90)
bad = 0
for key, cat, status, notes in rows:
    mark = "" if "✓" in status else ("  <-- " if status not in ("",) and "✓" not in status else "")
    if "✓" not in status:
        bad += 1
    print(f"{key:<20} {cat:<18} {status:<24} {notes}")
print("-" * 90)
print(f"{len(rows)} tools, {len(rows)-bad} fully standardized+compliant, {bad} needing attention")
