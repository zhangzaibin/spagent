"""Unit tests for the standardized tool output (envelope + contracts + payloads).

Zero-VRAM: pure Python, runs with plain pytest or `python test/test_tool_result.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "spagent"))

from core.tool_result import (  # noqa: E402
    ALL_CATEGORIES,
    BOX_CXCYWH_NORM,
    BOX_XYXY_PIXEL,
    CATEGORY_CONTRACTS,
    DETECTION,
    DEPTH,
    OPTICAL_FLOW,
    ORIENTATION,
    DetectionPayload,
    DepthPayload,
    FlowPayload,
    OrientationPayload,
    PointsPayload,
    TextPayload,
    ToolResult,
    validate_payload,
    visualization_paths,
)


def test_contracts_cover_all_categories():
    assert set(CATEGORY_CONTRACTS) == set(ALL_CATEGORIES)
    for c in CATEGORY_CONTRACTS.values():
        assert c.required_one_of, c.category


def test_detection_payload_and_mapping():
    p = DetectionPayload(
        boxes=[[0.5, 0.5, 0.2, 0.2]], labels=["dog"],
        box_format=BOX_CXCYWH_NORM, confidence=[0.9],
        image_width=100, image_height=100,
    )
    r = ToolResult(success=True, payload=p, description="found a dog",
                   output_path=None)
    # dict compatibility
    assert r["success"] is True
    assert r.get("labels") == ["dog"]
    assert "boxes" in r and dict(r)["boxes"] == [[0.5, 0.5, 0.2, 0.2]]
    assert r.get("missing") is None
    # both carriers agree
    assert r["detections"][0]["bbox"] == r["boxes"][0]
    assert r["detections"][0]["confidence"] == 0.9
    # contract
    ok, unmet = r.validate()
    assert ok and not unmet
    # conversion: cxcywh norm -> pixel xyxy
    assert p.to_xyxy_pixel() == [[40, 40, 60, 60]]


def test_detection_pixel_passthrough_and_alignment():
    p = DetectionPayload(boxes=[[10, 10, 50, 50]], labels=["a"],
                         box_format=BOX_XYXY_PIXEL)
    assert p.to_xyxy_pixel() == [[10, 10, 50, 50]]
    try:
        DetectionPayload(boxes=[[0, 0, 1, 1]], labels=["a"], confidence=[])
        assert False, "misaligned confidence should raise"
    except ValueError:
        pass


def test_one_of_validation_on_plain_dicts():
    # detection satisfied by either carrier
    ok, _ = validate_payload({"detections": [{"bbox": [0, 0, 1, 1]}]}, DETECTION)
    assert ok
    ok, _ = validate_payload({"boxes": [[0, 0, 1, 1]], "labels": ["x"]}, DETECTION)
    assert ok
    ok, unmet = validate_payload({"boxes": [[0, 0, 1, 1]]}, DETECTION)  # no labels
    assert not ok and unmet
    # depth: array or path
    ok, _ = validate_payload({"depth_path": "/tmp/d.npy"}, DEPTH)
    assert ok
    # orientation: euler group must be complete
    ok, _ = validate_payload({"azimuth": 1, "elevation": 2, "rotation": 3}, ORIENTATION)
    assert ok
    ok, _ = validate_payload({"azimuth": 1, "elevation": 2}, ORIENTATION)
    assert not ok
    # flow
    ok, _ = validate_payload({"flow_path": "f.npy"}, OPTICAL_FLOW)
    assert ok


def test_failure_result_is_envelope_only():
    r = ToolResult.fail("boom", category=DETECTION)
    assert not r["success"] and r["error"] == "boom"
    ok, _ = r.validate()  # failures don't need a payload
    assert ok


def test_payload_one_of_constructors_reject_empty():
    for ctor in (
        lambda: DepthPayload(),
        lambda: FlowPayload(),
        lambda: OrientationPayload(),
    ):
        try:
            ctor()
            assert False, "empty payload should raise"
        except ValueError:
            pass
    # points conversion needs dims
    pp = PointsPayload(points=[{"x": 0.5, "y": 0.5}], normalized=True)
    try:
        pp.to_pixel()
        assert False
    except ValueError:
        pass
    pp2 = PointsPayload(points=[{"x": 0.5, "y": 0.5}], normalized=True,
                        image_width=200, image_height=100)
    assert pp2.to_pixel()[0]["x"] == 100.0


def test_text_payload():
    r = ToolResult(success=True, payload=TextPayload("hello"), description="ocr")
    assert r["text"] == "hello"
    ok, _ = r.validate()
    assert ok


def test_visualization_paths_dedup_and_overlay(tmp_path=None):
    import tempfile, os
    d = tempfile.mkdtemp()
    a = os.path.join(d, "a.png"); b = os.path.join(d, "b.png")
    for f in (a, b):
        open(f, "w").close()
    # output_path == vis_path -> deduped; overlay consumed; missing file skipped
    res = {"output_path": a, "vis_path": a, "overlay_path": b,
           "crop_paths": [a, os.path.join(d, "missing.png")]}
    assert visualization_paths(res) == [a, b]


def test_dict_subclass_json_and_isinstance():
    import json
    p = DetectionPayload(boxes=[[1, 2, 3, 4]], labels=["a"],
                         box_format=BOX_XYXY_PIXEL, confidence=[0.5])
    r = ToolResult(success=True, payload=p, description="d")
    # real dict: isinstance checks (DataCollector, eval trace cleaners) work
    assert isinstance(r, dict)
    # JSON-serializable at the type level (values permitting)
    parsed = json.loads(json.dumps(r))
    assert parsed["labels"] == ["a"] and parsed["success"] is True
    # attribute views stay consistent with the mapping
    assert r.description == "d" and r.category == DETECTION
    # payload lists are copied: mutating the mapping doesn't corrupt payload
    r["boxes"].append([9, 9, 9, 9])
    assert len(p.boxes) == 1


def test_empty_string_path_carrier_rejected():
    from core.tool_result import IMAGE_GENERATION, OCR
    ok, _ = validate_payload({"output_path": ""}, IMAGE_GENERATION)
    assert not ok, "empty path must not satisfy a path carrier"
    ok, _ = validate_payload({"ply_filename": ""}, "3d_reconstruction")
    assert not ok
    # text is the one carrier where empty string is a legitimate finding
    ok, _ = validate_payload({"text": ""}, OCR)
    assert ok


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
