"""Integration smoke test: render module wired into the real SPAgent loop.

Zero VRAM: a scripted dummy Model + stub tools drive SPAgent.step()
end-to-end. Covers the spec's assertions:

- legacy: a plain-dict tool renders identically to the historical loop
  (description text in memory; output_path/vis_path duplicate preserved)
- standardized: a ToolResult tool projects per config; overlay_path reaches
  the model's image list (invisible to the legacy loop)
- mixed iteration: dict tool + ToolResult tool in one step completes
- render_config threading: constructor default and per-call override
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "spagent"))

from core import SPAgent  # noqa: E402
from core.model import Model  # noqa: E402
from core.tool import Tool  # noqa: E402
from core.tool_result import (  # noqa: E402
    BOX_CXCYWH_NORM,
    DetectionPayload,
    ToolResult,
)

ASSET = str(Path(__file__).parent.parent / "assets" / "dog.jpeg")


class ScriptedModel(Model):
    """Replays a fixed list of responses, ignoring the prompt."""

    def __init__(self, responses):
        super().__init__(model_name="scripted")
        self._responses = list(responses)
        self.calls = 0

    def _next(self):
        self.calls += 1
        return self._responses.pop(0) if self._responses else "<answer>done</answer>"

    def single_image_inference(self, image_path, prompt, **kw):
        return self._next()

    def multiple_images_inference(self, image_paths, prompt, **kw):
        return self._next()

    def text_only_inference(self, prompt, **kw):
        return self._next()


def _touch(d, name):
    p = os.path.join(d, name)
    open(p, "w").close()
    return p


class LegacyDictTool(Tool):
    """Un-migrated tool: returns a plain dict (historical shape)."""

    def __init__(self, out, crop):
        super().__init__(name="legacy_dict_tool", description="legacy")
        self._out, self._crop = out, crop

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    def call(self, **kw):
        return {
            "success": True,
            "description": "legacy says hi",
            "output_path": self._out,
            "vis_path": self._out,          # duplicate: legacy loop appends twice
            "crop_paths": [self._crop],
        }


class StandardizedTool(Tool):
    """Migrated tool: returns a ToolResult with an overlay visualization."""

    def __init__(self, overlay):
        super().__init__(name="standardized_tool", description="standardized")
        self._overlay = overlay

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    def call(self, **kw):
        payload = DetectionPayload(
            boxes=[[0.5, 0.5, 0.2, 0.2]], labels=["dog"],
            box_format=BOX_CXCYWH_NORM, confidence=[0.88],
            image_width=100, image_height=100,
        )
        return ToolResult(success=True, payload=payload,
                          description="std found a dog",
                          overlay_path=self._overlay)


def _tool_call(name):
    return f'<tool_call>{{"name": "{name}", "arguments": {{}}}}</tool_call>'


def _run_step(tools, responses, **agent_kw):
    model = ScriptedModel(responses)
    agent = SPAgent(model=model, tools=tools, **agent_kw)
    return agent.step(content="what is in the image?", images=ASSET,
                      max_tool_iterations=2)


def test_legacy_dict_tool_matches_historical_behavior():
    d = tempfile.mkdtemp()
    out, crop = _touch(d, "anno.png"), _touch(d, "crop.png")
    res = _run_step(
        [LegacyDictTool(out, crop)],
        [_tool_call("legacy_dict_tool"), "<answer>a dog</answer>"],
    )
    assert res.answer and "dog" in res.answer
    key = next(iter(res.tool_results))
    assert res.tool_results[key]["success"]
    # historical image collection: output_path, vis_path (dup), crop
    assert res.additional_images == [out, out, crop]
    # memory text uses the tool's own description (legacy path)
    entries = [e for e in res.memory.entries if e.entry_type == "tool_result"]
    assert entries and "legacy says hi" in entries[-1].text


def test_standardized_tool_projects_and_attaches_overlay():
    d = tempfile.mkdtemp()
    ov = _touch(d, "overlay.png")
    res = _run_step(
        [StandardizedTool(ov)],
        [_tool_call("standardized_tool"), "<answer>a dog</answer>"],
    )
    # overlay reaches the model (legacy loop would have dropped it)
    assert res.additional_images == [ov]
    # memory text is the rendered projection: draft + default fields
    entries = [e for e in res.memory.entries if e.entry_type == "tool_result"]
    text = entries[-1].text
    assert "std found a dog" in text
    assert "labels:" in text and "boxes:" in text and "confidence:" in text


def test_render_config_override_per_tool():
    d = tempfile.mkdtemp()
    ov = _touch(d, "overlay.png")
    cfg = {"tools": {"standardized_tool": {"fields": ["labels"]}}}
    res = _run_step(
        [StandardizedTool(ov)],
        [_tool_call("standardized_tool"), "<answer>ok</answer>"],
        render_config=cfg,
    )
    entries = [e for e in res.memory.entries if e.entry_type == "tool_result"]
    text = entries[-1].text
    assert "labels:" in text and "boxes:" not in text and "confidence:" not in text


def test_mixed_iteration_dict_and_toolresult():
    d = tempfile.mkdtemp()
    out, crop, ov = _touch(d, "o.png"), _touch(d, "c.png"), _touch(d, "ov.png")
    two_calls = (_tool_call("legacy_dict_tool")
                 + _tool_call("standardized_tool"))
    res = _run_step(
        [LegacyDictTool(out, crop), StandardizedTool(ov)],
        [two_calls, "<answer>both ran</answer>"],
    )
    assert len(res.tool_results) == 2
    assert all(r["success"] for r in res.tool_results.values())
    assert set(res.additional_images) == {out, crop, ov}


class ContractViolatingTool(Tool):
    """Claims the detection category but carries no required payload."""

    def __init__(self):
        super().__init__(name="violating_tool", description="bad")

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    def call(self, **kw):
        # dict with a known category but neither `detections` nor boxes+labels
        return {"success": True, "category": "detection",
                "description": "i promised boxes and delivered none"}


def test_contract_violation_warns_but_never_gates():
    import logging as _logging

    class _Capture(_logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record.getMessage())

    cap = _Capture()
    spagent_logger = _logging.getLogger("core.spagent")
    spagent_logger.addHandler(cap)
    prev_disable = _logging.root.manager.disable
    _logging.disable(_logging.NOTSET)  # this test needs warnings enabled
    try:
        res = _run_step(
            [ContractViolatingTool()],
            [_tool_call("violating_tool"), "<answer>done</answer>"],
        )
    finally:
        _logging.disable(prev_disable)
        spagent_logger.removeHandler(cap)
    # never gates: the step completes and the result is recorded
    assert res.answer and len(res.tool_results) == 1
    # but the drift is surfaced
    assert any("does not satisfy" in m and "detection" in m for m in cap.records), cap.records


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except Exception as e:
                failures += 1
                print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
