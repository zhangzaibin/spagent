"""REAL end-to-end for skills mode: yolo26 (local, CPU-friendly, no server).

Needs an env with ultralytics installed (the full spagent env); weights at
checkpoints/yolo26/yolo26n.pt auto-download on first use. Not part of the
zero-GPU suite (test_skills.py) — run explicitly:

    CUDA_VISIBLE_DEVICES="" python test/test_skills_real_yolo26.py

Covers: R3 skill-run CLI against the real backend + a SkillAgent
single-turn (scripted model, real tool execution).
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "spagent"))

from core.model import Model  # noqa: E402
from skills.agent import SkillAgent  # noqa: E402

ASSET = str(REPO_ROOT / "assets" / "dog.jpeg")


class ScriptedModel(Model):
    def __init__(self, responses):
        super().__init__(model_name="scripted")
        self._responses = list(responses)

    def _next(self):
        return self._responses.pop(0) if self._responses else "<answer>done</answer>"

    def single_image_inference(self, image_path, prompt, **kw):
        return self._next()

    def multiple_images_inference(self, image_paths, prompt, **kw):
        return self._next()

    def text_only_inference(self, prompt, **kw):
        return self._next()


def test_real_yolo26_skill_run_cli():
    proc = subprocess.run(
        [sys.executable, "-m", "spagent.skills.run", "yolo26_tool",
         "--args", json.dumps({"image_path": "assets/dog.jpeg"})],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    lines = [l for l in proc.stdout.strip().splitlines() if l]
    assert len(lines) == 1, f"stdout not single-line JSON: {lines}"
    result = json.loads(lines[0])
    assert result["success"] is True
    assert result["category"] == "detection"
    labels = [d.get("label") for d in result.get("detections", [])]
    assert "dog" in labels, f"expected a dog detection, got labels={labels}"


def test_real_yolo26_skill_agent_single_turn():
    agent = SkillAgent(model=ScriptedModel([
        "<skill_read>yolo26_tool</skill_read>",
        '<skill_run>{"skill": "yolo26_tool", '
        '"args": {"image_path": "assets/dog.jpeg"}}</skill_run>',
        "<answer>the image contains a dog</answer>",
    ]), use_mock=False)
    res = agent.step("what animal is in the image?", images=ASSET,
                     max_tool_iterations=4)
    assert "dog" in res.answer
    result = res.tool_results["yolo26_tool_iter2"]
    assert result["success"] is True
    labels = [d.get("label") for d in result.get("detections", [])]
    assert "dog" in labels
    # rendered projection reached memory; annotated image collected
    entries = [e for e in res.memory.entries if e.entry_type == "tool_result"
               and e.metadata.get("tool_name") == "yolo26_tool"]
    assert entries and "labels:" in entries[-1].text
    assert res.used_tools == ["yolo26_tool_iter2"]


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
