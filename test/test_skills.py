"""Skills-mode test suite (zero GPU — mock backends + scripted model).

Run with plain python:  python3 test/test_skills.py

Covers R5:
- generator idempotence (two generations are byte-identical; committed
  skills/ dir is in sync with the catalog)
- INDEX ↔ catalog consistency (24 skills, every tool name listed)
- R3 backend over 3+ mock tools: valid single-line ToolResult JSON on
  stdout, exit codes (0 success / 1 tool-failure / 2 usage / 3 server-down)
- SkillAgent end-to-end with a scripted dummy Model (pattern from
  test/test_render_integration.py:ScriptedModel) driving a mock skill
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "spagent"))

from core.model import Model  # noqa: E402
from skills.generate import check_drift, generate  # noqa: E402
from skills.registry import SkillRegistry, parse_frontmatter  # noqa: E402
from skills.run import run_skill, sanitize_for_json, SkillRunError  # noqa: E402
from skills.agent import SkillAgent  # noqa: E402
from tools.catalog import TOOL_CATALOG  # noqa: E402

ASSET = str(REPO_ROOT / "assets" / "dog.jpeg")
SKILLS_DIR = REPO_ROOT / "skills"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class ScriptedModel(Model):
    """Replays a fixed list of responses, ignoring the prompt."""

    def __init__(self, responses):
        super().__init__(model_name="scripted")
        self._responses = list(responses)
        self.prompts = []

    def _next(self, prompt):
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else "<answer>done</answer>"

    def single_image_inference(self, image_path, prompt, **kw):
        return self._next(prompt)

    def multiple_images_inference(self, image_paths, prompt, **kw):
        return self._next(prompt)

    def text_only_inference(self, prompt, **kw):
        return self._next(prompt)


def _read_tree(root: Path):
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*.md"))
    }


def _run_cli(*argv):
    proc = subprocess.run(
        [sys.executable, "-m", "spagent.skills.run", *argv],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    return proc


# ---------------------------------------------------------------------------
# R1: generator
# ---------------------------------------------------------------------------

def test_generator_idempotent():
    d1, d2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    try:
        _, n1 = generate(d1)
        written_again, _ = generate(d1)   # second run: nothing to rewrite
        assert written_again == [], f"regeneration rewrote: {written_again}"
        generate(d2)
        assert _read_tree(d1) == _read_tree(d2), "two generations differ"
        assert n1 == len(TOOL_CATALOG)
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


def test_committed_skills_in_sync_with_catalog():
    drift = check_drift(SKILLS_DIR)
    assert drift == [], f"skills/ drifted from catalog: {drift}"


def test_skill_md_structure():
    for entry in TOOL_CATALOG:
        path = SKILLS_DIR / entry.tool_name / "SKILL.md"
        assert path.exists(), f"missing {path}"
        text = path.read_text(encoding="utf-8")
        meta = parse_frontmatter(text)
        assert meta.get("name") == entry.tool_name
        assert meta.get("catalog_key") == entry.key
        for section in ("## When to use", "## Arguments", "## Output contract",
                        "## Invocation", "## Runtime requirements"):
            assert section in text, f"{path} lacks '{section}'"
    # dual-behavior tool documents both modes
    sup = (SKILLS_DIR / "supervision_tool" / "SKILL.md").read_text(encoding="utf-8")
    assert "Category `detection`" in sup and "Category `segmentation`" in sup
    assert "Dual-behavior" in sup


# ---------------------------------------------------------------------------
# R2: INDEX + registry
# ---------------------------------------------------------------------------

def test_index_matches_catalog():
    index = (SKILLS_DIR / "INDEX.md").read_text(encoding="utf-8")
    bullet_lines = [l for l in index.splitlines() if l.startswith("- **")]
    assert len(bullet_lines) == len(TOOL_CATALOG) == 24
    for entry in TOOL_CATALOG:
        assert f"- **{entry.tool_name}**" in index, f"{entry.tool_name} not in INDEX"
    for line in bullet_lines:
        assert "category:" in line and "runtime:" in line


def test_registry_loads_all_skills():
    reg = SkillRegistry(SKILLS_DIR)
    assert len(reg) == len(TOOL_CATALOG)
    zoom = reg.get("zoom_object_tool")
    assert zoom is not None and zoom.catalog_key == "zoom"
    assert "## Arguments" in zoom.body


# ---------------------------------------------------------------------------
# R3: execution backend
# ---------------------------------------------------------------------------

def test_run_cli_mock_tools_emit_valid_toolresult_json():
    cases = [
        ("zoom_object_tool",
         {"image_path": "assets/dog.jpeg", "text_prompt": "dog"}, "detection"),
        ("segment_image_tool",
         {"image_path": "assets/dog.jpeg"}, "segmentation"),
        ("supervision_tool",
         {"image_path": "assets/dog.jpeg", "task": "image_det"}, "detection"),
        ("molmo2_tool",
         {"image_path": "assets/dog.jpeg", "prompt": "dog"}, "point_grounding"),
    ]
    for tool_name, args, category in cases:
        proc = _run_cli(tool_name, "--args", json.dumps(args), "--use-mock")
        assert proc.returncode == 0, f"{tool_name}: {proc.stderr[-500:]}"
        lines = [l for l in proc.stdout.strip().splitlines() if l]
        assert len(lines) == 1, f"{tool_name}: stdout not single-line: {lines}"
        result = json.loads(lines[0])
        assert result["success"] is True
        assert result["category"] == category
        assert "description" in result


def test_run_cli_exit_codes():
    # tool-level failure (missing image) -> 1
    proc = _run_cli("zoom_object_tool", "--args",
                    '{"image_path": "does_not_exist.jpg", "text_prompt": "x"}',
                    "--use-mock")
    assert proc.returncode == 1
    assert json.loads(proc.stdout.strip())["success"] is False
    # unknown tool -> 2, structured error listing alternatives
    proc = _run_cli("nonexistent_tool", "--args", "{}")
    assert proc.returncode == 2
    err = json.loads(proc.stdout.strip())
    assert err["success"] is False and "available" in err
    # bad --args JSON -> 2
    proc = _run_cli("zoom_object_tool", "--args", "{not json")
    assert proc.returncode == 2
    # server down -> 3, error includes the launch command
    proc = _run_cli("depth_estimation_tool", "--args",
                    '{"image_path": "assets/dog.jpeg"}',
                    "--server-url", "http://127.0.0.1:1")
    assert proc.returncode == 3
    err = json.loads(proc.stdout.strip())
    assert err["success"] is False
    assert "launch_command" in err and "depth_server.py" in err["launch_command"]


def test_run_skill_in_process_and_unknown_skill():
    result, tool = run_skill(
        "zoom_object_tool",
        {"image_path": ASSET, "text_prompt": "dog"},
        use_mock=True,
    )
    assert result.get("success") is True and tool.name == "zoom_object_tool"
    try:
        run_skill("no_such_skill", {}, use_mock=True)
    except SkillRunError as e:
        assert e.payload["success"] is False
    else:
        raise AssertionError("unknown skill did not raise SkillRunError")


def test_sanitize_for_json_numpy_and_exotics():
    try:
        import numpy as np
    except ImportError:
        np = None
    payload = {
        "path": Path("/tmp/x.png"),
        "tup": (1, 2),
        "nested": {"inf": float("inf")},
    }
    if np is not None:
        payload["depth_data"] = np.zeros((4, 4), dtype="float32")
        payload["scalar"] = np.float32(0.5)
    out = sanitize_for_json(payload)
    json.dumps(out)  # must not raise
    if np is not None:
        assert out["depth_data"] == "<array shape=(4, 4) dtype=float32>"
        assert abs(out["scalar"] - 0.5) < 1e-6
    assert out["path"] == "/tmp/x.png"
    assert out["nested"]["inf"] == "inf"


# ---------------------------------------------------------------------------
# R4: SkillAgent end-to-end (scripted model + mock skill)
# ---------------------------------------------------------------------------

def _make_agent(responses):
    model = ScriptedModel(responses)
    return SkillAgent(model=model, skills_dir=SKILLS_DIR, use_mock=True), model


def test_skill_agent_read_run_answer():
    agent, model = _make_agent([
        "<skill_read>zoom_object_tool</skill_read>",
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "dog"}}'
        "</skill_run>",
        "<answer>there is a dog</answer>",
    ])
    res = agent.step("what animal is in the image?", images=ASSET,
                     max_tool_iterations=4)
    assert "dog" in res.answer
    assert res.tool_calls == [{
        "name": "zoom_object_tool",
        "arguments": {"image_path": "assets/dog.jpeg", "text_prompt": "dog"},
    }]
    assert list(res.tool_results) == ["zoom_object_tool_iter2"]
    assert res.tool_results["zoom_object_tool_iter2"]["success"] is True
    assert res.used_tools == ["zoom_object_tool_iter2"]
    # progressive disclosure: after the read, the full SKILL.md reaches the
    # model inside the continuation prompt (iteration 2)
    assert "## Arguments" in model.prompts[1]
    assert "zoom_object_tool" in model.prompts[1]
    # system prompt carries the INDEX, not the full docs
    assert "skills_index" in res.prompts["system_prompt"]
    assert "## Output contract" not in res.prompts["system_prompt"]
    # the run's rendered projection landed in memory (render() reuse)
    entries = [e for e in res.memory.entries if e.entry_type == "tool_result"
               and e.metadata.get("tool_name") == "zoom_object_tool"]
    assert entries and "labels:" in entries[-1].text and "boxes:" in entries[-1].text


def test_skill_agent_handles_unknown_skill_and_bad_json():
    agent, _ = _make_agent([
        '<skill_run>{"skill": "no_such_skill", "args": {}}</skill_run>'
        '<skill_run>{"skill": 42, "args": []}</skill_run>',
        "<answer>gave up</answer>",
    ])
    res = agent.step("q", images=ASSET, max_tool_iterations=3)
    assert res.answer and "gave up" in res.answer
    assert all(not r.get("success") for r in res.tool_results.values())
    assert res.used_tools == []


def test_skill_agent_forces_answer_when_missing():
    agent, model = _make_agent([
        "I will just think out loud without tags.",
        "still no tags",
        "<answer>forced final</answer>",
    ])
    res = agent.step("q", images=ASSET, max_tool_iterations=2)
    assert "forced final" in res.answer
    # 2 loop iterations + 1 final synthesis call
    assert len(model.prompts) == 3
    assert "final answer" in model.prompts[-1]


def test_skill_agent_render_config_override():
    cfg = {"tools": {"zoom_object_tool": {"fields": ["labels"]}}}
    agent, _ = _make_agent([
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "dog"}}'
        "</skill_run>",
        "<answer>ok</answer>",
    ])
    res = agent.step("q", images=ASSET, render_config=cfg)
    entries = [e for e in res.memory.entries if e.entry_type == "tool_result"]
    text = entries[-1].text
    assert "labels:" in text and "boxes:" not in text


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
