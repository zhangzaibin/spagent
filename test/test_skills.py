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


def test_registry_tool_keys_subset():
    reg = SkillRegistry(SKILLS_DIR, tool_keys=["pi3", "detection"])
    assert len(reg) == 2
    assert set(reg.names()) == {"pi3_tool", "detect_objects_tool"}
    assert reg.get("zoom_object_tool") is None
    # index is rebuilt from the filtered set, not the full on-disk INDEX.md
    assert "pi3_tool" in reg.index_text and "detect_objects_tool" in reg.index_text
    assert "zoom_object_tool" not in reg.index_text
    try:
        SkillRegistry(SKILLS_DIR, tool_keys=["not_a_real_tool"])
    except ValueError as e:
        assert "not_a_real_tool" in str(e)
    else:
        raise AssertionError("unknown tool_keys entry did not raise ValueError")


def test_generate_subset_omits_orphans_full_mode_reports_them():
    d = Path(tempfile.mkdtemp())
    try:
        (d / "not_a_real_tool_dir").mkdir()
        generate(d, tool_keys=["pi3"])
        assert (d / "pi3_tool" / "SKILL.md").exists()
        index = (d / "INDEX.md").read_text(encoding="utf-8")
        assert "pi3_tool" in index and "zoom_object_tool" not in index

        # subset check: the unrelated leftover folder is not an orphan
        subset_drift = check_drift(d, tool_keys=["pi3"])
        assert subset_drift == [], f"unexpected drift in subset mode: {subset_drift}"

        # full-catalog check: same folder IS reported as an orphan (plus
        # every other catalog skill is "missing" since only pi3 was written)
        full_drift = check_drift(d, tool_keys=None)
        assert any("orphan:" in line and "not_a_real_tool_dir" in line
                   for line in full_drift)
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
    assert list(res.tool_results) == ["zoom_object_tool_iter1"]
    assert res.tool_results["zoom_object_tool_iter1"]["success"] is True
    assert res.used_tools == ["zoom_object_tool_iter1"]
    # the read-only first turn is free: read -> run -> answer costs only 2
    # paid iterations, matching SPAgent's 2-round (tool_call + answer) floor
    assert res.iterations == 2
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


def test_skill_agent_free_read_cap_enforced():
    # read-only turns are free up to max_read_iterations (default 2); the
    # 3rd read-only turn in a row exceeds that cap and starts consuming the
    # paid budget just like run/answer turns -- otherwise a model could farm
    # unlimited "free" turns by reading one skill per turn forever.
    agent, model = _make_agent([
        "<skill_read>zoom_object_tool</skill_read>",
        "<skill_read>segment_image_tool</skill_read>",
        "<skill_read>detect_objects_tool</skill_read>",
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "dog"}}'
        "</skill_run>",
        "<answer>done</answer>",
    ])
    res = agent.step("q", images=ASSET, max_tool_iterations=4)
    assert res.prompts["free_read_iterations"] == "2"
    # 3rd read (paid) + run (paid) + answer (paid) = 3 paid iterations
    assert res.iterations == 3
    assert "done" in res.answer


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


def test_skill_agent_parses_args_containing_literal_close_tag():
    # a string arg that happens to contain the literal text "}</skill_run>"
    # must NOT truncate the JSON object — json.JSONDecoder.raw_decode (not
    # a lazy regex) has to find the *real* end of the object.
    trap = 'a}</skill_run>b'
    agent, _ = _make_agent([
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "' + trap + '"}}'
        "</skill_run>",
        "<answer>done</answer>",
    ])
    parsed = agent._parse_skill_runs(
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "' + trap + '"}}'
        "</skill_run>"
    )
    assert len(parsed) == 1
    assert parsed[0].get("parse_error") is None
    assert parsed[0]["skill"] == "zoom_object_tool"
    assert parsed[0]["args"]["text_prompt"] == trap


def test_skill_agent_parses_multiple_runs_after_a_trap_value():
    trap = 'x}</skill_run>y'
    response = (
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "' + trap + '"}}'
        "</skill_run>"
        '<skill_run>{"skill": "segment_image_tool", '
        '"args": {"image_path": "assets/dog.jpeg"}}</skill_run>'
    )
    agent, _ = _make_agent([])
    parsed = agent._parse_skill_runs(response)
    assert len(parsed) == 2
    assert parsed[0]["skill"] == "zoom_object_tool"
    assert parsed[0]["args"]["text_prompt"] == trap
    assert parsed[1]["skill"] == "segment_image_tool"
    assert parsed[1]["args"] == {"image_path": "assets/dog.jpeg"}


def test_skill_agent_parses_open_tag_literal_inside_args_string():
    # a string arg containing the literal text "<skill_run>" (the OPEN tag,
    # not the close tag tested above) must not be mistaken for the start of
    # a competing block: the scanner finds the real opening tag first, then
    # raw_decode consumes the whole JSON object atomically -- the embedded
    # literal text is never re-scanned as a tag boundary.
    trap = "demo tag literally: <skill_run> not real"
    response = (
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "' + trap + '"}}'
        "</skill_run>"
        '<skill_run>{"skill": "segment_image_tool", '
        '"args": {"image_path": "assets/dog.jpeg"}}</skill_run>'
    )
    agent, _ = _make_agent([])
    parsed = agent._parse_skill_runs(response)
    assert len(parsed) == 2
    assert parsed[0].get("parse_error") is None
    assert parsed[0]["skill"] == "zoom_object_tool"
    assert parsed[0]["args"]["text_prompt"] == trap
    assert parsed[1]["skill"] == "segment_image_tool"
    assert parsed[1]["args"] == {"image_path": "assets/dog.jpeg"}


def test_skill_agent_narrative_open_tag_does_not_swallow_following_real_block():
    # a literal "<skill_run>" appearing in plain prose (not followed by real
    # JSON) must not let the error-path fallback search reach past a
    # genuine block that starts later in the same response -- the fallback
    # must stop at the next opening tag, not the next closing tag (which
    # would actually belong to that later, real block).
    response = (
        "I mentioned <skill_run> in chat just now, ignore that. "
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "dog"}}'
        "</skill_run>"
    )
    agent, _ = _make_agent([])
    parsed = agent._parse_skill_runs(response)
    assert len(parsed) == 2
    assert parsed[0]["parse_error"] == "no JSON object found after <skill_run>"
    assert parsed[1].get("parse_error") is None
    assert parsed[1]["skill"] == "zoom_object_tool"
    assert parsed[1]["args"] == {"image_path": "assets/dog.jpeg", "text_prompt": "dog"}


def test_skill_agent_tool_keys_rejects_out_of_subset_run():
    # zoom_object_tool exists in the catalog but is excluded from this
    # agent's subset — the registry must be the hard boundary: the run is
    # rejected without ever reaching run_skill (which would happily
    # resolve it against the full catalog).
    model = ScriptedModel([
        '<skill_run>{"skill": "zoom_object_tool", '
        '"args": {"image_path": "assets/dog.jpeg", "text_prompt": "dog"}}'
        "</skill_run>",
        "<answer>done</answer>",
    ])
    agent = SkillAgent(model=model, skills_dir=SKILLS_DIR, use_mock=True,
                       tool_keys=["pi3"])
    assert agent.registry.get("zoom_object_tool") is None
    res = agent.step("q", images=ASSET, max_tool_iterations=3)
    result = res.tool_results["zoom_object_tool_iter1"]
    assert result["success"] is False
    assert "not in the active skill set" in result["error"]
    assert res.used_tools == []


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
