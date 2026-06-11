"""
Prompt Template Test

Verify that system prompts are built correctly for different configurations:
- Default 3D spatial prompt (no system_prompt passed)
- General vision prompt (GENERAL_VISION_SYSTEM_PROMPT)
- Custom prompt string with / without {tools_json} placeholder

Usage:
    python test/test_prompt.py
    python test/test_prompt.py --case all
    python test/test_prompt.py --case general
    python test/test_prompt.py --case 3d
    python test/test_prompt.py --case custom
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spagent.core.prompts import (
    SPATIAL_3D_WORKFLOW,
    GENERAL_VISION_WORKFLOW,
    SPATIAL_3D_SYSTEM_PROMPT,
    GENERAL_VISION_SYSTEM_PROMPT,
    TOOL_SELECTION_GUIDE,
    create_system_prompt,
    create_all_tools_system_prompt,
)
from spagent.tools import SegmentationTool, ObjectDetectionTool
from spagent.tools.catalog import build_all_tools, build_tools, resolve_tool_keys

DIVIDER = "=" * 80


class _MockModel:
    """Minimal model stub for prompt/workflow tests (no OpenAI dependency)."""

    model_name = "mock-model"

    def text_only_inference(self, prompt, **kwargs):
        return "<answer>A</answer>"

    def single_image_inference(self, image_path, prompt, **kwargs):
        return "<answer>A</answer>"

    def multiple_images_inference(self, image_paths, prompt, **kwargs):
        return "<answer>A</answer>"


def get_mock_tool_schemas():
    """Return tool schemas using mock tools (no real server needed)."""
    tools = [
        ObjectDetectionTool(use_mock=True),
        SegmentationTool(use_mock=True),
    ]
    schemas = [t.to_function_schema() for t in tools]
    return schemas


def print_section(title: str, content: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)
    print(content)
    print(DIVIDER)


def test_default_3d_prompt():
    """create_system_prompt() with no workflow arg → 3D spatial prompt."""
    schemas = get_mock_tool_schemas()
    prompt = create_system_prompt(schemas)

    print_section("CASE: Default 3D Prompt  (create_system_prompt, no args)", prompt)

    assert "azimuth" in prompt, "Expected 3D azimuth instruction"
    assert "elevation" in prompt, "Expected 3D elevation instruction"
    assert "detect_objects_tool" in prompt, "Expected tool name in prompt"
    assert "segment_image_tool" in prompt, "Expected tool name in prompt"
    print("✓  Default 3D prompt assertions passed.")


def test_general_vision_prompt():
    """create_system_prompt() with GENERAL_VISION_WORKFLOW → no 3D content."""
    schemas = get_mock_tool_schemas()
    prompt = create_system_prompt(schemas, workflow=GENERAL_VISION_WORKFLOW)

    print_section("CASE: General Vision Prompt  (create_system_prompt + GENERAL_VISION_WORKFLOW)", prompt)

    assert "azimuth" not in prompt, "General prompt should NOT contain azimuth"
    assert "elevation" not in prompt, "General prompt should NOT contain elevation"
    assert "detect_objects_tool" in prompt, "Expected tool name in prompt"
    assert "segment_image_tool" in prompt, "Expected tool name in prompt"
    print("✓  General vision prompt assertions passed.")


def test_spagent_system_prompt_none():
    """SPAgent with system_prompt=None → falls back to 3D prompt."""
    from spagent import SPAgent

    schemas = get_mock_tool_schemas()
    tools_json = json.dumps(schemas, indent=2)

    # Simulate what solve_problem does internally
    import json as _json
    from spagent.core.prompts import create_system_prompt as _csp

    prompt = _csp(schemas)  # default path (no system_prompt_template)
    print_section("CASE: SPAgent system_prompt=None  (default 3D)", prompt)

    assert "azimuth" in prompt
    print("✓  SPAgent default path (3D) assertions passed.")


def test_spagent_general_vision_system_prompt():
    """SPAgent with system_prompt=GENERAL_VISION_SYSTEM_PROMPT → tools injected, no 3D."""
    schemas = get_mock_tool_schemas()
    tools_json = json.dumps(schemas, indent=2)

    # Simulate what solve_problem does when system_prompt_template is set
    template = GENERAL_VISION_SYSTEM_PROMPT
    assert "{tools_json}" in template, "Template must contain {tools_json} placeholder"

    prompt = template.replace("{tools_json}", tools_json)
    print_section("CASE: SPAgent system_prompt=GENERAL_VISION_SYSTEM_PROMPT", prompt)

    assert "azimuth" not in prompt, "General prompt should NOT contain azimuth"
    assert "detect_objects_tool" in prompt, "Tool name should be injected via {tools_json}"
    assert "segment_image_tool" in prompt, "Tool name should be injected via {tools_json}"
    print("✓  SPAgent GENERAL_VISION_SYSTEM_PROMPT assertions passed.")


def test_custom_prompt_no_placeholder():
    """When no {tools_json} placeholder, tools block is appended automatically."""
    schemas = get_mock_tool_schemas()
    tools_json = json.dumps(schemas, indent=2)

    custom = "You are a custom specialist assistant. Only answer in JSON."

    # Simulate SPAgent solve_problem fallback path (no placeholder → append)
    tools_block = (
        f"\n# Tools\nYou have access to the following tools:\n"
        f"<tools>\n{tools_json}\n</tools>\n"
    )
    prompt = custom + tools_block
    print_section("CASE: Custom prompt (no placeholder) → tools appended", prompt)

    assert "detect_objects_tool" in prompt, "Tool name should be appended"
    assert "custom specialist" in prompt, "Custom preamble should be preserved"
    print("✓  Custom prompt (no placeholder) assertions passed.")


def test_workflow_constants():
    """Quick check that the workflow constants look sane."""
    print_section("CONSTANT: SPATIAL_3D_WORKFLOW", SPATIAL_3D_WORKFLOW)
    print_section("CONSTANT: GENERAL_VISION_WORKFLOW", GENERAL_VISION_WORKFLOW)

    assert "azimuth" in SPATIAL_3D_WORKFLOW
    assert "azimuth" not in GENERAL_VISION_WORKFLOW
    print("✓  Workflow constant assertions passed.")


def test_all_tools_prompt():
    """create_all_tools_system_prompt() includes schemas and selection guide."""
    tools, skipped = build_all_tools(use_mock=True, skip_unavailable=True)
    schemas = [t.to_function_schema() for t in tools]
    prompt = create_all_tools_system_prompt(schemas)

    print_section("CASE: All-Tools Prompt  (create_all_tools_system_prompt)", prompt[:4000] + "\n...[truncated]...")

    assert TOOL_SELECTION_GUIDE.strip() in prompt, "Expected tool selection guide"
    assert "Tool Selection Guide" in prompt
    assert "depth_estimation_tool" in prompt
    assert "pi3x_tool" in prompt
    assert "image_generation_sana_tool" in prompt
    assert "azimuth=0, elevation=0" in prompt
    if skipped:
        print(f"  (skipped during catalog build: {skipped})")
    print("✓  All-tools prompt assertions passed.")


def test_all_tools_agent_resolve_prompt():
    """SPAgent workflow_mode='all_tools' resolves the all-tools system prompt."""
    from spagent import SPAgent

    tools, _ = build_all_tools(use_mock=True, skip_unavailable=True)
    agent = SPAgent(model=_MockModel(), tools=tools, workflow_mode="all_tools")

    system_prompt, continuation_hint, workflow = agent._resolve_workflow_prompts(
        question="Which object is closer?",
        image_paths=["dummy.jpg"],
        tool_schemas=agent.tool_registry.get_function_schemas(),
    )

    print_section("CASE: SPAgent workflow_mode=all_tools", system_prompt[:4000] + "\n...[truncated]...")

    assert workflow == "all_tools"
    assert "Tool Selection Guide" in system_prompt
    assert "Continue investigating" in continuation_hint
    print("✓  SPAgent all_tools workflow assertions passed.")


def test_image_budget_helper():
    """_apply_image_budget keeps originals and trims old tool images."""
    from spagent import SPAgent

    agent = SPAgent(model=_MockModel())
    originals = ["a.jpg", "b.jpg"]
    additional = [f"tool_{i}.jpg" for i in range(6)]
    trimmed = agent._apply_image_budget(originals, additional, max_images_in_context=4)

    assert trimmed == ["a.jpg", "b.jpg", "tool_4.jpg", "tool_5.jpg"]
    print("✓  Image budget helper assertions passed.")


def test_compact_function_schemas():
    """ToolRegistry.get_function_schemas(compact=True) truncates descriptions."""
    from spagent.core.tool import ToolRegistry

    tools, _ = build_all_tools(use_mock=True, skip_unavailable=True)
    registry = ToolRegistry()
    for tool in tools[:3]:
        registry.register(tool)

    full = registry.get_function_schemas(compact=False)
    compact = registry.get_function_schemas(compact=True)

    assert len(full) == len(compact) == 3
    assert len(compact[0]["function"]["description"]) <= len(full[0]["function"]["description"])
    print("✓  Compact schema assertions passed.")


def test_build_tools_subset():
    """build_tools(tool_keys=[...]) loads only the requested tools."""
    tools, skipped = build_tools(
        tool_keys=["detection", "segment_image_tool", "pi3x"],
        use_mock=True,
        skip_unavailable=True,
    )
    names = {tool.name for tool in tools}
    assert names == {"detect_objects_tool", "segment_image_tool", "pi3x_tool"}
    resolved, unknown = resolve_tool_keys(["depth", "unknown_tool"])
    assert resolved == ["depth"]
    assert unknown == ["unknown_tool"]
    print("✓  build_tools subset assertions passed.")


def test_all_tools_step_smoke():
    """build_all_tools(use_mock=True) + one step returns an answer (model mocked)."""
    from spagent import SPAgent

    tools, skipped = build_all_tools(use_mock=True, skip_unavailable=True)
    agent = SPAgent(model=_MockModel(), tools=tools, workflow_mode="all_tools")

    def _fake_inference(images, prompt, **kwargs):
        return (
            "<think>Mock inference for smoke test.</think>\n"
            "<answer>A</answer>"
        )

    agent._run_model_inference = _fake_inference  # type: ignore[method-assign]

    result = agent.step(
        content="Which option is correct?",
        images=None,
        max_tool_iterations=1,
        max_images_in_context=4,
    )

    assert result.answer
    assert result.prompts.get("workflow") == "all_tools"
    if skipped:
        print(f"  (skipped during catalog build: {skipped})")
    print("✓  All-tools step smoke test passed.")


CASES = {
    "3d": test_default_3d_prompt,
    "general": test_general_vision_prompt,
    "all_tools": test_all_tools_prompt,
    "spagent_all_tools": test_all_tools_agent_resolve_prompt,
    "image_budget": test_image_budget_helper,
    "compact_schemas": test_compact_function_schemas,
    "build_tools_subset": test_build_tools_subset,
    "all_tools_step": test_all_tools_step_smoke,
    "spagent_none": test_spagent_system_prompt_none,
    "spagent_general": test_spagent_general_vision_system_prompt,
    "custom": test_custom_prompt_no_placeholder,
    "constants": test_workflow_constants,
}


def main():
    parser = argparse.ArgumentParser(description="Test system prompt construction")
    parser.add_argument(
        "--case",
        choices=list(CASES.keys()) + ["all"],
        default="all",
        help="Which test case to run (default: all)",
    )
    args = parser.parse_args()

    if args.case == "all":
        for name, fn in CASES.items():
            print(f"\n>>> Running: {name}")
            fn()
    else:
        CASES[args.case]()

    print(f"\n{'='*80}")
    print("  All selected tests passed.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
