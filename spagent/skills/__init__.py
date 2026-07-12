"""
Skills mode for SPAgent — a second, additive tool-execution path.

Every catalog tool is packaged as a *skill*: a self-describing folder
(``skills/<tool_name>/SKILL.md``) generated from ``tools/catalog.py`` and
``core.tool_result.CATEGORY_CONTRACTS`` (single source of truth), plus a
compact ``skills/INDEX.md`` for progressive disclosure.

Layers:

- ``spagent.skills.generate`` — R1: generate skill folders + INDEX (idempotent)
- ``spagent.skills`` CLI       — R2: ``list | show <name> | sync``
- ``spagent.skills.run``       — R3: execution backend (ToolResult JSON on stdout)

The existing tool-call path is unchanged; skills mode is opt-in.
"""

import sys
from pathlib import Path

# Same import shim as spagent/tools/*: make `core` / `tools` importable
# whether this package is imported as `spagent.skills` (repo root on path)
# or as `skills` (spagent/ on path, the test-suite convention).
sys.path.append(str(Path(__file__).parent.parent))

from .registry import Skill, SkillRegistry, load_skill_file, default_skills_dir  # noqa: E402
from .spec import SkillSpec, build_skill_specs, render_skill_md, render_index_md  # noqa: E402
from .run import SkillRunError, run_skill, sanitize_for_json  # noqa: E402

__all__ = [
    "Skill",
    "SkillRegistry",
    "SkillRunError",
    "SkillSpec",
    "build_skill_specs",
    "default_skills_dir",
    "load_skill_file",
    "render_index_md",
    "render_skill_md",
    "run_skill",
    "sanitize_for_json",
]
