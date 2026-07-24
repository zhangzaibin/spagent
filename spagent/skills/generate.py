"""
R1 generator: ``python -m spagent.skills.generate [--skills-dir DIR] [--check]``

Writes one skill folder per catalog entry (``skills/<tool_name>/SKILL.md``)
plus ``skills/INDEX.md``, all derived from the tool catalog (see spec.py).
Regeneration is idempotent; ``--check`` reports drift without writing
(exit 1 when the on-disk skills differ from what the catalog generates).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from .registry import default_skills_dir  # noqa: E402
from .spec import build_skill_specs, render_index_md, render_skill_md  # noqa: E402

INDEX_NAME = "INDEX.md"
SKILL_NAME = "SKILL.md"


def generate_content() -> Tuple[Dict[str, str], str]:
    """Return ``({tool_name: SKILL.md text}, INDEX.md text)`` from the catalog."""
    specs = build_skill_specs()
    return {s.tool_name: render_skill_md(s) for s in specs}, render_index_md(specs)


def check_drift(skills_dir: Path) -> List[str]:
    """Compare on-disk skills with freshly generated content.

    Returns a list of human-readable drift lines (empty = in sync).
    """
    skills, index = generate_content()
    drift: List[str] = []
    for tool_name, content in skills.items():
        path = skills_dir / tool_name / SKILL_NAME
        if not path.exists():
            drift.append(f"missing: {path}")
        elif path.read_text(encoding="utf-8") != content:
            drift.append(f"stale: {path}")
    index_path = skills_dir / INDEX_NAME
    if not index_path.exists():
        drift.append(f"missing: {index_path}")
    elif index_path.read_text(encoding="utf-8") != index:
        drift.append(f"stale: {index_path}")
    # skill folders that no longer correspond to a catalog entry
    if skills_dir.is_dir():
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir() and child.name not in skills:
                drift.append(f"orphan: {child} (not in catalog)")
    return drift


def generate(skills_dir: Path) -> Tuple[List[str], int]:
    """Write all skill folders + INDEX. Returns ``(paths_written, n_skills)``."""
    skills, index = generate_content()
    written: List[str] = []
    for tool_name, content in skills.items():
        folder = skills_dir / tool_name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / SKILL_NAME
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
            written.append(str(path))
    index_path = skills_dir / INDEX_NAME
    if not index_path.exists() or index_path.read_text(encoding="utf-8") != index:
        index_path.write_text(index, encoding="utf-8")
        written.append(str(index_path))
    return written, len(skills)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spagent.skills.generate",
        description="Generate skill folders + INDEX.md from the tool catalog.",
    )
    parser.add_argument("--skills-dir", type=Path, default=default_skills_dir(),
                        help="Target directory (default: <repo>/skills)")
    parser.add_argument("--check", action="store_true",
                        help="Report drift without writing; exit 1 on drift")
    args = parser.parse_args(argv)

    if args.check:
        drift = check_drift(args.skills_dir)
        if drift:
            print("\n".join(drift))
            print(f"DRIFT: {len(drift)} item(s) out of sync with the catalog.")
            return 1
        print(f"OK: {args.skills_dir} is in sync with the catalog.")
        return 0

    written, total = generate(args.skills_dir)
    print(f"Generated {total} skills under {args.skills_dir} "
          f"({len(written)} file(s) updated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
