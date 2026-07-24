"""
R2 skill registry CLI:

    python -m spagent.skills list          # one line per installed skill
    python -m spagent.skills show <name>   # print a skill's full SKILL.md
    python -m spagent.skills sync          # regenerate from catalog + report drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generate import check_drift, generate
from .registry import SkillRegistry, default_skills_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spagent.skills",
        description="SPAgent skills registry.",
    )
    parser.add_argument("--skills-dir", type=Path, default=default_skills_dir(),
                        help="Skills directory (default: <repo>/skills)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List installed skills (INDEX view)")
    show = sub.add_parser("show", help="Print one skill's full SKILL.md")
    show.add_argument("name", help="Skill name (tool function name)")
    sub.add_parser("sync", help="Regenerate from the catalog and report drift")
    args = parser.parse_args(argv)

    if args.command == "sync":
        drift = check_drift(args.skills_dir)
        written, total = generate(args.skills_dir)
        if drift:
            print("Drift repaired:")
            print("\n".join(f"  {d}" for d in drift))
        orphans = [d for d in drift if d.startswith("orphan:")]
        print(f"Synced {total} skills under {args.skills_dir} "
              f"({len(written)} file(s) rewritten).")
        if orphans:
            print("NOTE: orphan skill folders were NOT deleted; remove them "
                  "manually if the tools are really gone.")
        return 0

    registry = SkillRegistry(args.skills_dir)

    if args.command == "list":
        print(registry.index_text.rstrip())
        return 0

    if args.command == "show":
        skill = registry.get(args.name)
        if skill is None:
            print(f"Unknown skill: {args.name!r}. Installed: "
                  f"{', '.join(registry.names())}", file=sys.stderr)
            return 2
        print(skill.body.rstrip())
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
