"""
Skill registry: load generated skill folders for progressive disclosure.

Phase 1: the orchestrator's context holds only ``skills/INDEX.md`` (one line
per skill). Phase 2: a full ``SKILL.md`` is read on demand when the model
asks for it. The dependency-free frontmatter parser follows PR #157's.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def default_skills_dir() -> Path:
    """``<repo_root>/skills`` (this file lives at <repo>/spagent/skills/)."""
    return Path(__file__).resolve().parent.parent.parent / "skills"


def parse_frontmatter(text: str) -> Dict[str, str]:
    """Minimal YAML frontmatter parser (simple ``key: value`` pairs only)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    meta: Dict[str, str] = {}
    for line in m.group(1).split("\n"):
        if re.match(r"^[A-Za-z_][\w-]*:", line):
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta


@dataclass(frozen=True)
class Skill:
    """One loaded skill package."""

    name: str                 # tool function name, e.g. "zoom_object_tool"
    description: str          # one-liner from frontmatter
    category: str
    runtime: str
    catalog_key: str
    body: str                 # full SKILL.md text (frontmatter included)
    path: str


def load_skill_file(path: Path) -> Optional[Skill]:
    """Load one SKILL.md; returns None (with a warning) when malformed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Cannot read skill file %s: %s", path, e)
        return None
    meta = parse_frontmatter(text)
    if not meta.get("name"):
        logger.warning("Skill file %s has no `name` in frontmatter", path)
        return None
    return Skill(
        name=meta["name"],
        description=meta.get("description", ""),
        category=meta.get("category", ""),
        runtime=meta.get("runtime", ""),
        catalog_key=meta.get("catalog_key", meta["name"]),
        body=text,
        path=str(path),
    )


class SkillRegistry:
    """All skills under one skills dir, plus the INDEX text."""

    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = Path(skills_dir) if skills_dir else default_skills_dir()
        self._skills: Dict[str, Skill] = {}
        self._index_text: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.skills_dir.is_dir():
            raise FileNotFoundError(
                f"Skills directory not found: {self.skills_dir}. "
                "Run `python -m spagent.skills.generate` first."
            )
        for md in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill = load_skill_file(md)
            if skill:
                self._skills[skill.name] = skill
        index_path = self.skills_dir / "INDEX.md"
        if index_path.exists():
            self._index_text = index_path.read_text(encoding="utf-8")
        logger.info("Loaded %d skill(s) from %s", len(self._skills), self.skills_dir)

    @property
    def index_text(self) -> str:
        """INDEX.md content (falls back to one line per loaded skill)."""
        if self._index_text is not None:
            return self._index_text
        return "\n".join(
            f"- **{s.name}** — {s.description} — category: {s.category} — "
            f"runtime: {s.runtime}"
            for s in self._skills.values()
        )

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def names(self) -> List[str]:
        return list(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)
