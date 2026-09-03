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

from tools.catalog import list_catalog_keys, resolve_tool_keys  # noqa: E402

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
    """All skills under one skills dir, plus the INDEX text.

    ``tool_keys``: restrict to this subset (catalog keys or tool function
    names, see ``tools.catalog.resolve_tool_keys``). ``None`` (default)
    loads every ``SKILL.md`` found under ``skills_dir``, unchanged from
    before this parameter existed. Unknown identifiers raise ``ValueError``
    up front rather than silently loading the full set.
    """

    def __init__(self, skills_dir: Optional[Path] = None,
                 tool_keys: Optional[List[str]] = None):
        self.skills_dir = Path(skills_dir) if skills_dir else default_skills_dir()
        self._skills: Dict[str, Skill] = {}
        self._index_text: Optional[str] = None
        self._allowed_keys: Optional[set] = None
        if tool_keys is not None:
            resolved, unknown = resolve_tool_keys(tool_keys)
            if unknown:
                raise ValueError(
                    f"Unknown tool identifier(s): {unknown}. "
                    f"Available keys: {list_catalog_keys()}"
                )
            self._allowed_keys = set(resolved)
        self._load()

    def _load(self) -> None:
        if not self.skills_dir.is_dir():
            raise FileNotFoundError(
                f"Skills directory not found: {self.skills_dir}. "
                "Run `python -m spagent.skills.generate` first."
            )
        for md in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill = load_skill_file(md)
            if skill is None:
                continue
            if self._allowed_keys is not None and \
                    skill.catalog_key not in self._allowed_keys:
                continue
            self._skills[skill.name] = skill
        index_path = self.skills_dir / "INDEX.md"
        if self._allowed_keys is None and index_path.exists():
            self._index_text = index_path.read_text(encoding="utf-8")
        logger.info("Loaded %d skill(s) from %s", len(self._skills), self.skills_dir)

    @property
    def index_text(self) -> str:
        """INDEX.md content.

        Without a subset filter, this is the on-disk INDEX.md verbatim
        (falls back to one line per loaded skill if missing). With a
        subset filter, the on-disk INDEX.md is full-catalog and would leak
        the excluded skills' names back in, so it is always rebuilt from
        the filtered ``self._skills`` instead.
        """
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
