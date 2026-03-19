"""
Skill System for SPAgent

Implements progressive disclosure of tool capabilities:
- Phase 1: Model sees only skill name + summary (compact index)
- Phase 2: After model selects a skill, full usage prompt is injected

Skills are defined in standalone SKILL.md files (YAML frontmatter + Markdown body)
so they can be edited, versioned, and evolved independently of Python code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
import logging
import re

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass
class Skill:
    """
    A skill represents a tool's metadata envelope for progressive disclosure.

    Attributes:
        name: Unique identifier (e.g. "depth_estimation")
        title: Display title (e.g. "Depth Estimation")
        summary: 1-2 sentence description visible in the skill index
        usage_prompt: Full usage instructions injected after selection
        tool_name: Associated Tool.name used for execution dispatch
        source_path: Path to the .md file this skill was loaded from (None if built from code)
    """
    name: str
    title: str
    summary: str
    usage_prompt: str
    tool_name: str
    source_path: Optional[str] = None


# ── File-based loading ────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_yaml_frontmatter(text: str) -> Dict[str, str]:
    """Minimal YAML frontmatter parser (no PyYAML dependency).

    Handles simple key: value pairs and multi-line values using ``>`` folded
    scalar syntax.  This is intentionally lightweight — we only need a handful
    of string fields.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    raw = m.group(1)
    result: Dict[str, str] = {}
    current_key: Optional[str] = None
    buffer_lines: List[str] = []

    def _flush():
        if current_key is not None:
            result[current_key] = " ".join(buffer_lines).strip()

    for line in raw.split("\n"):
        # New top-level key
        if re.match(r"^[a-zA-Z_][\w]*:", line):
            _flush()
            key, _, val = line.partition(":")
            current_key = key.strip()
            val = val.strip()
            if val == ">" or val == "|":
                buffer_lines = []
            else:
                buffer_lines = [val]
        else:
            buffer_lines.append(line.strip())
    _flush()
    return result


def load_skill_from_file(path: Path) -> Optional[Skill]:
    """Load a single Skill from a SKILL.md file.

    File format::

        ---
        name: depth_estimation
        title: Depth Estimation
        summary: >
          One or two sentence summary shown in the skill index.
        tool_name: depth_estimation_tool
        ---

        ## Full Usage Prompt (Markdown body)
        ...
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read skill file {path}: {e}")
        return None

    meta = _parse_yaml_frontmatter(text)
    if not meta.get("name") or not meta.get("tool_name"):
        logger.warning(f"Skill file {path} missing required frontmatter fields (name, tool_name)")
        return None

    body_match = _FRONTMATTER_RE.match(text)
    usage_prompt = text[body_match.end():].strip() if body_match else ""

    return Skill(
        name=meta["name"],
        title=meta.get("title", meta["name"].replace("_", " ").title()),
        summary=meta.get("summary", ""),
        usage_prompt=usage_prompt,
        tool_name=meta["tool_name"],
        source_path=str(path),
    )


def load_skills_from_directory(directory: Optional[Path] = None) -> Dict[str, Skill]:
    """Scan a directory for ``*.md`` files and load each as a Skill.

    Returns a dict keyed by ``skill.name``.
    """
    directory = directory or _SKILLS_DIR
    skills: Dict[str, Skill] = {}
    if not directory.is_dir():
        logger.warning(f"Skills directory not found: {directory}")
        return skills

    for md_file in sorted(directory.glob("*.md")):
        skill = load_skill_from_file(md_file)
        if skill:
            skills[skill.name] = skill
            logger.debug(f"Loaded skill from file: {skill.name} ({md_file.name})")
    logger.info(f"Loaded {len(skills)} skill(s) from {directory}")
    return skills


# ── Registry ──────────────────────────────────────────────────────────────────

class SkillRegistry:
    """Registry for managing available skills with progressive disclosure."""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    # ── bulk loading ──────────────────────────────────────────────────────

    def load_from_directory(self, directory: Optional[Path] = None):
        """Load all SKILL.md files from *directory* into the registry."""
        for skill in load_skills_from_directory(directory).values():
            self.register(skill)

    # ── single-item operations ────────────────────────────────────────────

    def register(self, skill: Skill):
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name} -> tool: {skill.tool_name}")

    def unregister(self, skill_name: str):
        if skill_name in self._skills:
            del self._skills[skill_name]
            logger.info(f"Unregistered skill: {skill_name}")

    def get(self, skill_name: str) -> Optional[Skill]:
        return self._skills.get(skill_name)

    def get_by_tool_name(self, tool_name: str) -> Optional[Skill]:
        """Look up a skill by its associated tool name."""
        for skill in self._skills.values():
            if skill.tool_name == tool_name:
                return skill
        return None

    def list_skills(self) -> List[str]:
        return list(self._skills.keys())

    def get_all_skills(self) -> Dict[str, Skill]:
        return self._skills.copy()

    def get_skill_index(self) -> str:
        """
        Generate a compact XML index containing only name, title, and summary
        for each registered skill.  This is what the model sees initially.
        """
        if not self._skills:
            return ""
        lines = []
        for skill in self._skills.values():
            lines.append(
                f'<skill name="{skill.name}" title="{skill.title}">\n'
                f'  {skill.summary}\n'
                f'</skill>'
            )
        return "\n".join(lines)

    def get_skill_detail(self, skill_name: str) -> Optional[str]:
        """Return the full usage_prompt for a selected skill."""
        skill = self._skills.get(skill_name)
        if skill:
            return skill.usage_prompt
        return None
