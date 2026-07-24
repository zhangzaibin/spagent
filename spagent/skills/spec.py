"""
Skill specs: derive everything a SKILL.md needs from the tool catalog.

Single source of truth (R1): tool name/description/parameters come from the
live ``Tool`` instances built via ``tools.catalog``; output contracts come
from ``core.tool_result.CATEGORY_CONTRACTS``; deployment facts come from
``skills.runtime``. The only hand-written content is the optional curated
overlay (``spagent/skills/overlays/<key>.md``, harvested from PR #157),
appended verbatim under "Guidance (curated)".

Rendering is deterministic (no timestamps, stable ordering), so
regeneration is idempotent.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool_result import CATEGORY_CONTRACTS, DETECTION, SEGMENTATION  # noqa: E402
from tools.catalog import TOOL_CATALOG, ToolCatalogEntry, _merge_kwargs  # noqa: E402

from .runtime import RuntimeSpec, get_runtime_spec  # noqa: E402

logger = logging.getLogger(__name__)

_OVERLAY_DIR = Path(__file__).parent / "overlays"

# supervision_tool resolves its output category per call (task argument).
_DUAL_CATEGORY_KEYS = {"supervision": (DETECTION, SEGMENTATION)}


@dataclass(frozen=True)
class SkillSpec:
    """Everything needed to render one skill folder."""

    key: str                      # catalog key ("zoom")
    tool_name: str                # runtime Tool.name ("zoom_object_tool")
    group: str
    categories: tuple             # 1 entry normally; 2 for dual-behavior tools
    description: str              # full tool description
    parameters: Dict[str, Any]    # OpenAI-style JSON schema
    runtime: RuntimeSpec
    mock_available: bool
    default_server_url: Optional[str]
    overlay: Optional[str]        # curated markdown or None

    @property
    def summary(self) -> str:
        """First sentence of the description (INDEX one-liner)."""
        first_line = self.description.strip().splitlines()[0] if self.description.strip() else ""
        # Sentence ends at a period followed by whitespace/EOL (so version
        # strings like "PaddleOCR-VL-1.5" survive).
        m = re.match(r"(.*?\.)(?:\s|$)", first_line)
        summary = (m.group(1) if m else first_line).strip()
        return summary or self.tool_name


# ---------------------------------------------------------------------------
# Schema extraction (build each tool once, in mock mode, then discard)
# ---------------------------------------------------------------------------

_MISSING_MODULE_RE = re.compile(r"No module named '([^']+)'")


def _build_for_schema(entry: ToolCatalogEntry):
    """Instantiate a catalog tool just to read ``description``/``parameters``.

    Minimal environments (system python without cv2/torch) can fail a mock
    constructor on an optional import. Since the instance is only inspected
    — never called — we retry with a throwaway stub module for the missing
    dependency, and afterwards evict the stub plus anything imported under
    it so the real modules load cleanly later in the same process.
    """
    kwargs = _merge_kwargs(entry, use_mock=True, overrides=None)
    try:
        return entry.cls(**kwargs)
    except Exception as exc:  # retry with stubs for missing optional deps
        last_error = exc

    stubs: Dict[str, types.ModuleType] = {}
    before = set(sys.modules)
    try:
        for _ in range(5):
            match = _MISSING_MODULE_RE.search(str(last_error))
            if not match:
                break
            missing = match.group(1)
            if missing in sys.modules:
                break  # present but broken — stubbing would not help
            stubs[missing] = sys.modules[missing] = types.ModuleType(missing)
            try:
                tool = entry.cls(**kwargs)
                logger.info("Schema probe for %s used stub(s): %s",
                            entry.key, sorted(stubs))
                return tool
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"cannot build {entry.key} for schema extraction: {last_error}"
        ) from last_error
    finally:
        # Undo the stubs, and evict only the modules that captured a stub
        # reference during the retry (e.g. a mock service that did
        # `import cv2`) so they re-import cleanly against the real
        # dependency later. Untainted imports (numpy internals, ...) must
        # stay: C-extension modules cannot be loaded twice per process.
        stub_objects = list(stubs.values())
        for name in stubs:
            sys.modules.pop(name, None)
        for name in set(sys.modules) - before:
            mod = sys.modules.get(name)
            if mod is not None and any(
                v is s for v in vars(mod).values() for s in stub_objects
            ):
                sys.modules.pop(name, None)


def build_skill_specs() -> List[SkillSpec]:
    """One SkillSpec per catalog entry, in catalog order."""
    specs: List[SkillSpec] = []
    for entry in TOOL_CATALOG:
        tool = _build_for_schema(entry)
        overlay_path = _OVERLAY_DIR / f"{entry.key}.md"
        overlay = overlay_path.read_text(encoding="utf-8").strip() if overlay_path.exists() else None
        categories = _DUAL_CATEGORY_KEYS.get(entry.key, (entry.category,))
        specs.append(SkillSpec(
            key=entry.key,
            tool_name=entry.tool_name,
            group=entry.group,
            categories=categories,
            description=tool.description or "",
            parameters=tool.parameters or {},
            runtime=get_runtime_spec(entry.key),
            mock_available=entry.accepts_use_mock,
            default_server_url=entry.default_kwargs.get("server_url"),
            overlay=overlay,
        ))
    return specs


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _md_cell(text: Any) -> str:
    """Flatten a value into a single markdown table cell."""
    s = str(text).replace("|", "\\|")
    return " ".join(s.split())


def _example_value(name: str, schema: Dict[str, Any]) -> Any:
    ptype = schema.get("type", "string")
    enum = schema.get("enum")
    if enum:
        return enum[0]
    if "image" in name:
        if ptype in ("array", "list"):
            return ["assets/dog.jpeg"]
        return "assets/dog.jpeg"
    if name in ("text_prompt", "object_name", "object_category"):
        return "dog"
    if name == "prompt":
        return "a dog"
    if name == "class_names":
        return ["dog"]
    if ptype in ("array", "list"):
        return []
    if ptype == "integer":
        return 1
    if ptype == "number":
        return 0
    if ptype == "boolean":
        return False
    return "<value>"


def example_args(spec: SkillSpec) -> Dict[str, Any]:
    """A runnable example argument object built from required parameters."""
    props = spec.parameters.get("properties", {})
    required = spec.parameters.get("required", [])
    return {name: _example_value(name, props.get(name, {})) for name in required}


def _argument_table(parameters: Dict[str, Any]) -> List[str]:
    props: Dict[str, Any] = parameters.get("properties", {}) or {}
    required = set(parameters.get("required", []) or [])
    lines = [
        "| Argument | Type | Required | Default | Description |",
        "|---|---|---|---|---|",
    ]
    for name, schema in props.items():  # dict order == schema order
        ptype = schema.get("type", "")
        if schema.get("enum"):
            enum_text = " \\| ".join(map(str, schema["enum"]))
            ptype = f"{ptype} ({enum_text})" if ptype else enum_text
        default = schema.get("default", "")
        lines.append(
            f"| `{name}` | {_md_cell(ptype)} | "
            f"{'yes' if name in required else 'no'} | "
            f"{_md_cell(default) if default != '' else '—'} | "
            f"{_md_cell(schema.get('description', ''))} |"
        )
    return lines


def _contract_section(category: str, heading_note: str = "") -> List[str]:
    contract = CATEGORY_CONTRACTS[category]
    lines = [f"### Category `{category}`{heading_note}", ""]
    lines.append("Raw payload — any ONE of these carrier groups must be present:")
    for group in contract.required_one_of:
        lines.append("- " + " + ".join(f"`{k}`" for k in group))
    if contract.optional_fields:
        lines.append("")
        lines.append("Common optional fields: "
                     + ", ".join(f"`{k}`" for k in contract.optional_fields))
    if contract.default_projection:
        lines.append("")
        lines.append("Default render projection: "
                     + ", ".join(f"`{k}`" for k in contract.default_projection))
    else:
        lines.append("")
        lines.append("Default render projection: description + visualization "
                     "images only.")
    return lines


def render_skill_md(spec: SkillSpec) -> str:
    """Render one SKILL.md (deterministic)."""
    primary_category = spec.categories[0]
    lines: List[str] = [
        "---",
        f"name: {spec.tool_name}",
        f"description: {_md_cell(spec.summary)}",
        f"category: {primary_category}",
        f"group: {spec.group}",
        f"runtime: {spec.runtime.runtime}",
        f"catalog_key: {spec.key}",
        "---",
        "",
        f"# {spec.tool_name}",
        "",
        "> Generated from `spagent/tools/catalog.py` — do not edit by hand. "
        "Regenerate with `python -m spagent.skills sync`.",
        "",
        "## When to use",
        "",
        spec.description.strip(),
        "",
        "## Arguments",
        "",
    ]
    lines.extend(_argument_table(spec.parameters))

    lines += ["", "## Output contract", ""]
    lines.append(
        "Every result is a JSON-serializable `ToolResult` envelope: `success`, "
        "`description`, `category`, `error` (on failure), plus visualization "
        "paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) "
        "when the tool produces images."
    )
    lines.append("")
    if len(spec.categories) > 1:
        lines.append(
            "**Dual-behavior tool:** the output category is resolved per call "
            "from the `task` argument — `task='image_det'` returns the "
            "detection contract, `task='image_seg'` returns the segmentation "
            "contract."
        )
        lines.append("")
        for cat in spec.categories:
            lines.extend(_contract_section(cat))
            lines.append("")
        lines.pop()
    else:
        lines.extend(_contract_section(primary_category))

    args_json = json.dumps(example_args(spec))
    mock_flag = " --use-mock" if spec.mock_available else ""
    lines += [
        "",
        "## Invocation",
        "",
        "```bash",
        f"python -m spagent.skills.run {spec.tool_name} --args '{args_json}'{mock_flag}",
        "```",
        "",
        "Prints the ToolResult as single-line JSON on stdout (non-JSON-safe "
        "values such as numpy arrays are summarized as "
        "`\"<array shape=... dtype=...>\"`); exits non-zero when "
        "`success` is false."
        + (" Drop `--use-mock` to hit the real backend."
           if spec.mock_available else ""),
    ]
    if spec.default_server_url:
        lines.append(
            f"Use `--server-url URL` to override the default backend "
            f"(`{spec.default_server_url}`), and `--output-dir DIR` to "
            "redirect artifacts when the tool supports it."
        )

    lines += ["", "## Runtime requirements", ""]
    lines.append(f"- Runtime class: **{spec.runtime.runtime}**")
    if spec.default_server_url:
        lines.append(f"- Server: `{spec.default_server_url}` "
                     "(health check: `GET /health`)")
    if spec.runtime.launch_command:
        lines.append(f"- Launch: `{spec.runtime.launch_command}`")
    if spec.runtime.checkpoint:
        lines.append(f"- Checkpoint: `{spec.runtime.checkpoint}`")
    for req in spec.runtime.requires:
        lines.append(f"- Requires: {req}")
    lines.append("- Mock available: "
                 + ("yes (`--use-mock`, no GPU/server needed)"
                    if spec.mock_available else "no"))
    if spec.runtime.notes:
        lines.append(f"- Notes: {spec.runtime.notes}")

    if spec.overlay:
        lines += ["", "## Guidance (curated)", "", spec.overlay]

    return "\n".join(lines).rstrip() + "\n"


def render_index_md(specs: List[SkillSpec]) -> str:
    """Render skills/INDEX.md — one line per skill (progressive disclosure)."""
    lines = [
        "# SPAgent skills index",
        "",
        "> Generated from `spagent/tools/catalog.py` — do not edit by hand. "
        "Regenerate with `python -m spagent.skills sync`.",
        "",
        "One line per skill. Read `skills/<name>/SKILL.md` for arguments, "
        "output contract, and runtime requirements BEFORE first use.",
        "",
    ]
    for spec in specs:
        runtime = spec.runtime.runtime
        if not spec.mock_available:
            runtime += ", no mock"
        lines.append(
            f"- **{spec.tool_name}** — {spec.summary} — "
            f"category: {'/'.join(spec.categories)} — runtime: {runtime}"
        )
    return "\n".join(lines) + "\n"
