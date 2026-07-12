"""
R3 execution backend — the skill "executable" SKILL.md examples invoke:

    python -m spagent.skills.run <tool_name> --args '<json>' \
        [--use-mock] [--server-url URL] [--output-dir DIR]

Builds the tool through ``tools.catalog.build_tools`` (catalog defaults +
CLI overrides), calls it, and prints the ToolResult as single-line JSON on
stdout. Values that are not JSON-safe (numpy arrays such as ``depth_data``)
are summarized as ``"<array shape=... dtype=...>"`` placeholders. Any
stdout noise from the tool itself is diverted to stderr so stdout stays
machine-parseable.

Exit codes: 0 = success, 1 = tool returned ``success: false`` (or raised),
2 = usage/config error, 3 = server health check failed (the structured
error includes the launch command from the skill's runtime spec).
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from tools.catalog import (  # noqa: E402
    TOOL_CATALOG,
    build_tools,
    resolve_tool_keys,
)

from .runtime import get_runtime_spec  # noqa: E402

logger = logging.getLogger(__name__)

_MAX_STR = 4000


class SkillRunError(Exception):
    """Configuration/availability failure with a structured JSON payload."""

    def __init__(self, payload: Dict[str, Any], exit_code: int = 2):
        super().__init__(payload.get("error", "skill run error"))
        self.payload = payload
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# JSON-safe projection of a ToolResult
# ---------------------------------------------------------------------------

def sanitize_for_json(value: Any, _depth: int = 0) -> Any:
    """Recursively convert a tool result into JSON-serializable values.

    Numpy arrays become shape/dtype placeholders (never crash the
    serializer); scalars unwrap; unknown objects degrade to truncated
    ``repr`` strings.
    """
    if _depth > 12:
        return "<max depth>"
    if value is None or isinstance(value, (bool, int, str)):
        return value if not isinstance(value, str) or len(value) <= _MAX_STR \
            else value[:_MAX_STR] + "…"
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) \
            else str(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):  # numpy array/scalar
        if getattr(value, "ndim", 1) == 0:
            return sanitize_for_json(value.item(), _depth + 1)
        return f"<array shape={tuple(value.shape)} dtype={value.dtype}>"
    if hasattr(value, "item") and hasattr(value, "dtype"):  # other np scalar
        return sanitize_for_json(value.item(), _depth + 1)
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): sanitize_for_json(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(v, _depth + 1) for v in value]
    r = repr(value)
    return r if len(r) <= _MAX_STR else r[:_MAX_STR] + "…"


# ---------------------------------------------------------------------------
# Tool construction + health check
# ---------------------------------------------------------------------------

def _resolve_entry(tool_name: str):
    resolved, unknown = resolve_tool_keys([tool_name])
    if unknown or not resolved:
        raise SkillRunError({
            "success": False,
            "skill": tool_name,
            "error": f"Unknown skill/tool: {tool_name!r}",
            "available": [e.tool_name for e in TOOL_CATALOG],
        })
    key = resolved[0]
    return next(e for e in TOOL_CATALOG if e.key == key)


def _constructor_accepts(cls, param: str) -> bool:
    try:
        return param in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _health_check(server_url: str, timeout: float = 5.0) -> Optional[str]:
    """Return None when healthy, else a short error string."""
    import requests
    url = server_url.rstrip("/") + "/health"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return None
        return f"health check returned HTTP {resp.status_code}"
    except requests.RequestException as e:
        return f"health check failed: {type(e).__name__}: {e}"


def build_skill_tool(
    tool_name: str,
    use_mock: bool = False,
    server_url: Optional[str] = None,
    output_dir: Optional[str] = None,
    check_server: bool = True,
    extra_overrides: Optional[Dict[str, Any]] = None,
):
    """Build one catalog tool with CLI overrides; health-check its server.

    ``extra_overrides``: additional constructor kwargs (SkillAgent's
    per-tool config); the explicit ``server_url``/``output_dir`` arguments
    win over them.
    """
    entry = _resolve_entry(tool_name)
    runtime_spec = get_runtime_spec(entry.key)

    overrides: Dict[str, Any] = dict(extra_overrides or {})
    if server_url:
        if "server_url" not in entry.default_kwargs and \
                not _constructor_accepts(entry.cls, "server_url"):
            raise SkillRunError({
                "success": False,
                "skill": entry.tool_name,
                "error": f"{entry.tool_name} does not take a server URL "
                         f"(runtime: {runtime_spec.runtime})",
            })
        overrides["server_url"] = server_url
    if output_dir:
        if _constructor_accepts(entry.cls, "output_dir"):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            overrides["output_dir"] = output_dir
        else:
            logger.warning(
                "%s does not accept output_dir; artifacts go to the tool's "
                "default location (usually ./outputs)", entry.tool_name)

    effective_url = (server_url or overrides.get("server_url")
                     or entry.default_kwargs.get("server_url"))
    if check_server and not use_mock and effective_url:
        problem = _health_check(effective_url)
        if problem:
            raise SkillRunError({
                "success": False,
                "skill": entry.tool_name,
                "error": f"Server for {entry.tool_name} is unreachable at "
                         f"{effective_url}: {problem}",
                "server_url": effective_url,
                "launch_command": runtime_spec.launch_command,
                "hint": "Start the server with the launch command (see the "
                        "skill's Runtime requirements), or pass --use-mock.",
            }, exit_code=3)

    try:
        tools, _ = build_tools(
            [entry.key],
            use_mock=use_mock,
            overrides={entry.key: overrides} if overrides else None,
            strict=True,
        )
    except Exception as e:
        raise SkillRunError({
            "success": False,
            "skill": entry.tool_name,
            "error": f"Failed to construct {entry.tool_name}: {e}",
            "hint": "Check the skill's Runtime requirements section "
                    "(missing dependency, checkpoint, or API key?).",
        }) from e
    return tools[0], entry


def run_skill(
    tool_name: str,
    args: Dict[str, Any],
    use_mock: bool = False,
    server_url: Optional[str] = None,
    output_dir: Optional[str] = None,
    check_server: bool = True,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Mapping, Any]:
    """In-process skill execution: returns ``(raw_result, tool)``.

    ``raw_result`` is the live ToolResult/dict (NOT sanitized) so callers
    like SkillAgent can feed it straight into ``core.render.render``.
    Raises :class:`SkillRunError` for config/availability failures.
    """
    tool, entry = build_skill_tool(
        tool_name, use_mock=use_mock, server_url=server_url,
        output_dir=output_dir, check_server=check_server,
        extra_overrides=extra_overrides,
    )
    try:
        # Keep stdout clean for the JSON envelope: tools/mocks that print
        # go to stderr instead.
        with contextlib.redirect_stdout(sys.stderr):
            result = tool.call(**args)
    except Exception as e:
        logger.exception("Skill %s raised", entry.tool_name)
        result = {
            "success": False,
            "description": f"{entry.tool_name} raised {type(e).__name__}",
            "error": f"{type(e).__name__}: {e}",
        }
    if not isinstance(result, Mapping):
        result = {
            "success": False,
            "description": f"{entry.tool_name} returned a non-mapping result",
            "error": f"unexpected result type: {type(result).__name__}",
        }
    return result, tool


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spagent.skills.run",
        description="Run one skill and print its ToolResult as JSON.",
    )
    parser.add_argument("tool_name", help="Tool function name or catalog key")
    parser.add_argument("--args", default="{}",
                        help="Tool arguments as a JSON object")
    parser.add_argument("--use-mock", action="store_true",
                        help="Use the mock backend (no GPU/server)")
    parser.add_argument("--server-url", default=None,
                        help="Override the tool's default server URL")
    parser.add_argument("--output-dir", default=None,
                        help="Redirect artifacts (tools that support it)")
    parser.add_argument("--no-health-check", action="store_true",
                        help="Skip the server health check")
    parsed = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    try:
        args = json.loads(parsed.args)
        if not isinstance(args, dict):
            raise ValueError("--args must be a JSON object")
    except ValueError as e:
        print(json.dumps({"success": False,
                          "error": f"Invalid --args JSON: {e}"}))
        return 2

    try:
        result, _ = run_skill(
            parsed.tool_name,
            args,
            use_mock=parsed.use_mock,
            server_url=parsed.server_url,
            output_dir=parsed.output_dir,
            check_server=not parsed.no_health_check,
        )
    except SkillRunError as e:
        print(json.dumps(sanitize_for_json(e.payload)))
        return e.exit_code

    print(json.dumps(sanitize_for_json(result)))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
