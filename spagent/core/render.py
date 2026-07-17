"""
Parse/render module: project a tool result into the VLM-facing message.

Two-stage design (docs/Tool/TOOL_CONFIGURATIONS.md): tools PRODUCE a rich,
lossless standardized result; this module PROJECTS it — selecting which
fields reach the model according to user config, with a per-category default
and an "all" preset.

Projection resolution precedence (most specific wins):

    per-tool > per-category > global preset

Field lists REPLACE (never merge) at whichever level is set.

Config shape::

    render_config = {
        "preset": "default",              # "default" | "all"  (global fallback)
        "categories": {
            "detection": {"fields": ["labels", "boxes", "confidence"],
                          "coords": "pixel_xyxy"},
        },
        "tools": {
            "zoom_object_tool": {"fields": ["labels", "crop_paths"]},
        },
    }

Plain-dict results from un-migrated tools route through a LEGACY projection
that reproduces today's agent-loop behavior exactly (description passthrough;
output_path / vis_path / crop_paths in that order, existence-checked, no
dedup; ``.mp4`` paths returned as-is for the loop's frame extractor).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tool_result import (
    CATEGORY_CONTRACTS,
    ToolResult,
    visualization_paths,
)

logger = logging.getLogger(__name__)

PRESET_DEFAULT = "default"
PRESET_ALL = "all"

# Envelope/meta keys that are never part of a field projection
_NON_PAYLOAD_KEYS = frozenset({
    "success", "description", "error", "category", "result",
    "output_path", "vis_path", "overlay_path", "crop_paths", "message",
})


@dataclass
class RenderedOutput:
    """What the agent loop feeds the model for one tool result."""

    text: str
    images: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_projection(
    config: Optional[Mapping],
    category: Optional[str],
    tool_name: Optional[str],
) -> Dict[str, Any]:
    """Resolve the effective projection: tool > category > preset.

    Returns ``{"fields": list|None, "coords": str|None, "preset": str}`` —
    ``fields=None`` means "use the preset" (default projection or all).
    """
    config = config or {}
    resolved: Dict[str, Any] = {
        "preset": config.get("preset", PRESET_DEFAULT),
        "fields": None,
        "coords": None,
    }
    cat_cfg = (config.get("categories") or {}).get(category) if category else None
    tool_cfg = (config.get("tools") or {}).get(tool_name) if tool_name else None
    for cfg in (cat_cfg, tool_cfg):  # tool applied last -> wins
        if not cfg:
            continue
        if "fields" in cfg:
            resolved["fields"] = list(cfg["fields"])  # replace, never merge
        if "coords" in cfg:
            resolved["coords"] = cfg["coords"]
        if "preset" in cfg:
            resolved["preset"] = cfg["preset"]
    return resolved


# ---------------------------------------------------------------------------
# Value formatting (compact, VLM-friendly text)
# ---------------------------------------------------------------------------

_MAX_LIST_ITEMS = 20
_MAX_TEXT_CHARS = 2000


def _fmt(value: Any) -> str:
    if hasattr(value, "shape") and hasattr(value, "dtype"):  # numpy array/scalar
        if getattr(value, "ndim", 1) == 0:
            # 0-d numpy scalar (real clients often emit np.float32 confidences)
            # — show the value, not "<array shape=()>".
            return _fmt(value.item())
        return f"<array shape={tuple(value.shape)} dtype={value.dtype}>"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        items = [_fmt(v) for v in value[:_MAX_LIST_ITEMS]]
        suffix = f", …(+{len(value) - _MAX_LIST_ITEMS} more)" if len(value) > _MAX_LIST_ITEMS else ""
        return "[" + ", ".join(items) + suffix + "]"
    if isinstance(value, Mapping):
        inner = ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items())
        return "{" + inner + "}"
    s = str(value)
    return s if len(s) <= _MAX_TEXT_CHARS else s[:_MAX_TEXT_CHARS] + "…"


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def legacy_projection(result: Mapping) -> RenderedOutput:
    """Reproduce the historical agent-loop consumption.

    Order and semantics mirror the pre-render SPAgent loop:
    ``output_path`` (existence-checked; ``.mp4`` kept as-is for the caller's
    frame extraction), then ``vis_path`` (existence-checked, NOT deduped
    against output_path), then each of ``crop_paths``. Failures contribute no
    images. Text is the tool's ``description`` (or empty).

    One deliberate generalization: the caller now applies ``.mp4`` frame
    extraction to ANY returned image path, not just ``output_path``. No tool
    emits video outside ``output_path``, so observed behavior is unchanged;
    previously a video in ``vis_path``/``crop_paths`` would have been
    attached raw as an "image" (arguably a bug).
    """
    text = result.get("description") or ""
    images: List[str] = []
    if result.get("success"):
        out_path = result.get("output_path")
        if out_path is not None and Path(out_path).exists():
            images.append(out_path)
        vis_path = result.get("vis_path")
        if vis_path is not None and Path(vis_path).exists():
            images.append(vis_path)
        for crop in result.get("crop_paths") or []:
            if crop and Path(crop).exists():
                images.append(crop)
    return RenderedOutput(text=text, images=images)


def _select_fields(result: Mapping, category: str, projection: Dict[str, Any]) -> List[str]:
    contract = CATEGORY_CONTRACTS[category]
    if projection["fields"] is not None:
        return projection["fields"]
    if projection["preset"] == PRESET_ALL:
        ordered: List[str] = []
        for group in contract.required_one_of:
            ordered.extend(k for k in group if k not in ordered)
        ordered.extend(k for k in contract.optional_fields if k not in ordered)
        # any populated extras beyond the contract, deterministic order
        ordered.extend(sorted(
            k for k in result
            if k not in ordered and k not in _NON_PAYLOAD_KEYS
        ))
        return ordered
    return list(contract.default_projection)


def _project_boxes(result: Mapping, coords: Optional[str]) -> Any:
    """Return boxes, converting to pixel xyxy when requested and possible."""
    boxes = result.get("boxes")
    if coords != "pixel_xyxy" or not boxes:
        return boxes
    payload = getattr(result, "payload", None)
    if payload is not None and hasattr(payload, "to_xyxy_pixel"):
        try:
            return payload.to_xyxy_pixel()
        except ValueError as e:  # e.g. missing image dims
            logger.warning("coords=pixel_xyxy requested but %s; using native", e)
    return boxes


def standardized_projection(
    result: Mapping,
    category: str,
    projection: Dict[str, Any],
) -> RenderedOutput:
    """Project a contract-conformant result: draft description + selected fields."""
    if not result.get("success"):
        err = result.get("error") or "unknown error"
        return RenderedOutput(text=result.get("description") or f"Tool failed: {err}")

    lines: List[str] = []
    draft = result.get("description") or ""
    if draft:
        lines.append(draft)

    field_names = _select_fields(result, category, projection)
    for name in field_names:
        value = result.get(name)
        if name == "boxes":
            value = _project_boxes(result, projection.get("coords"))
        if value is None or (isinstance(value, (list, tuple, dict, str)) and not value):
            continue
        lines.append(f"{name}: {_fmt(value)}")

    return RenderedOutput(
        text="\n".join(lines),
        images=visualization_paths(result),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render(
    result: Mapping,
    config: Optional[Mapping] = None,
    tool_name: Optional[str] = None,
) -> RenderedOutput:
    """Project one tool result into ``(text, images)`` for the VLM.

    - ``ToolResult`` (or any mapping whose ``category`` names a known
      contract) → standardized projection under the resolved config.
    - Plain dict from an un-migrated tool → legacy projection, byte-identical
      to the historical agent-loop behavior.

    ``.mp4`` paths may appear in ``images``; the agent loop extracts frames.
    """
    category = result.get("category") if isinstance(result, Mapping) else None
    if isinstance(result, ToolResult) and result.category:
        category = result.category
    if category in CATEGORY_CONTRACTS:
        projection = resolve_projection(config, category, tool_name)
        return standardized_projection(result, category, projection)
    return legacy_projection(result)
