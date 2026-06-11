"""
Tool catalog and factory for building all SPAgent tools at once.

Provides a single entry point to instantiate every tool listed in the README,
with grouped metadata reused by the all-tools system prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from core.tool import Tool

from .depth_tool import DepthEstimationTool
from .segmentation_tool import SegmentationTool
from .detection_tool import ObjectDetectionTool
from .supervision_tool import SupervisionTool
from .yoloe_tool import YOLOETool
from .yolo26_tool import YOLO26Tool
from .moondream_tool import MoondreamTool
from .molmo2_tool import Molmo2Tool
from .pi3_tool import Pi3Tool
from .pi3x_tool import Pi3XTool
from .vggt_tool import VGGTTool
from .mapanything_tool import MapAnythingTool
from .orient_anything_v2_tool import OrientAnythingV2Tool
from .sana_tool import SanaTool
from .veo_tool import VeoTool
from .sora_tool import SoraTool
from .wan_tool import WanTool
from .vace_tool import VaceTool
from .qwenvl_tool import QwenVLTool

logger = logging.getLogger(__name__)

ToolGroup = str  # "2d_perception" | "vlm" | "3d" | "generation"


@dataclass(frozen=True)
class ToolCatalogEntry:
    """Metadata for one catalog tool."""

    key: str
    cls: Type[Tool]
    group: ToolGroup
    tool_name: str
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    accepts_use_mock: bool = True


# Default server URLs aligned with docs/Tool/TOOL_USING.md
DEFAULT_SERVER_URLS: Dict[str, str] = {
    "depth": "http://127.0.0.1:20019",
    "segmentation": "http://127.0.0.1:20020",
    "detection": "http://127.0.0.1:20022",
    "moondream": "http://127.0.0.1:20024",
    "molmo2": "http://127.0.0.1:20025",
    "pi3": "http://127.0.0.1:20030",
    "pi3x": "http://127.0.0.1:20031",
    "vggt": "http://127.0.0.1:20032",
    "mapanything": "http://127.0.0.1:20033",
    "orient_anything_v2": "http://127.0.0.1:20034",
    "vace": "http://127.0.0.1:20034",
    "sana": "http://127.0.0.1:30000",
    "yoloe": "http://127.0.0.1:8000",
    "supervision": "http://127.0.0.1:8000",
}


TOOL_CATALOG: List[ToolCatalogEntry] = [
    # 2D perception
    ToolCatalogEntry(
        "depth",
        DepthEstimationTool,
        "2d_perception",
        "depth_estimation_tool",
        {"server_url": DEFAULT_SERVER_URLS["depth"]},
    ),
    ToolCatalogEntry(
        "segmentation",
        SegmentationTool,
        "2d_perception",
        "segment_image_tool",
        {"server_url": DEFAULT_SERVER_URLS["segmentation"]},
    ),
    ToolCatalogEntry(
        "detection",
        ObjectDetectionTool,
        "2d_perception",
        "detect_objects_tool",
        {"server_url": DEFAULT_SERVER_URLS["detection"]},
    ),
    ToolCatalogEntry(
        "supervision",
        SupervisionTool,
        "2d_perception",
        "supervision_tool",
        {"server_url": DEFAULT_SERVER_URLS["supervision"]},
    ),
    ToolCatalogEntry(
        "yoloe",
        YOLOETool,
        "2d_perception",
        "yoloe_detection_tool",
        {"server_url": DEFAULT_SERVER_URLS["yoloe"]},
    ),
    ToolCatalogEntry(
        "yolo26",
        YOLO26Tool,
        "2d_perception",
        "yolo26_tool",
        {},
        accepts_use_mock=False,
    ),
    ToolCatalogEntry(
        "qwenvl",
        QwenVLTool,
        "2d_perception",
        "qwenvl_detection_tool",
        {},
    ),
    # VLM
    ToolCatalogEntry(
        "moondream",
        MoondreamTool,
        "vlm",
        "moondream_tool",
        {"server_url": DEFAULT_SERVER_URLS["moondream"]},
    ),
    ToolCatalogEntry(
        "molmo2",
        Molmo2Tool,
        "vlm",
        "molmo2_tool",
        {"server_url": DEFAULT_SERVER_URLS["molmo2"]},
    ),
    # 3D
    ToolCatalogEntry(
        "pi3",
        Pi3Tool,
        "3d",
        "pi3_tool",
        {"server_url": DEFAULT_SERVER_URLS["pi3"]},
    ),
    ToolCatalogEntry(
        "pi3x",
        Pi3XTool,
        "3d",
        "pi3x_tool",
        {"server_url": DEFAULT_SERVER_URLS["pi3x"]},
    ),
    ToolCatalogEntry(
        "vggt",
        VGGTTool,
        "3d",
        "vggt_tool",
        {"server_url": DEFAULT_SERVER_URLS["vggt"]},
    ),
    ToolCatalogEntry(
        "mapanything",
        MapAnythingTool,
        "3d",
        "mapanything_tool",
        {"server_url": DEFAULT_SERVER_URLS["mapanything"]},
    ),
    ToolCatalogEntry(
        "orient_anything_v2",
        OrientAnythingV2Tool,
        "3d",
        "orient_anything_v2_tool",
        {"server_url": DEFAULT_SERVER_URLS["orient_anything_v2"]},
    ),
    # Generation
    ToolCatalogEntry(
        "sana",
        SanaTool,
        "generation",
        "image_generation_sana_tool",
        {"server_url": DEFAULT_SERVER_URLS["sana"]},
    ),
    ToolCatalogEntry(
        "veo",
        VeoTool,
        "generation",
        "video_generation_veo_tool",
        {},
    ),
    ToolCatalogEntry(
        "sora",
        SoraTool,
        "generation",
        "video_generation_sora_tool",
        {},
    ),
    ToolCatalogEntry(
        "wan",
        WanTool,
        "generation",
        "video_generation_wan_tool",
        {},
    ),
    ToolCatalogEntry(
        "vace",
        VaceTool,
        "generation",
        "video_generation_vace_tool",
        {"server_url": DEFAULT_SERVER_URLS["vace"]},
    ),
]


def _catalog_index() -> Dict[str, ToolCatalogEntry]:
    return {entry.key: entry for entry in TOOL_CATALOG}


def _catalog_by_tool_name() -> Dict[str, str]:
    """Map runtime tool.name -> catalog key."""
    return {entry.tool_name: entry.key for entry in TOOL_CATALOG}


def list_catalog_keys() -> List[str]:
    """Return all catalog keys in registration order."""
    return [entry.key for entry in TOOL_CATALOG]


def resolve_tool_keys(names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Normalize user-provided tool identifiers to catalog keys.

    Accepts either catalog keys (``"depth"``, ``"pi3x"``) or registered tool
    function names (``"depth_estimation_tool"``, ``"pi3x_tool"``).

    Returns:
        ``(resolved_keys, unknown_names)``
    """
    by_key = _catalog_index()
    by_tool_name = _catalog_by_tool_name()

    resolved: List[str] = []
    unknown: List[str] = []
    seen: set[str] = set()

    for name in names:
        key = name
        if key not in by_key:
            key = by_tool_name.get(name, name)
        if key in by_key:
            if key not in seen:
                resolved.append(key)
                seen.add(key)
        else:
            unknown.append(name)

    return resolved, unknown


def build_tools(
    tool_keys: Optional[List[str]] = None,
    use_mock: bool = False,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_unavailable: bool = True,
    strict: bool = False,
) -> Tuple[List[Tool], List[str]]:
    """
    Instantiate a selected subset of catalog tools.

    Args:
        tool_keys: Catalog keys or tool function names to load. When ``None``,
            all catalog tools are loaded (same as :func:`build_all_tools`).
        use_mock: When True, prefer mock clients where supported.
        overrides: Per-tool constructor overrides keyed by catalog ``key``.
        skip_unavailable: If True, log and skip tools that fail to construct.
        strict: If True, raise when ``tool_keys`` contains unknown names or when
            a requested tool fails to construct.

    Returns:
        ``(tools, skipped_keys)``
    """
    by_key = _catalog_index()

    if tool_keys is None:
        selected_keys = list_catalog_keys()
        unknown: List[str] = []
    else:
        selected_keys, unknown = resolve_tool_keys(tool_keys)
        if unknown:
            message = f"Unknown tool identifier(s): {unknown}. Available keys: {list_catalog_keys()}"
            if strict:
                raise ValueError(message)
            logger.warning(message)

    tools: List[Tool] = []
    skipped: List[str] = []

    for key in selected_keys:
        entry = by_key[key]
        kwargs = _merge_kwargs(entry, use_mock, overrides)
        try:
            tool = entry.cls(**kwargs)
            tools.append(tool)
            logger.info("Catalog: loaded tool %s (%s)", tool.name, entry.key)
        except Exception as exc:
            message = f"Catalog: failed to load {entry.key}: {exc}"
            if skip_unavailable and not strict:
                logger.warning(message)
                skipped.append(entry.key)
            else:
                raise RuntimeError(message) from exc

    return tools, skipped


def get_catalog_by_group() -> Dict[ToolGroup, List[ToolCatalogEntry]]:
    """Return catalog entries grouped by capability."""
    grouped: Dict[ToolGroup, List[ToolCatalogEntry]] = {}
    for entry in TOOL_CATALOG:
        grouped.setdefault(entry.group, []).append(entry)
    return grouped


def _merge_kwargs(
    entry: ToolCatalogEntry,
    use_mock: bool,
    overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build constructor kwargs for one catalog entry."""
    kwargs = dict(entry.default_kwargs)
    if entry.accepts_use_mock:
        kwargs["use_mock"] = use_mock

    if overrides and entry.key in overrides:
        kwargs.update(overrides[entry.key])

    return kwargs


def build_all_tools(
    use_mock: bool = False,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_unavailable: bool = True,
) -> Tuple[List[Tool], List[str]]:
    """
    Instantiate every tool in the catalog.

    Args:
        use_mock: When True, prefer mock clients where supported.
        overrides: Per-tool constructor overrides keyed by catalog ``key``
            (e.g. ``{"depth": {"server_url": "http://localhost:20019"}}``).
        skip_unavailable: If True, log and skip tools that fail to construct
            instead of raising.

    Returns:
        ``(tools, skipped_keys)`` where ``skipped_keys`` lists catalog keys
        that could not be instantiated.
    """
    return build_tools(
        tool_keys=None,
        use_mock=use_mock,
        overrides=overrides,
        skip_unavailable=skip_unavailable,
    )


def list_catalog_tool_names(use_mock: bool = True) -> List[str]:
    """Return function names for all tools that can be built in mock mode."""
    tools, _ = build_all_tools(use_mock=use_mock, skip_unavailable=True)
    return [tool.name for tool in tools]
