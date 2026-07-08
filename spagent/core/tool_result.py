"""
Standardized tool output: envelope, category contracts, and typed payloads.

Design (see docs/Tool/TOOL_CONFIGURATIONS.md — the registry table there is
normative):

- Every tool returns a *universal envelope*: ``success``, ``description``
  (a draft summary; the render module owns the final VLM-facing text),
  ``error`` on failure, and a rendered visualization for visual tools.
- Each of the 10 tool categories additionally requires a *raw informative
  payload*, accepted in any ONE OF several forms (e.g. a box as pixel xyxy OR
  normalized cxcywh; a mask as array OR file path OR polygon).
- Anything beyond that is optional per tool and never gates ``success``.

``ToolResult`` is dict-compatible (a ``Mapping``): existing consumers that do
``result.get("output_path")`` keep working unchanged, so tools can migrate
one at a time while plain-dict tools coexist in the same iteration.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Categories (fine-grained; the catalog's coarse `group` stays for prompts)
# ---------------------------------------------------------------------------

DETECTION = "detection"
SEGMENTATION = "segmentation"
IMAGE_GENERATION = "image_generation"
VIDEO_GENERATION = "video_generation"
RECONSTRUCTION_3D = "3d_reconstruction"
POINT_GROUNDING = "point_grounding"
DEPTH = "depth"
ORIENTATION = "orientation"
OPTICAL_FLOW = "optical_flow"
OCR = "ocr"

ALL_CATEGORIES: Tuple[str, ...] = (
    DETECTION,
    SEGMENTATION,
    IMAGE_GENERATION,
    VIDEO_GENERATION,
    RECONSTRUCTION_3D,
    POINT_GROUNDING,
    DEPTH,
    ORIENTATION,
    OPTICAL_FLOW,
    OCR,
)

# Box coordinate formats accepted by the Detection contract
BOX_XYXY_PIXEL = "xyxy_pixel"
BOX_XYXY_NORM = "xyxy_norm"
BOX_CXCYWH_NORM = "cxcywh_norm"

# Envelope visualization keys, in the order the renderer attaches them
VISUALIZATION_KEYS: Tuple[str, ...] = (
    "output_path",
    "vis_path",
    "overlay_path",
    "crop_paths",
)


# ---------------------------------------------------------------------------
# Category contracts — the code form of the doc's registry table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CategoryContract:
    """Output contract for one tool category.

    ``required_one_of``: the raw payload is satisfied when ANY inner group of
    keys is fully present (and truthy) on the result.
    ``default_projection``: payload fields the default render projection puts
    into the VLM-facing text (the "all" preset shows every populated field).
    """

    category: str
    required_one_of: Tuple[Tuple[str, ...], ...]
    optional_fields: Tuple[str, ...] = ()
    default_projection: Tuple[str, ...] = ()


CATEGORY_CONTRACTS: Dict[str, CategoryContract] = {
    DETECTION: CategoryContract(
        DETECTION,
        required_one_of=(("detections",), ("boxes", "labels")),
        optional_fields=(
            "confidence", "box_format", "image_width", "image_height",
            "class_id", "crop_paths", "masks",
        ),
        default_projection=("labels", "boxes", "confidence"),
    ),
    SEGMENTATION: CategoryContract(
        SEGMENTATION,
        required_one_of=(("masks",), ("mask_path",), ("polygon",), ("rle",)),
        optional_fields=("area", "bbox", "class_name", "shape",
                         "image_width", "image_height"),
        default_projection=(),
    ),
    IMAGE_GENERATION: CategoryContract(
        IMAGE_GENERATION,
        required_one_of=(("output_path",), ("image_paths",)),
        optional_fields=("seed", "model", "size", "file_size_bytes"),
        default_projection=("output_path",),
    ),
    VIDEO_GENERATION: CategoryContract(
        VIDEO_GENERATION,
        required_one_of=(("output_path",),),
        optional_fields=("duration", "resolution", "fps", "codec",
                         "result_dir", "frame_paths"),
        default_projection=("output_path",),
    ),
    RECONSTRUCTION_3D: CategoryContract(
        RECONSTRUCTION_3D,
        required_one_of=(("ply_filename",), ("points",)),
        optional_fields=("points_count", "view_count", "camera_views",
                         "camera_poses", "mesh_path", "scale_info"),
        default_projection=("ply_filename", "points_count"),
    ),
    POINT_GROUNDING: CategoryContract(
        POINT_GROUNDING,
        required_one_of=(("points",), ("points_by_image",)),
        optional_fields=("confidence", "labels", "image_width",
                         "image_height", "raw_text"),
        default_projection=("points",),
    ),
    DEPTH: CategoryContract(
        DEPTH,
        required_one_of=(("depth_data",), ("depth_path",)),
        optional_fields=("shape", "value_range", "confidence_map",
                         "normal_map"),
        default_projection=(),
    ),
    ORIENTATION: CategoryContract(
        ORIENTATION,
        required_one_of=(
            ("azimuth", "elevation", "rotation"),
            ("rotation_matrix",),
            ("quaternion",),
        ),
        optional_fields=("symmetry_order", "confidence"),
        default_projection=("azimuth", "elevation", "rotation"),
    ),
    OPTICAL_FLOW: CategoryContract(
        OPTICAL_FLOW,
        required_one_of=(("flow_path",), ("flow_array",)),
        optional_fields=("flow_shape", "flow_magnitude_mean",
                         "motion_boundaries", "confidence_map"),
        default_projection=("flow_magnitude_mean", "flow_path"),
    ),
    OCR: CategoryContract(
        OCR,
        required_one_of=(("text",),),
        optional_fields=("text_boxes", "confidence", "structured_data",
                         "reading_order"),
        default_projection=("text",),
    ),
}


def validate_payload(result: Mapping, category: str) -> Tuple[bool, List[Tuple[str, ...]]]:
    """Check a result (dict or ToolResult) against a category's raw contract.

    Returns ``(ok, unmet_groups)``. ``ok`` is True when any
    ``required_one_of`` group has every key present and not ``None``.
    Emptiness does NOT fail the check: a successful detection with zero
    boxes (``detections=[]``) or an OCR of a blank page (``text=""``) is a
    legitimate finding — the contract requires the carrier, not content.
    """
    contract = CATEGORY_CONTRACTS.get(category)
    if contract is None:
        raise KeyError(f"Unknown category: {category!r}")

    def carrier_present(k: str) -> bool:
        if k not in result:
            return False
        v = result.get(k)
        if v is None:
            return False
        # An empty string is a missing artifact for path-like carriers
        # (ply_filename="", flow_path="", …) — not a legitimate empty
        # finding. Only `text` may legitimately be the empty string
        # (e.g. OCR of a blank page).
        if isinstance(v, str) and v == "" and k != "text":
            return False
        return True

    unmet: List[Tuple[str, ...]] = []
    for group in contract.required_one_of:
        if all(carrier_present(k) for k in group):
            return True, []
        unmet.append(group)
    return False, unmet


def _truthy(v: Any) -> bool:
    """Non-empty check that tolerates numpy arrays (no ambiguous bool())."""
    if v is None:
        return False
    if hasattr(v, "size"):  # numpy array
        return bool(getattr(v, "size"))
    if isinstance(v, (str, bytes, list, tuple, dict)):
        return len(v) > 0
    return True


# ---------------------------------------------------------------------------
# Typed payloads — one per category; constructors accept any allowed form
# ---------------------------------------------------------------------------

class Payload:
    """Base: a payload contributes flat fields to the result mapping."""

    category: str = ""

    def to_fields(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class DetectionPayload(Payload):
    """Boxes + labels in any ONE OF the accepted coordinate formats."""

    category = DETECTION

    def __init__(
        self,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        box_format: str = BOX_CXCYWH_NORM,
        confidence: Optional[Sequence[Optional[float]]] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        if box_format not in (BOX_XYXY_PIXEL, BOX_XYXY_NORM, BOX_CXCYWH_NORM):
            raise ValueError(f"unknown box_format: {box_format!r}")
        if len(boxes) != len(labels):
            raise ValueError("boxes and labels must be parallel")
        if confidence is not None and len(confidence) != len(boxes):
            raise ValueError("confidence must align with boxes (use None entries)")
        self.boxes = [list(b) for b in boxes]
        self.labels = list(labels)
        self.box_format = box_format
        self.confidence = list(confidence) if confidence is not None else None
        self.image_width = image_width
        self.image_height = image_height

    @classmethod
    def from_detections(
        cls,
        detections: Sequence[Mapping],
        box_format: str = BOX_CXCYWH_NORM,
        **kw,
    ) -> "DetectionPayload":
        """Build from a ``[{bbox, label, confidence?}]`` carrier."""
        boxes = [d.get("bbox") or d.get("box") for d in detections]
        labels = [d.get("label", "obj") for d in detections]
        conf = [d.get("confidence") for d in detections]
        return cls(boxes, labels, box_format=box_format, confidence=conf, **kw)

    def to_xyxy_pixel(
        self, image_width: Optional[int] = None, image_height: Optional[int] = None
    ) -> List[List[int]]:
        """Convert boxes to pixel xyxy (needs image dims for normalized input)."""
        w = image_width or self.image_width
        h = image_height or self.image_height
        if self.box_format == BOX_XYXY_PIXEL:
            return [[int(round(v)) for v in b] for b in self.boxes]
        if w is None or h is None:
            raise ValueError("image dims required to convert normalized boxes")
        out: List[List[int]] = []
        for b in self.boxes:
            if self.box_format == BOX_XYXY_NORM:
                x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
            else:  # cxcywh_norm
                cx, cy, bw, bh = b[0] * w, b[1] * h, b[2] * w, b[3] * h
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            out.append([max(0, int(round(x1))), max(0, int(round(y1))),
                        min(w, int(round(x2))), min(h, int(round(y2)))])
        return out

    def to_fields(self) -> Dict[str, Any]:
        detections = [
            {"id": i, "bbox": b, "label": l,
             **({"confidence": c} if c is not None else {})}
            for i, (b, l, c) in enumerate(
                zip(self.boxes, self.labels,
                    self.confidence or [None] * len(self.boxes)))
        ]
        fields: Dict[str, Any] = {
            "detections": detections,
            "boxes": self.boxes,
            "labels": self.labels,
            "box_format": self.box_format,
        }
        if self.confidence is not None:
            fields["confidence"] = self.confidence
        if self.image_width is not None:
            fields["image_width"] = self.image_width
        if self.image_height is not None:
            fields["image_height"] = self.image_height
        return fields


class SegmentationPayload(Payload):
    """A mask in any ONE OF: array, file path, polygon coords, RLE."""

    category = SEGMENTATION

    def __init__(self, masks: Any = None, mask_path: Optional[str] = None,
                 polygon: Optional[Sequence] = None, rle: Any = None,
                 image_width: Optional[int] = None,
                 image_height: Optional[int] = None):
        if not any(_truthy(v) for v in (masks, mask_path, polygon, rle)):
            raise ValueError("one of masks/mask_path/polygon/rle is required")
        self.masks = masks
        self.mask_path = mask_path
        self.polygon = list(polygon) if polygon is not None else None
        self.rle = rle
        self.image_width = image_width
        self.image_height = image_height

    def to_fields(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        for k in ("masks", "mask_path", "polygon", "rle",
                  "image_width", "image_height"):
            v = getattr(self, k)
            if _truthy(v):
                f[k] = v
        return f


class PointsPayload(Payload):
    """Points in ONE OF normalized [0,1] or pixel coordinates."""

    category = POINT_GROUNDING

    def __init__(self, points: Sequence[Mapping], normalized: bool,
                 image_width: Optional[int] = None,
                 image_height: Optional[int] = None,
                 labels: Optional[Sequence[str]] = None):
        self.points = [dict(p) for p in points]
        self.normalized = normalized
        self.image_width = image_width
        self.image_height = image_height
        self.labels = list(labels) if labels is not None else None

    def to_pixel(self) -> List[Dict[str, float]]:
        if not self.normalized:
            return self.points
        if self.image_width is None or self.image_height is None:
            raise ValueError("image dims required to convert normalized points")
        return [
            {**p, "x": p["x"] * self.image_width, "y": p["y"] * self.image_height}
            for p in self.points
        ]

    def to_fields(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {"points": self.points,
                             "points_normalized": self.normalized}
        if self.labels is not None:
            f["labels"] = self.labels
        if self.image_width is not None:
            f["image_width"] = self.image_width
        if self.image_height is not None:
            f["image_height"] = self.image_height
        return f


class DepthPayload(Payload):
    """Depth field in ONE OF raw array or file path."""

    category = DEPTH

    def __init__(self, depth_data: Any = None, depth_path: Optional[str] = None,
                 shape: Optional[Sequence[int]] = None):
        if not _truthy(depth_data) and not _truthy(depth_path):
            raise ValueError("one of depth_data/depth_path is required")
        self.depth_data = depth_data
        self.depth_path = depth_path
        self.shape = list(shape) if shape is not None else None

    def to_fields(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        if _truthy(self.depth_data):
            f["depth_data"] = self.depth_data
        if _truthy(self.depth_path):
            f["depth_path"] = self.depth_path
        if self.shape:
            f["shape"] = self.shape
        return f


class FlowPayload(Payload):
    """Optical flow field in ONE OF raw (H,W,2) array or file path."""

    category = OPTICAL_FLOW

    def __init__(self, flow_array: Any = None, flow_path: Optional[str] = None,
                 flow_shape: Optional[Sequence[int]] = None,
                 flow_magnitude_mean: Optional[float] = None):
        if not _truthy(flow_array) and not _truthy(flow_path):
            raise ValueError("one of flow_array/flow_path is required")
        self.flow_array = flow_array
        self.flow_path = flow_path
        self.flow_shape = list(flow_shape) if flow_shape is not None else None
        self.flow_magnitude_mean = flow_magnitude_mean

    def to_fields(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        if _truthy(self.flow_array):
            f["flow_array"] = self.flow_array
        if _truthy(self.flow_path):
            f["flow_path"] = self.flow_path
        if self.flow_shape:
            f["flow_shape"] = self.flow_shape
        if self.flow_magnitude_mean is not None:
            f["flow_magnitude_mean"] = self.flow_magnitude_mean
        return f


class OrientationPayload(Payload):
    """Rotation in ONE OF euler angles, rotation matrix, or quaternion."""

    category = ORIENTATION

    def __init__(self, azimuth: Optional[float] = None,
                 elevation: Optional[float] = None,
                 rotation: Optional[float] = None,
                 rotation_matrix: Any = None, quaternion: Any = None,
                 symmetry_order: Optional[int] = None):
        euler_ok = all(v is not None for v in (azimuth, elevation, rotation))
        if not euler_ok and not _truthy(rotation_matrix) and not _truthy(quaternion):
            raise ValueError("one of euler{azimuth,elevation,rotation}/"
                             "rotation_matrix/quaternion is required")
        self.azimuth, self.elevation, self.rotation = azimuth, elevation, rotation
        self.rotation_matrix = rotation_matrix
        self.quaternion = quaternion
        self.symmetry_order = symmetry_order

    def to_fields(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        for k in ("azimuth", "elevation", "rotation", "symmetry_order"):
            v = getattr(self, k)
            if v is not None:
                f[k] = v
        if _truthy(self.rotation_matrix):
            f["rotation_matrix"] = self.rotation_matrix
        if _truthy(self.quaternion):
            f["quaternion"] = self.quaternion
        return f


class PointCloudPayload(Payload):
    """3D reconstruction in ONE OF .ply path or raw points array."""

    category = RECONSTRUCTION_3D

    def __init__(self, ply_filename: Optional[str] = None, points: Any = None,
                 points_count: Optional[int] = None,
                 camera_views: Optional[Sequence] = None):
        if not _truthy(ply_filename) and not _truthy(points):
            raise ValueError("one of ply_filename/points is required")
        self.ply_filename = ply_filename
        self.points = points
        self.points_count = points_count
        self.camera_views = camera_views

    def to_fields(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        if _truthy(self.ply_filename):
            f["ply_filename"] = self.ply_filename
        if _truthy(self.points):
            f["points"] = self.points
        if self.points_count is not None:
            f["points_count"] = self.points_count
        if _truthy(self.camera_views):
            f["camera_views"] = self.camera_views
        return f


class MediaPayload(Payload):
    """Generated media: image (path or list) or video (.mp4 path)."""

    def __init__(self, category: str, output_path: Optional[str] = None,
                 image_paths: Optional[Sequence[str]] = None,
                 metadata: Optional[Mapping] = None):
        if category not in (IMAGE_GENERATION, VIDEO_GENERATION):
            raise ValueError("MediaPayload category must be image/video generation")
        if not _truthy(output_path) and not _truthy(image_paths):
            raise ValueError("one of output_path/image_paths is required")
        self.category = category
        self.output_path = output_path
        self.image_paths = list(image_paths) if image_paths else None
        self.metadata = dict(metadata) if metadata else {}

    def to_fields(self) -> Dict[str, Any]:
        f: Dict[str, Any] = dict(self.metadata)
        if _truthy(self.output_path):
            f["output_path"] = self.output_path
        if _truthy(self.image_paths):
            f["image_paths"] = self.image_paths
        return f


class TextPayload(Payload):
    """OCR / document text."""

    category = OCR

    def __init__(self, text: str, structured_data: Any = None):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        self.text = text
        self.structured_data = structured_data

    def to_fields(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {"text": self.text}
        if _truthy(self.structured_data):
            f["structured_data"] = self.structured_data
        return f


# ---------------------------------------------------------------------------
# ToolResult — the dict-compatible envelope
# ---------------------------------------------------------------------------

class ToolResult(dict):
    """Standardized tool result: envelope + typed payload, a ``dict`` subclass.

    Being a real ``dict`` keeps every existing consumer working unchanged:
    ``result.get("boxes")`` / ``result["success"]`` / ``"labels" in result``,
    ``isinstance(result, dict)`` checks (DataCollector, eval trace cleaners),
    and ``json.dumps(result)`` (subject to the *values* being serializable,
    same as plain-dict tools). New code can use ``result.payload`` for the
    typed view and ``result.validate()`` for contract checking.

    The mapping content is the single source of truth; the envelope
    attributes (``success``, ``description``, …) are read-only views over it.
    """

    def __init__(
        self,
        success: bool,
        payload: Optional[Payload] = None,
        category: Optional[str] = None,
        description: str = "",
        error: Optional[str] = None,
        output_path: Optional[str] = None,
        vis_path: Optional[str] = None,
        overlay_path: Optional[str] = None,
        crop_paths: Optional[Sequence[str]] = None,
        **extras: Any,
    ):
        data: Dict[str, Any] = {}
        if payload is not None:
            # Shallow-copy list-valued payload fields so later mutation of the
            # mapping cannot corrupt the typed payload (and vice versa).
            for k, v in payload.to_fields().items():
                data[k] = list(v) if isinstance(v, list) else v
        data.update(extras)
        data["success"] = success
        data["description"] = description
        resolved_category = category or (payload.category if payload else None)
        if resolved_category:
            data["category"] = resolved_category
        if error is not None:
            data["error"] = error
        if output_path is not None:
            data["output_path"] = output_path
        if vis_path is not None:
            data["vis_path"] = vis_path
        if overlay_path is not None:
            data["overlay_path"] = overlay_path
        if crop_paths:
            data["crop_paths"] = list(crop_paths)
        super().__init__(data)
        self._payload = payload

    # -- read-only envelope views over the mapping ---------------------------
    @property
    def payload(self) -> Optional[Payload]:
        return self._payload

    @property
    def success(self) -> bool:
        return self.get("success", False)

    @property
    def category(self) -> Optional[str]:
        return self.get("category")

    @property
    def description(self) -> str:
        return self.get("description", "")

    @property
    def error(self) -> Optional[str]:
        return self.get("error")

    @property
    def output_path(self) -> Optional[str]:
        return self.get("output_path")

    @property
    def vis_path(self) -> Optional[str]:
        return self.get("vis_path")

    @property
    def overlay_path(self) -> Optional[str]:
        return self.get("overlay_path")

    @property
    def crop_paths(self) -> List[str]:
        return self.get("crop_paths", [])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"ToolResult(success={self.success}, "
                f"category={self.category!r}, keys={sorted(self)})")

    # -- convenience ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def validate(self) -> Tuple[bool, List[Tuple[str, ...]]]:
        """Validate the raw payload against this result's category contract."""
        if self.category is None:
            raise ValueError("result has no category to validate against")
        if not self.success:
            return True, []  # failures are envelope-only (error + description)
        return validate_payload(self, self.category)

    @classmethod
    def fail(cls, error: str, category: Optional[str] = None,
             description: str = "", **extras: Any) -> "ToolResult":
        return cls(success=False, error=error, category=category,
                   description=description or error, **extras)


def visualization_paths(result: Mapping) -> List[str]:
    """Existing visualization files, deduped, in envelope order.

    Consumes all four envelope keys, including ``overlay_path`` (which the
    legacy agent loop ignored). ``.mp4`` paths are returned as-is; the caller
    handles frame extraction.
    """
    seen: List[str] = []
    for key in ("output_path", "vis_path", "overlay_path"):
        p = result.get(key)
        if p and Path(p).exists() and p not in seen:
            seen.append(p)
    for p in result.get("crop_paths") or []:
        if p and Path(p).exists() and p not in seen:
            seen.append(p)
    return seen
