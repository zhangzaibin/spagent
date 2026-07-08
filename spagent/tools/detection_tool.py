"""
Object Detection Tools

Two GroundingDINO-based tools for different visual understanding tasks:

- ZoomObjectTool  (zoom_object_tool)
    Crops the detected region with context padding and returns close-up image(s).
    Use when you need to inspect fine-grained attributes: color, text, material,
    pattern, shape.

- LocalizeObjectTool  (localize_object_tool)
    Draws bounding boxes on the full image and returns it alongside a text
    summary of each detection's position.  Use when you need to understand
    WHERE objects are: counting, spatial layout, relative positioning.

ObjectDetectionTool is kept as a backward-compatible alias for ZoomObjectTool.
"""

import sys
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Adaptive threshold back-off schedule: (box_threshold, text_threshold)
_RETRY_THRESHOLD_SCHEDULE: List[Tuple[float, float]] = [
    (0.35, 0.25),
    (0.25, 0.20),
    (0.15, 0.12),
    (0.10, 0.08),
]

# Crop quality (used by ZoomObjectTool)
_CROP_CONTEXT_PAD = 0.35  # expand each side by 35% of the box dimension
_CROP_MIN_PX = 120         # minimum crop dimension (pixels) before resize
_CROP_TARGET_PX = 512      # target longest-side size after resize

# Annotation quality (used by LocalizeObjectTool)
_ANNO_TARGET_PX = 1024     # resize annotated full image to this longest side
_ANNO_BOX_COLOR = (0, 200, 50)   # BGR green for bounding boxes
_ANNO_TEXT_COLOR = (255, 255, 255)
_ANNO_BOX_THICKNESS = 3
_ANNO_FONT_SCALE = 0.7


# ---------------------------------------------------------------------------
# Shared parameter schema (both tools accept the same arguments)
# ---------------------------------------------------------------------------

_SHARED_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_path": {
            "type": "string",
            "description": "Path to the input image.",
        },
        "text_prompt": {
            "type": "string",
            "description": (
                "Object(s) to detect. Specify AT MOST 2 names separated by '.' "
                "— e.g. 'car' or 'car . truck'. "
                "Do NOT list more than 2 names; make multiple tool calls instead."
            ),
        },
        "box_threshold": {
            "type": "number",
            "description": "Confidence threshold for box detection.",
            "default": 0.35,
        },
        "text_threshold": {
            "type": "number",
            "description": "Confidence threshold for text matching.",
            "default": 0.25,
        },
    },
    "required": ["image_path", "text_prompt"],
}


# ---------------------------------------------------------------------------
# Base class — shared client, retry logic, and bbox helpers
# ---------------------------------------------------------------------------

class _BaseDetectionTool(Tool):
    """Shared GroundingDINO logic. Not instantiated directly."""

    def __init__(
        self,
        name: str,
        description: str,
        use_mock: bool = True,
        server_url: str = "http://10.8.131.51:30969",
        max_crops: int = 3,
    ):
        super().__init__(name=name, description=description)
        self.use_mock = use_mock
        self.server_url = server_url
        self.max_crops = max_crops
        self._client = None
        self._output_dir = Path("outputs")
        self._output_dir.mkdir(exist_ok=True)
        self._init_client()

    # ------------------------------------------------------------------
    # Client bootstrap
    # ------------------------------------------------------------------

    def _init_client(self):
        if self.use_mock:
            try:
                from external_experts.GroundingDINO.mock_gdino_service import MockGroundingDINOService
                self._client = MockGroundingDINOService()
                logger.info("Using mock GroundingDINO service")
            except ImportError:
                class _SimpleMock:
                    def infer_image(self, image_path, text_prompt, **kwargs):
                        return {
                            "success": True,
                            "boxes": [[100, 100, 200, 200]],
                            "labels": ["object"],
                            "confidence": [0.8],
                        }
                self._client = _SimpleMock()
                logger.info("Using simple mock GroundingDINO service")
        else:
            from external_experts.GroundingDINO.grounding_dino_client import GroundingDINOClient
            self._client = GroundingDINOClient(server_url=self.server_url)
            logger.info(f"Using real GroundingDINO service at {self.server_url}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> Dict[str, Any]:
        return _SHARED_PARAMETERS

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _normalize_detections(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        detections = result.get("detections")
        if detections:
            return detections
        boxes = result.get("boxes", [])
        labels = result.get("labels", [])
        return [
            {"id": i, "bbox": b, "label": labels[i] if i < len(labels) else "obj"}
            for i, b in enumerate(boxes)
        ]

    def _surface_boxes(
        self, raw: Dict[str, Any], detections: List[Dict[str, Any]]
    ) -> Tuple[List, List, List]:
        """Return parallel ``(boxes, labels, confidence)`` arrays for the result.

        The real GroundingDINO client returns coordinates only inside
        ``detections`` (each ``bbox`` is normalized ``cxcywh`` in ``[0, 1]``); it
        never populates top-level ``boxes``/``labels``/``confidence``. Reading
        ``raw.get("boxes")`` therefore yielded empty arrays on the real path —
        only the mock fallback filled them. Derive the arrays from
        ``detections`` when raw doesn't provide them, so both paths expose the
        boxes. ``boxes`` mirror ``detections[i]["bbox"]`` (normalized cxcywh).
        """
        if raw.get("boxes"):
            return (
                raw.get("boxes", []),
                raw.get("labels", []),
                raw.get("confidence", []),
            )
        boxes = [d.get("bbox") for d in detections]
        labels = [d.get("label", "obj") for d in detections]
        # Keep confidence index-aligned with boxes/labels (None when absent)
        # so consumers can zip the three arrays safely.
        confidence = [d.get("confidence") for d in detections]
        return boxes, labels, confidence

    def _bbox_to_pixel_xyxy(self, bbox, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
        """Convert normalized cxcywh → pixel xyxy (or passthrough if already pixel)."""
        a, b, c, d = bbox
        if max(abs(a), abs(b), abs(c), abs(d)) <= 2.0:
            cx, cy, w, h = a * img_w, b * img_h, c * img_w, d * img_h
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
        else:
            x1, y1, x2, y2 = a, b, c, d
        return (
            max(0, int(round(x1))),
            max(0, int(round(y1))),
            min(img_w, int(round(x2))),
            min(img_h, int(round(y2))),
        )

    def _call_client(self, image_path, text_prompt, box_threshold, text_threshold):
        kwargs = dict(
            image_path=image_path,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if hasattr(self._client, "infer_image"):
            return self._client.infer_image(**kwargs)
        elif hasattr(self._client, "infer"):
            return self._client.infer(**kwargs)
        else:
            return self._client.detect(**kwargs)

    # ------------------------------------------------------------------
    # Detection with adaptive threshold back-off
    # ------------------------------------------------------------------

    def _run_detection(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[Optional[Dict], List[Dict], float]:
        """
        Run detection with adaptive threshold back-off.

        Returns:
            (raw_result, detections, used_box_threshold)
            raw_result is None if nothing found after all retries.
        """
        schedule: List[Tuple[float, float]] = [(box_threshold, text_threshold)]
        for bt, tt in _RETRY_THRESHOLD_SCHEDULE:
            if bt < box_threshold:
                schedule.append((bt, tt))

        raw: Optional[Dict] = None
        used_bt = box_threshold

        for attempt, (bt, tt) in enumerate(schedule):
            if attempt > 0:
                logger.info(
                    f"No detections at box_threshold={schedule[attempt-1][0]:.2f}; "
                    f"retrying with box_threshold={bt:.2f}"
                )
            raw = self._call_client(image_path, text_prompt, bt, tt)
            used_bt = bt
            if raw and raw.get("success"):
                dets = self._normalize_detections(raw)
                if dets:
                    return raw, dets, used_bt
                raw = None  # success but 0 boxes — keep trying

        return None, [], used_bt

    # ------------------------------------------------------------------
    # Shared call() preamble (text_prompt validation, image existence check)
    # ------------------------------------------------------------------

    def _prepare_call(
        self, image_path: str, text_prompt: str
    ) -> Tuple[Optional[Dict], str]:
        """
        Validate inputs and normalise text_prompt.

        Returns:
            (early_error_dict | None, normalised_text_prompt)
        """
        sep = "," if "," in text_prompt else "."
        names = [n.strip() for n in text_prompt.split(sep) if n.strip()]
        if len(names) > 2:
            truncated = " . ".join(names[:2])
            logger.warning(f"text_prompt truncated to 2: {text_prompt!r} → {truncated!r}")
            text_prompt = truncated
        elif len(names) > 1:
            text_prompt = " . ".join(names)

        if not Path(image_path).exists():
            return (
                {
                    "success": False,
                    "error": f"Image file not found: {image_path}",
                    "detections": [],
                    "crop_paths": [],
                },
                text_prompt,
            )
        return None, text_prompt

    # ------------------------------------------------------------------
    # "Nothing found" response helper
    # ------------------------------------------------------------------

    def _no_detection_response(self, text_prompt: str, used_bt: float) -> Dict:
        msg = (
            f"Detection ran for '{text_prompt}' but no region passed the "
            f"confidence threshold (tried down to box_threshold={used_bt:.2f}). "
            f"This does NOT mean the object is absent — it may be too small, "
            f"partially occluded, or low-contrast. "
            f"Please examine the full image directly. "
            f"You may also retry with a synonym "
            f"(e.g. 'motorbike' for 'motorcycle', 'luggage' for 'suitcase')."
        )
        logger.warning(msg)
        return {
            "success": True,
            "detections": [],
            "crop_paths": [],
            "message": msg,
            "description": msg,
        }


# ---------------------------------------------------------------------------
# Tool 1 — ZoomObjectTool  (zoom_object_tool)
# ---------------------------------------------------------------------------

class ZoomObjectTool(_BaseDetectionTool):
    """
    Zoom into a specific object: detect it and return cropped close-up image(s).

    Best for: inspecting fine-grained attributes — color, texture, text, material,
    pattern, markings — of a named object.
    """

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://10.8.131.51:30969",
        max_crops: int = 3,
    ):
        super().__init__(
            name="zoom_object_tool",
            description=(
                "Zoom into a specific object by detecting it and returning a cropped "
                "close-up image. Use this tool when you need to closely examine an "
                "object's fine-grained attributes.\n\n"
                "WHEN TO USE (attribute inspection):\n"
                "- \"What COLOR is the X?\" → zoom into X to see its color clearly\n"
                "- \"What does the X say / show?\" → zoom in to read text or markings\n"
                "- \"What PATTERN / MATERIAL / SHAPE is the X?\" → zoom in for texture detail\n"
                "- Any question requiring magnified inspection of a specific object\n\n"
                "WHEN NOT TO USE:\n"
                "- WHERE/HOW MANY questions → use localize_object_tool instead\n"
                "- Mood/emotion, abstract scene-level concepts\n"
                "- When you need pixel masks (prefer segment_image_tool)\n\n"
                "Key feature: the top detected regions are cropped with surrounding "
                "context and returned as high-resolution close-up images for analysis.\n\n"
                "text_prompt rules:\n"
                "- Specify AT MOST 2 object names separated by '.' (e.g. 'scarf' or 'helmet . person')\n"
                "- Name the object you want to zoom into, not the whole scene\n"
                "- If nothing found: retry with a synonym "
                "(e.g. 'motorbike' for 'motorcycle', 'bag' for 'handbag')"
            ),
            use_mock=use_mock,
            server_url=server_url,
            max_crops=max_crops,
        )

    # ------------------------------------------------------------------
    # Crop helper
    # ------------------------------------------------------------------

    def _crop_detections(self, image_path: str, detections: List[Dict]) -> List[str]:
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python required for cropping")
            return []

        orig = cv2.imread(image_path)
        if orig is None:
            logger.error(f"Cannot read image for cropping: {image_path}")
            return []

        img_h, img_w = orig.shape[:2]
        stem = Path(image_path).stem
        self._output_dir.mkdir(exist_ok=True)

        top_dets = sorted(
            detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True
        )[: self.max_crops]

        crop_paths: List[str] = []
        for rank, det in enumerate(top_dets):
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = self._bbox_to_pixel_xyxy(bbox, img_h, img_w)
            if x2 <= x1 or y2 <= y1:
                continue

            # Context padding
            bw, bh = x2 - x1, y2 - y1
            pad_x = max(int(round(_CROP_CONTEXT_PAD * bw)), 10)
            pad_y = max(int(round(_CROP_CONTEXT_PAD * bh)), 10)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(img_w, x2 + pad_x)
            cy2 = min(img_h, y2 + pad_y)

            # Minimum window
            if cx2 - cx1 < _CROP_MIN_PX:
                extra = (_CROP_MIN_PX - (cx2 - cx1)) // 2
                cx1, cx2 = max(0, cx1 - extra), min(img_w, cx2 + extra)
            if cy2 - cy1 < _CROP_MIN_PX:
                extra = (_CROP_MIN_PX - (cy2 - cy1)) // 2
                cy1, cy2 = max(0, cy1 - extra), min(img_h, cy2 + extra)

            crop_img = orig[cy1:cy2, cx1:cx2]

            # Resize to target
            ch, cw = crop_img.shape[:2]
            max_side = max(ch, cw)
            if max_side != _CROP_TARGET_PX:
                scale = _CROP_TARGET_PX / max_side
                new_w = max(1, int(round(cw * scale)))
                new_h = max(1, int(round(ch * scale)))
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
                crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=interp)

            conf = det.get("confidence", 0.0)
            label = re.sub(r"[^\w\-]+", "_", str(det.get("label", "obj"))).strip("_") or "obj"
            crop_path = self._output_dir / f"crop_{rank}_{label}_{stem}.jpg"
            cv2.imwrite(str(crop_path), crop_img)
            crop_paths.append(str(crop_path))
            logger.info(
                f"  zoom crop[{rank}] {label} conf={conf:.3f} "
                f"bbox=({x1},{y1},{x2},{y2}) padded=({cx1},{cy1},{cx2},{cy2}) "
                f"→ {crop_img.shape[1]}×{crop_img.shape[0]}px"
            )

        return crop_paths

    # ------------------------------------------------------------------
    # call()
    # ------------------------------------------------------------------

    def call(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Detect object and return cropped close-up images for attribute inspection.
        """
        try:
            err, text_prompt = self._prepare_call(image_path, text_prompt)
            if err:
                return err

            logger.info(f"[zoom] {image_path!r} prompt={text_prompt!r}")
            raw, detections, used_bt = self._run_detection(
                image_path, text_prompt, box_threshold, text_threshold
            )

            if not detections:
                return self._no_detection_response(text_prompt, used_bt)

            crop_paths = self._crop_detections(image_path, detections)
            msg = (
                f"Zoomed into {len(detections)} '{text_prompt}' region(s) "
                f"(box_threshold={used_bt:.2f}). "
                f"{len(crop_paths)} close-up crop(s) provided for attribute inspection."
            )
            logger.info(msg)
            boxes, labels, confidence = self._surface_boxes(raw, detections)
            return {
                "success": True,
                "result": raw,
                "detections": detections,
                "boxes": boxes,
                "labels": labels,
                "confidence": confidence,
                "output_path": raw.get("output_path"),
                "vis_path": raw.get("vis_path"),
                "crop_paths": crop_paths,
                "message": msg,
                "description": msg,
            }

        except Exception as e:
            logger.error(f"zoom_object_tool error: {e}")
            return {"success": False, "error": str(e), "detections": [], "crop_paths": []}


# ---------------------------------------------------------------------------
# Tool 2 — LocalizeObjectTool  (localize_object_tool)
# ---------------------------------------------------------------------------

class LocalizeObjectTool(_BaseDetectionTool):
    """
    Locate objects in the scene: detect and draw bounding boxes on the full image.

    Best for: counting instances, understanding spatial layout, relative positioning,
    confirming an object's location before zooming in.
    """

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://10.8.131.51:30969",
        max_crops: int = 3,
    ):
        super().__init__(
            name="localize_object_tool",
            description=(
                "Locate objects in an image by detecting them and drawing bounding "
                "boxes on the full scene. Use this tool when you need to understand "
                "WHERE objects are or HOW MANY there are.\n\n"
                "WHEN TO USE (spatial / counting questions):\n"
                "- \"How many X are in the image?\" → count instances\n"
                "- \"Is X to the left/right/above/below Y?\" → understand layout\n"
                "- \"Which X is closest to Y?\" → relative positioning\n"
                "- \"Where is the X?\" → find its location in the scene\n"
                "- Understanding the overall arrangement of multiple objects\n\n"
                "WHEN NOT TO USE:\n"
                "- COLOR / TEXTURE / MATERIAL / TEXT questions → use zoom_object_tool\n"
                "- Mood/emotion, abstract scene-level concepts\n"
                "- When you need pixel masks (prefer segment_image_tool)\n\n"
                "Key feature: returns the full image with labeled bounding boxes, "
                "plus a text summary of each detection's normalized position "
                "(e.g. 'center at x=0.42, y=0.61, covering 18%×22% of image').\n\n"
                "text_prompt rules:\n"
                "- Specify AT MOST 2 object names separated by '.' (e.g. 'person' or 'car . truck')\n"
                "- If nothing found: retry with a synonym"
            ),
            use_mock=use_mock,
            server_url=server_url,
            max_crops=max_crops,
        )

    # ------------------------------------------------------------------
    # Annotate full image with bounding boxes
    # ------------------------------------------------------------------

    def _annotate_image(
        self, image_path: str, detections: List[Dict]
    ) -> Tuple[Optional[str], str]:
        """
        Draw colored bounding boxes + labels on the full image.

        Returns:
            (annotated_image_path | None, text_summary)
        """
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python required for annotation")
            return None, ""

        orig = cv2.imread(image_path)
        if orig is None:
            logger.error(f"Cannot read image for annotation: {image_path}")
            return None, ""

        img_h, img_w = orig.shape[:2]
        annotated = orig.copy()
        summary_lines: List[str] = []

        for i, det in enumerate(detections):
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = self._bbox_to_pixel_xyxy(bbox, img_h, img_w)
            conf = float(det.get("confidence", 0.0))
            label = str(det.get("label", "obj"))

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _ANNO_BOX_COLOR, _ANNO_BOX_THICKNESS)

            # Draw label background + text
            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, _ANNO_FONT_SCALE, 2
            )
            ty = max(y1 - 6, th + baseline)
            cv2.rectangle(
                annotated,
                (x1, ty - th - baseline),
                (x1 + tw + 4, ty + baseline),
                _ANNO_BOX_COLOR,
                -1,
            )
            cv2.putText(
                annotated, text, (x1 + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, _ANNO_FONT_SCALE, _ANNO_TEXT_COLOR, 2,
            )

            # Text summary: normalized position
            cx_norm = ((x1 + x2) / 2) / img_w
            cy_norm = ((y1 + y2) / 2) / img_h
            w_norm = (x2 - x1) / img_w
            h_norm = (y2 - y1) / img_h
            summary_lines.append(
                f"  [{i+1}] {label}  conf={conf:.2f}  "
                f"center=(x={cx_norm:.2f}, y={cy_norm:.2f})  "
                f"size={w_norm:.2f}×{h_norm:.2f} of image"
            )

        # Resize annotated image to target longest side
        ah, aw = annotated.shape[:2]
        max_side = max(ah, aw)
        if max_side > _ANNO_TARGET_PX:
            scale = _ANNO_TARGET_PX / max_side
            annotated = cv2.resize(
                annotated,
                (max(1, int(round(aw * scale))), max(1, int(round(ah * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        stem = Path(image_path).stem
        out_path = self._output_dir / f"localize_{stem}.jpg"
        self._output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), annotated)

        text_summary = "\n".join(summary_lines)
        return str(out_path), text_summary

    # ------------------------------------------------------------------
    # call()
    # ------------------------------------------------------------------

    def call(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Detect objects and return full annotated image + position summary.
        """
        try:
            err, text_prompt = self._prepare_call(image_path, text_prompt)
            if err:
                return err

            logger.info(f"[localize] {image_path!r} prompt={text_prompt!r}")
            raw, detections, used_bt = self._run_detection(
                image_path, text_prompt, box_threshold, text_threshold
            )

            if not detections:
                return self._no_detection_response(text_prompt, used_bt)

            anno_path, text_summary = self._annotate_image(image_path, detections)
            msg = (
                f"Located {len(detections)} '{text_prompt}' instance(s) "
                f"(box_threshold={used_bt:.2f}). "
                f"Annotated full image provided. Positions:\n{text_summary}"
            )
            logger.info(msg)
            boxes, labels, confidence = self._surface_boxes(raw, detections)
            return {
                "success": True,
                "result": raw,
                "detections": detections,
                "boxes": boxes,
                "labels": labels,
                "confidence": confidence,
                "output_path": anno_path,       # annotated full image
                "vis_path": anno_path,
                "crop_paths": [],               # no crops for localize mode
                "message": msg,
                "description": msg,
            }

        except Exception as e:
            logger.error(f"localize_object_tool error: {e}")
            return {"success": False, "error": str(e), "detections": [], "crop_paths": []}


# ---------------------------------------------------------------------------
# Backward-compatible shim
# ---------------------------------------------------------------------------

class ObjectDetectionTool(ZoomObjectTool):
    """Backward-compatible wrapper preserving the legacy ``detect_objects_tool``
    name and ``crop=`` constructor API used by existing call sites such as
    quick_eval.py, run_spagent_vlmeval.py, and test/test_tool.py.
    """

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://10.8.131.51:30969",
        crop: bool = True,
        max_crops: int = 3,
    ):
        super().__init__(
            use_mock=use_mock,
            server_url=server_url,
            max_crops=(max_crops if crop else 0),
        )
        self.name = "detect_objects_tool"
