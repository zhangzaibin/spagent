"""
Mock GroundingDINO service — format-faithful to the real server.

The real server (grounding_dino_server.py) returns detections whose ``bbox``
is **normalized cxcywh in [0, 1]** (unpacked straight from
``groundingdino.util.inference.predict``), with NO top-level boxes/labels:

    {"success": True,
     "detections": [{"id": 0, "bbox": [cx, cy, w, h],
                     "confidence": 0.87, "label": "dog"}],
     "shape": [h, w]}

The previous inline ``_SimpleMock`` fallback emitted a DIFFERENT shape
(top-level pixel-xyxy ``boxes``, no ``detections``), so mock-mode tests
exercised a format the real path never produces. This mock exists so that
``use_mock=True`` behaves like the real service.
"""

from typing import Any, Dict, List

# One deterministic detection per prompt term (max 2, mirroring the tools'
# two-object prompt limit), spread horizontally so crops/annotations differ.
_SLOTS = [
    {"bbox": [0.35, 0.5, 0.4, 0.6], "confidence": 0.87},
    {"bbox": [0.75, 0.4, 0.3, 0.5], "confidence": 0.62},
]


class MockGroundingDINOService:
    """Drop-in mock exposing the real client's ``infer`` method/shape."""

    def infer(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        sep = "," if "," in text_prompt else "."
        names = [n.strip() for n in text_prompt.split(sep) if n.strip()][:2]
        if not names:
            names = ["object"]
        detections: List[Dict[str, Any]] = [
            {"id": i, "bbox": list(_SLOTS[i]["bbox"]),
             "confidence": _SLOTS[i]["confidence"], "label": name}
            for i, name in enumerate(names)
        ]
        return {
            "success": True,
            "detections": detections,
            "annotated_image": None,
            "output_path": None,
            "shape": None,
        }
