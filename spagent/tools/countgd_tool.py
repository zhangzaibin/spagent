"""
CountGD Tool — text-prompted object counting from a single image.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class CountGDTool(Tool):
    """Text-prompted object counting using CountGD."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        confidence_thresh: float = 0.23,
        device: str = "cuda",
        server_url: Optional[str] = None,
        use_mock: bool = False,
    ):
        super().__init__(
            name="countgd_tool",
            description=(
                "CountGD: text-prompted object counting from a single image. "
                "Given a text description (e.g. 'car', 'person', 'apple'), counts all matching "
                "objects in the image and returns the count, bounding boxes, and an annotated "
                "visualization. Useful for crowd counting, inventory estimation, and object "
                "frequency analysis."
            ),
        )
        self.use_mock = use_mock
        self._server_url = server_url
        self._client = None
        self._client_kwargs = dict(
            checkpoint=checkpoint,
            confidence_thresh=confidence_thresh,
            device=device,
        )

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.use_mock:
            self._client = _MockCountGDClient()
        elif self._server_url:
            from external_experts.CountGD.countgd_client import CountGDClient
            self._client = CountGDClient(self._server_url)
        else:
            from external_experts.CountGD.countgd_local import CountGDLocalClient
            self._client = CountGDLocalClient(**self._client_kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image.",
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Text description of the object to count. "
                        "Examples: 'car', 'person', 'apple', 'chair'. "
                        "Be specific for best accuracy."
                    ),
                },
            },
            "required": ["image_path", "text"],
        }

    def call(
        self,
        image_path: Union[str, List[str]],
        text: str = "object",
    ) -> Dict[str, Any]:
        if isinstance(image_path, list):
            image_path = image_path[0]

        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        try:
            self._ensure_client()
            raw = self._client.count(image_path=image_path, text=text)
            if raw.get("success"):
                raw["result"] = {
                    "count": raw.get("count", 0),
                    "boxes": raw.get("boxes", []),
                }
            return raw
        except Exception as e:
            logger.exception("CountGDTool error")
            return {"success": False, "error": str(e)}


class _MockCountGDClient:
    def count(self, image_path: str, text: str = "object") -> Dict[str, Any]:
        boxes = [[50, 50, 200, 200], [220, 80, 380, 250], [400, 100, 560, 300]]
        return {
            "success": True,
            "result": {"count": 3, "boxes": boxes},
            "count": 3,
            "boxes": boxes,
            "output_path": image_path,
            "description": f"[mock] CountGD counted 3 '{text}' object(s) in the image.",
        }
