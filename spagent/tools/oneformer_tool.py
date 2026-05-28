"""
OneFormer Tool — universal image segmentation (semantic / instance / panoptic).
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class OneFormerTool(Tool):
    """Universal image segmentation using OneFormer (semantic, instance, panoptic)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        server_url: Optional[str] = None,
        use_mock: bool = False,
    ):
        super().__init__(
            name="oneformer_tool",
            description=(
                "OneFormer: universal image segmentation supporting three tasks — "
                "semantic (pixel-level class labels), instance (individual object masks), "
                "and panoptic (combined semantic + instance). Returns an annotated "
                "segmentation image and a list of detected segments with labels and scores."
            ),
        )
        self.use_mock = use_mock
        self._server_url = server_url
        self._client = None
        self._client_kwargs = dict(model_id=model_id, device=device)

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.use_mock:
            self._client = _MockOneFormerClient()
        elif self._server_url:
            from external_experts.OneFormer.oneformer_client import OneFormerClient
            self._client = OneFormerClient(self._server_url)
        else:
            from external_experts.OneFormer.oneformer_local import OneFormerLocalClient
            self._client = OneFormerLocalClient(**self._client_kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input RGB image.",
                },
                "task": {
                    "type": "string",
                    "enum": ["semantic", "instance", "panoptic"],
                    "description": (
                        "Segmentation task to run. "
                        "'semantic': label every pixel with a class. "
                        "'instance': detect and mask individual object instances. "
                        "'panoptic': combine semantic and instance segmentation. "
                        "Default: 'panoptic'."
                    ),
                },
            },
            "required": ["image_path"],
        }

    def call(
        self,
        image_path: Union[str, List[str]],
        task: str = "panoptic",
    ) -> Dict[str, Any]:
        if isinstance(image_path, list):
            image_path = image_path[0]

        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        if task not in ("semantic", "instance", "panoptic"):
            return {"success": False, "error": f"Invalid task '{task}'. Choose semantic/instance/panoptic."}

        try:
            self._ensure_client()
            raw = self._client.segment(image_path=image_path, task=task)
            if raw.get("success"):
                raw["result"] = {
                    "task": raw.get("task", task),
                    "segments": raw.get("segments", []),
                    "num_segments": raw.get("num_segments", 0),
                }
            return raw
        except Exception as e:
            logger.exception("OneFormerTool error")
            return {"success": False, "error": str(e)}


class _MockOneFormerClient:
    def segment(self, image_path: str, task: str = "panoptic", **kwargs) -> Dict[str, Any]:
        segments = [
            {"id": 1, "label": "wall", "score": 0.92},
            {"id": 2, "label": "floor", "score": 0.88},
            {"id": 3, "label": "chair", "score": 0.75},
        ]
        return {
            "success": True,
            "task": task,
            "result": {
                "task": task,
                "segments": segments,
                "num_segments": len(segments),
            },
            "segments": segments,
            "num_segments": len(segments),
            "output_path": image_path,
            "description": f"[mock] OneFormer {task} segmentation: 3 segments (wall, floor, chair).",
        }
