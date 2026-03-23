"""
Qwen VL 2.5 Detection Tool

Wraps Qwen VL 2.5 for referring detection and reasoning detection
via DashScope API.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class QwenVLTool(Tool):
    """Tool for object detection using Qwen VL 2.5.

    Supports two modes:
    - ref_detection: detect objects matching a text description
    - reasoning_detection: detect objects based on a reasoning question
    """

    def __init__(self, use_mock: bool = True, api_key: str = None, model: str = "qwen-vl-max-latest"):
        super().__init__(
            name="qwenvl_detection_tool",
            description=(
                "Detect objects in an image using Qwen VL 2.5. Supports referring detection "
                "(locate objects matching a text description) and reasoning detection "
                "(detect objects relevant to a reasoning question). Returns bounding boxes "
                "with normalized coordinates [0,1]."
            ),
        )
        self.use_mock = use_mock
        self.api_key = api_key
        self.model_name = model
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            try:
                from external_experts.QwenVL.mock_qwenvl_service import MockQwenVLService
                self._client = MockQwenVLService()
                logger.info("Using mock QwenVL service")
            except ImportError as e:
                logger.error(f"Failed to import mock QwenVL service: {e}")
                raise
        else:
            try:
                from external_experts.QwenVL.qwenvl_client import QwenVLClient
                self._client = QwenVLClient(api_key=self.api_key, model=self.model_name)
                logger.info("Using real QwenVL service (DashScope API)")
            except ImportError as e:
                logger.error(f"Failed to import QwenVL client: {e}")
                raise

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path or URL of the input image.",
                },
                "text_prompt": {
                    "type": "string",
                    "description": (
                        "For ref_detection: object description to locate (e.g. 'red car'). "
                        "For reasoning_detection: a reasoning question about the scene."
                    ),
                },
                "task": {
                    "type": "string",
                    "enum": ["ref_detection", "reasoning_detection"],
                    "description": "Detection mode. Default is 'ref_detection'.",
                    "default": "ref_detection",
                },
            },
            "required": ["image_path", "text_prompt"],
        }

    def call(
        self,
        image_path: str,
        text_prompt: str,
        task: str = "ref_detection",
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Running QwenVL {task}: {text_prompt[:60]}")

            if not image_path.startswith("http") and not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            if task not in ("ref_detection", "reasoning_detection"):
                return {"success": False, "error": f"Unknown task: {task}. Use 'ref_detection' or 'reasoning_detection'."}

            result = self._client.detect(
                image_path=image_path,
                text_prompt=text_prompt,
                task=task,
            )

            if result and result.get("success"):
                logger.info(f"QwenVL detected {len(result.get('boxes', []))} objects")
                return {
                    "success": True,
                    "result": result,
                    "boxes": result.get("boxes", []),
                    "labels": result.get("labels", []),
                    "raw_response": result.get("raw_response", ""),
                }
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"QwenVL detection failed: {error_msg}")
                return {"success": False, "error": f"QwenVL detection failed: {error_msg}"}

        except Exception as e:
            logger.error(f"QwenVL tool error: {e}")
            return {"success": False, "error": str(e)}
