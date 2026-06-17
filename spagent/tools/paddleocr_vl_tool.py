"""
PaddleOCR-VL-1.5 Tool

Wraps PaddleOCR-VL-1.5 (0.9 B VLM) for document-level OCR and
structured recognition tasks: plain OCR, table, chart, formula,
text spotting, and seal recognition.

Model: PaddlePaddle/PaddleOCR-VL-1.5  (Apache 2.0)
Paper: https://arxiv.org/abs/2505.09816
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)

VALID_TASKS = ("ocr", "table", "chart", "formula", "spotting", "seal")


class PaddleOCRVLTool(Tool):
    """Document OCR and structured recognition using PaddleOCR-VL-1.5."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        server_url: Optional[str] = None,
        use_mock: bool = False,
    ):
        """
        Args:
            checkpoint: HuggingFace model ID or local path. Reads
                        $PADDLEOCR_VL_CHECKPOINT if not given.
                        Ignored when server_url is set or use_mock is True.
            device: Torch device for local inference ('cuda' or 'cpu').
            server_url: If provided, forward calls to the Flask server at
                        this URL (e.g. 'http://0.0.0.0:20037') instead of
                        loading the model locally.
            use_mock: Return fixed mock output without loading any model.
        """
        super().__init__(
            name="paddleocr_vl_tool",
            description=(
                "PaddleOCR-VL-1.5: document-level OCR and structured recognition. "
                "Supports six task modes — "
                "'ocr' (plain text extraction), "
                "'table' (table structure + content), "
                "'chart' (chart data extraction), "
                "'formula' (LaTeX formula recognition), "
                "'spotting' (text region detection with transcription), "
                "'seal' (circular/elliptical seal recognition). "
                "Returns extracted text or structured output as a string. "
                "Best for documents, scanned images, receipts, slides, and academic papers."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client_kwargs = dict(checkpoint=checkpoint, device=device)
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.use_mock:
            self._client = _MockPaddleOCRVLClient()
        elif self.server_url:
            from external_experts.PaddleOCRVL.paddleocr_vl_client import PaddleOCRVLClient
            self._client = PaddleOCRVLClient(server_url=self.server_url)
        else:
            from external_experts.PaddleOCRVL.paddleocr_vl_local import PaddleOCRVLLocalClient
            self._client = PaddleOCRVLLocalClient(**self._client_kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image (document page, screenshot, etc.).",
                },
                "task": {
                    "type": "string",
                    "enum": list(VALID_TASKS),
                    "description": (
                        "Recognition mode: "
                        "'ocr' — extract all text (default); "
                        "'table' — parse table structure and cell content; "
                        "'chart' — read chart title, axes, and data values; "
                        "'formula' — transcribe mathematical formula to LaTeX; "
                        "'spotting' — detect text regions and transcribe each one; "
                        "'seal' — read circular or elliptical seal text."
                    ),
                    "default": "ocr",
                },
            },
            "required": ["image_path"],
        }

    def call(self, image_path: str, task: str = "ocr") -> Dict[str, Any]:
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        if task not in VALID_TASKS:
            return {
                "success": False,
                "error": f"Unknown task '{task}'. Valid tasks: {list(VALID_TASKS)}",
            }

        try:
            self._ensure_client()
            raw = self._client.recognize(image_path=image_path, task=task)
            if raw.get("success"):
                raw["result"] = {"text": raw.get("text", ""), "task": task}
            return raw
        except Exception as exc:
            logger.exception("PaddleOCRVLTool error")
            return {"success": False, "error": str(exc)}


class _MockPaddleOCRVLClient:
    def recognize(self, image_path: str, task: str = "ocr", **kwargs) -> Dict[str, Any]:
        mock_outputs = {
            "ocr": "Hello World\nThis is a sample OCR result.",
            "table": "| Column A | Column B |\n|----------|----------|\n| Cell 1   | Cell 2   |",
            "chart": "Title: Sales 2024\nX-axis: Month\nY-axis: Revenue ($)\nData: Jan=100, Feb=120",
            "formula": r"E = mc^2",
            "spotting": "[Region 1] Hello\n[Region 2] World",
            "seal": "Official Seal Text",
        }
        text = mock_outputs.get(task, "[mock OCR output]")
        return {
            "success": True,
            "text": text,
            "task": task,
            "result": {"text": text, "task": task},
        }
