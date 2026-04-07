"""
Moondream3 Tool

This module contains the Moondream3Tool that wraps
Moondream 3 (vision-language model) functionality for the SPAgent system.
Moondream 3 supports image captioning, VQA, object detection, pointing, and OCR.

Supports:
- Moondream Station (local): http://localhost:2020/v1 -> POST /v1/query
- Project md_server (Moondream): http://localhost:20024 -> POST /infer
"""

import base64
import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


def _is_station_url(server_url: str) -> bool:
    """True if URL is Moondream Station (e.g. .../v1 or :2020)."""
    return "/v1" in server_url.rstrip("/") or ":2020" in server_url


class _MoondreamStationClient:
    """Client for Moondream Station API: POST {base}/query with image_url (base64) + question."""

    def __init__(self, server_url: str, request_timeout: int = 300):
        self.base = server_url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base = f"{self.base}/v1"
        self._session = None
        # VLM inference can be slow; use long client timeout. If you get "Request timeout",
        # that is from the *server* (Moondream Station) — try `settings` in Station CLI to raise server timeout.
        self.request_timeout = request_timeout

    # def query(self, image_path: str, question: str) -> Dict[str, Any]:
    #     import requests
    #     try:
    #         with open(image_path, "rb") as f:
    #             raw = f.read()
    #         b64 = base64.b64encode(raw).decode("utf-8")
    #         ext = Path(image_path).suffix.lower()
    #         mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/jpeg"
    #         image_url = f"data:{mime};base64,{b64}"
    #         resp = requests.post(
    #             f"{self.base}/query",
    #             json={"image_url": image_url, "question": question},
    #             headers={"Content-Type": "application/json"},
    #             timeout=120,
    #         )
    #         resp.raise_for_status()
    #         data = resp.json()
    #         answer = data.get("answer") or data.get("response") or data.get("text") or ""
    #         if isinstance(answer, dict):
    #             answer = answer.get("answer", answer.get("content", ""))
    #         return {"success": True, "answer": str(answer) if answer else ""}
    #     except Exception as e:
    #         logger.error(f"Moondream Station query failed: {e}")
    #         return {"success": False, "error": str(e)}
    def query(self, image_path: str, question: str) -> Dict[str, Any]:
        import requests
        try:
            with open(image_path, "rb") as f:
                raw = f.read()

            b64 = base64.b64encode(raw).decode("utf-8")
            ext = Path(image_path).suffix.lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/jpeg"
            image_url = f"data:{mime};base64,{b64}"

            payload = {
                "image_url": image_url,
                "question": question,
                "stream": False
            }

            resp = requests.post(
                f"{self.base}/query",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )

            resp.raise_for_status()

            data = resp.json()

            # Server-side error (e.g. timeout)
            if data.get("status") == "timeout" or data.get("error"):
                return {"success": False, "error": data.get("error", "Request timeout")}

            answer = (
                data.get("answer")
                or data.get("response")
                or data.get("text")
                or (data.get("result", {}) if isinstance(data.get("result"), dict) else {}).get("answer")
                or ""
            )

            return {"success": True, "answer": str(answer) if answer else ""}
        except Exception as e:
            logger.error(f"Moondream Station query failed: {e}")
            return {"success": False, "error": str(e)}

class Moondream3Tool(Tool):
    """Tool for vision-language tasks using Moondream 3 (frontier-level VLM with pointing, captioning, VQA, etc.)"""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://localhost:20025",
        request_timeout: int = 300,
    ):
        """
        Initialize Moondream3 tool.

        Args:
            use_mock: Whether to use mock client for testing.
            server_url: URL of the Moondream3 server (default port 20025 to avoid conflict with Moondream).
            request_timeout: Seconds to wait for the server (default 300). If you see "Request timeout",
                it is usually the *server* (e.g. Moondream Station) timing out — increase timeout in Station
                via `settings` CLI if available.
        """
        super().__init__(
            name="moondream3_tool",
            description="Ask Moondream 3 (vision-language model) a question about an image. Input an image path and a natural language question; returns the model's answer based on image content. Use for visual question answering (VQA), image understanding, and visual reasoning."
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.request_timeout = request_timeout
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Moondream3 client (mock or real)."""
        if self.use_mock:
            self._client = _SimpleMockMoondream3()
            logger.info("Using mock Moondream3 service")
        elif _is_station_url(self.server_url):
            self._client = _MoondreamStationClient(
                server_url=self.server_url,
                request_timeout=self.request_timeout,
            )
            logger.info(f"Using Moondream Station at {self.server_url}")
        else:
            try:
                from external_experts.moondream3.md3_client import Moondream3Client
                self._client = Moondream3Client(server_url=self.server_url)
                logger.info(f"Using real Moondream3 service at {self.server_url}")
            except ImportError:
                try:
                    from external_experts.moondream.md_client import MoondreamClient
                    self._client = MoondreamClient(server_url=self.server_url)
                    logger.info(f"Using Moondream client at {self.server_url}")
                except ImportError as e:
                    logger.error(f"Failed to import Moondream3/Moondream client: {e}")
                    raise

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema (OpenAI function format)."""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image file."
                },
                "question": {
                    "type": "string",
                    "description": "Question about the image to ask Moondream3."
                }
            },
            "required": ["image_path", "question"]
        }

    def call(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        Ask Moondream3 a question about the image.

        Args:
            image_path: Path to input image.
            question: Natural language question about the image.

        Returns:
            Dict with success, result (answer), or error.
        """
        try:
            logger.info(f"Moondream3 VQA: image={image_path}, question={question!r}")

            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            if not (question and question.strip()):
                return {
                    "success": False,
                    "error": "Question is required."
                }

            result = self._client.query(image_path, question.strip())

            if result and result.get("success"):
                answer = result.get("answer", result.get("response", ""))
                logger.info("Moondream3 VQA completed successfully")
                return {
                    "success": True,
                    "result": {"answer": answer, **result},
                    "answer": answer,
                }
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"Moondream3 VQA failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                }
        except Exception as e:
            logger.error(f"Moondream3 tool error: {e}")
            return {"success": False, "error": str(e)}


class _SimpleMockMoondream3:
    """Inline mock for Moondream3 when use_mock=True or client not available."""

    def query(self, image_path: str, question: str) -> Dict[str, Any]:
        return {
            "success": True,
            "answer": f"Based on the image, the answer to \"{question}\" would depend on the visual content. (Mock response.)"
        }
