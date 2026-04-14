"""
Molmo2 Tool

SPAgent wrapper for Molmo2 visual reasoning and pointing.
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)

def _normalize_image_paths(image_path: Union[str, List[str]]) -> List[str]:
    if isinstance(image_path, str):
        return [image_path]
    if isinstance(image_path, (list, tuple)):
        return [str(path) for path in image_path]
    raise TypeError(f"Unsupported image_path type: {type(image_path)}")


class Molmo2Tool(Tool):
    """Tool for Molmo2 reasoning and point-driven grounding."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://localhost:20035",
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="molmo2_tool",
            description=(
                "Run Molmo2 multimodal reasoning on one or more images. "
                "Use this for visual question answering, captioning, multi-image comparison, "
                "or point grounding when you want Molmo2 to point to an object or region."
            )
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.output_dir = str(output_dir) if output_dir is not None else None
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            from external_experts.Molmo2.mock_molmo2_service import MockMolmo2Service

            self._client = MockMolmo2Service(output_dir=self.output_dir)
            logger.info("Using mock Molmo2 service")
        else:
            from external_experts.Molmo2.molmo2_client import Molmo2Client

            self._client = Molmo2Client(server_url=self.server_url, output_dir=self.output_dir)
            logger.info("Using real Molmo2 service at %s", self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": "Path to a single input image.",
                        },
                        {
                            "type": "array",
                            "description": "List of input image paths for multi-image reasoning.",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    ],
                    "description": "Path to one image or a list of image paths.",
                },
                "task": {
                    "type": "string",
                    "enum": ["qa", "caption", "point"],
                    "description": (
                        "qa: answer a visual question or compare images; "
                        "caption: describe the image(s); "
                        "point: point to an object or region and optionally save annotated image(s)."
                    ),
                    "default": "qa",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Instruction or question for Molmo2. Required for qa and point. "
                        "Optional for caption."
                    ),
                },
                "save_annotated": {
                    "type": "boolean",
                    "description": "When task=point, save annotated image(s) with the predicted points.",
                    "default": True,
                },
                "max_new_tokens": {
                    "type": "integer",
                    "description": "Maximum number of tokens to generate.",
                    "default": 256,
                    "minimum": 1,
                    "maximum": 2048,
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature for generation. Use 0 for deterministic outputs.",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
            },
            "required": ["image_path", "task"],
        }

    def _default_prompt(self, task: str, image_count: int) -> str:
        if task == "caption":
            if image_count == 1:
                return "Describe this image in detail."
            return "Describe these images and compare their main similarities and differences."
        raise ValueError(f"Prompt is required for task '{task}'.")

    def call(
        self,
        image_path: Union[str, List[str]],
        task: str = "qa",
        prompt: Optional[str] = None,
        save_annotated: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        try:
            image_paths = _normalize_image_paths(image_path)
            for path in image_paths:
                if not Path(path).exists():
                    return {"success": False, "error": f"Image file not found: {path}"}

            if task not in {"qa", "caption", "point"}:
                return {"success": False, "error": f"Unknown task: {task}"}

            effective_prompt = prompt.strip() if prompt and prompt.strip() else self._default_prompt(task, len(image_paths))
            logger.info("Running Molmo2 task=%s on %d image(s)", task, len(image_paths))

            result = self._client.infer(
                image_paths=image_paths,
                task=task,
                prompt=effective_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                save_annotated=save_annotated,
            )
            if not result or not result.get("success"):
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                return {"success": False, "error": f"Molmo2 inference failed: {error_msg}"}

            generated_text = result.get("generated_text", "").strip()
            result_payload: Dict[str, Any] = {
                "task": task,
                "prompt": effective_prompt,
                "image_paths": image_paths,
                "generated_text": generated_text,
            }

            if task == "point":
                result_payload["points_by_image"] = result.get("points_by_image", [])
                result_payload["num_points"] = result.get("num_points", 0)

            response: Dict[str, Any] = {
                "success": True,
                "result": result_payload,
                "response_text": generated_text,
                "output_path": result.get("output_path"),
                "output_paths": result.get("output_paths", []),
            }

            if task == "point":
                response["description"] = (
                    f"Molmo2 point grounding completed on {len(image_paths)} image(s), "
                    f"returning {response['result']['num_points']} point(s)."
                )
                response["summary"] = response["description"]
            else:
                response["summary"] = generated_text[:240]
                response["description"] = (
                    f"Molmo2 {task} completed on {len(image_paths)} image(s). "
                    f"Generated response: {generated_text[:180]}"
                )

            return response
        except Exception as e:
            logger.error("Molmo2Tool error: %s", e)
            return {"success": False, "error": str(e)}
