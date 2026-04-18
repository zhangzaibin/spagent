"""
Sana Image Generation Tool

Wraps Sana image generation for the SPAgent system.
Supports text-to-image generation via an OpenAI-compatible SGLang server.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class SanaTool(Tool):
    """Tool for image generation using Sana."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://127.0.0.1:30000",
        model: str = "default",
        timeout: int = 300,
    ):
        super().__init__(
            name="image_generation_sana_tool",
            description=(
                "Generate an image from a text prompt using Sana. "
                "Use this when you need to visualize a hypothetical scene, target state, "
                "plan outcome, or imagined world state. The generated image is synthetic "
                "and should be treated as a visualization rather than direct evidence from "
                "the original observation."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.model_name = model
        self.timeout = timeout
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            try:
                from external_experts.Sana.mock_sana_service import MockSanaService

                self._client = MockSanaService()
                logger.info("Using mock Sana service")
            except ImportError as e:
                logger.error(f"Failed to import mock Sana service: {e}")
                raise
        else:
            try:
                from external_experts.Sana.sana_client import SanaClient

                self._client = SanaClient(
                    server_url=self.server_url,
                    model=self.model_name,
                    timeout=self.timeout,
                )
                logger.info("Using real Sana service at %s", self.server_url)
            except ImportError as e:
                logger.error(f"Failed to import Sana client: {e}")
                raise

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Text prompt describing the image to generate. "
                        "Be explicit about the scene, objects, layout, and desired appearance."
                    ),
                },
                "size": {
                    "type": "string",
                    "description": "Output image size. Default is '1024x1024'.",
                    "enum": ["512x512", "1024x1024"],
                    "default": "1024x1024",
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Number of diffusion inference steps. Default is 20.",
                    "default": 20,
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Classifier-free guidance scale. Default is 4.5.",
                    "default": 4.5,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility. Default is 42.",
                    "default": 42,
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Optional negative prompt describing what should be avoided.",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of images to generate. Default is 1.",
                    "default": 1,
                },
            },
            "required": ["prompt"],
        }

    def call(
        self,
        prompt: str,
        size: str = "1024x1024",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        seed: int = 42,
        negative_prompt: str = None,
        n: int = 1,
    ) -> Dict[str, Any]:
        try:
            logger.info("Generating image with Sana: %s...", prompt[:80])

            if not prompt or not prompt.strip():
                return {"success": False, "error": "Prompt must be a non-empty string."}

            result = self._client.generate_image(
                prompt=prompt,
                size=size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                negative_prompt=negative_prompt,
                n=n,
            )

            if result and result.get("success"):
                logger.info("Sana image generated: %s", result.get("output_path"))
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                    "image_paths": result.get("image_paths", []),
                    "file_size_bytes": result.get("file_size_bytes"),
                    "model": result.get("model"),
                    "size": result.get("size"),
                    "seed": result.get("seed"),
                }

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            logger.error("Sana generation failed: %s", error_msg)
            return {"success": False, "error": f"Sana generation failed: {error_msg}"}

        except Exception as e:
            logger.error("Sana tool error: %s", e)
            return {"success": False, "error": str(e)}
