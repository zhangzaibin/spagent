"""
VACE Video Generation Tool

Wraps local VACE first-frame video generation service for the SPAgent system.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class VaceTool(Tool):
    """Tool for first-frame video generation using local VACE service."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://localhost:20034",
        mode: str = "inference",
    ):
        super().__init__(
            name="video_generation_vace_tool",
            description=(
                "Generate a video from one reference image and a text prompt via the local VACE first-frame "
                "pipeline; returns the path to the generated .mp4. "
                "Hard rule: output exactly one <tool_call> for this tool per assistant turn—never multiple "
                "tool_use / <tool_call> blocks for this tool in the same response. "
                "This tool is very slow and GPU-heavy (minutes per call); a second call in one turn is forbidden. "
                "When several frames or views exist, pick the single most critical image before that one call; "
                "do not sweep many frames. The prompt should describe the desired motion or outcome for that first frame. "
                "Use for controlled motion, camera change, or short temporal animation—not for bulk frame analysis."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        # Same keyword as Pi3Tool / MapAnythingTool for evaluate_img.py; not VACE pipeline --mode.
        self.tool_mode = mode
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            class _MockVaceService:
                def infer_firstframe(self, image_path: str, prompt: str, **kwargs):
                    return {
                        "success": True,
                        "output_path": "outputs/mock_vace_video.mp4",
                        "result_dir": "outputs",
                        "mock": True,
                        "prompt": prompt,
                        "image_path": image_path,
                    }

            self._client = _MockVaceService()
            logger.info("Using mock VACE service")
            return

        try:
            from external_experts.vace.vace_client import VaceClient

            self._client = VaceClient(server_url=self.server_url)
            logger.info(f"Using real VACE service at {self.server_url}")
        except ImportError as exc:
            logger.error(f"Failed to import VACE client: {exc}")
            raise

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": (
                        "Path to the first-frame reference image (one file). If several views or frames exist, "
                        "pick the most important frame and pass its path here."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Motion prompt describing how the generated video should move. What you want the video to do (motion, viewpoint change, etc.); the rollout follows this instruction."
                    ),
                },
                "base": {
                    "type": "string",
                    "description": "VACE base model backend. Default: 'wan'.",
                    "default": "wan",
                },
                "task": {
                    "type": "string",
                    "description": "VACE task name. Default: 'frameref'.",
                    "default": "frameref",
                },
                "mode": {
                    "type": "string",
                    "description": "VACE mode for first-frame generation. Default: 'firstframe'.",
                    "default": "firstframe",
                },
            },
            "required": ["image_path", "prompt"],
        }

    def call(
        self,
        image_path: str,
        prompt: str,
        base: str = "wan",
        task: str = "frameref",
        mode: str = "firstframe",
    ) -> Dict[str, Any]:
        try:
            if not prompt or not prompt.strip():
                return {"success": False, "error": "prompt is required for VACE video generation"}

            if not image_path:
                return {"success": False, "error": "image_path is required for VACE video generation"}

            # Relative paths must be absolute before HTTP: the server resolves non-absolute paths against
            # its VACE repo root (see vace_server.VaceRunner.run_firstframe), not the agent's cwd.
            resolved_image = os.path.abspath(os.path.expanduser(image_path))

            if not self.use_mock and not os.path.exists(resolved_image):
                return {"success": False, "error": f"Image file not found: {resolved_image}"}

            logger.info(f"Generating VACE firstframe video: {prompt[:80]}...")
            result = self._client.infer_firstframe(
                image_path=resolved_image,
                prompt=prompt,
                base=base,
                task=task,
                mode=mode,
            )

            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                    "description": "VACE firstframe video generated successfully.",
                }

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return {"success": False, "error": f"VACE generation failed: {error_msg}"}

        except Exception as exc:
            logger.error(f"VACE tool error: {exc}")
            return {"success": False, "error": str(exc)}
