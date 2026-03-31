"""
Ctrl-World Tool

Wraps Ctrl-World trajectory/video generation for the SPAgent system.
Supports generating rollout videos from an initial image and action sequence.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)

SUPPORTED_TASK_TYPES = [
    "replay",
    "keyboard",
    "pickplace",
    "towel_fold",
    "wipe_table",
    "tissue",
    "close_laptop",
    "stack",
    "drawer",
]


class CtrlWorldTool(Tool):
    """Tool for controllable world-model rollout generation."""

    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:20040"):
        super().__init__(
            name="world_model_ctrl_world_tool",
            description=(
                "Generate a rollout video trajectory using Ctrl-World from an initial image, "
                "an action sequence, and optional task instruction. Use this when the task "
                "needs controllable world-model simulation for robot manipulation."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            try:
                from external_experts.CtrlWorld.mock_ctrl_world_service import MockCtrlWorldService

                self._client = MockCtrlWorldService()
                logger.info("Using mock Ctrl-World service")
            except ImportError as e:
                logger.error(f"Failed to import mock Ctrl-World service: {e}")
                raise
        else:
            try:
                from external_experts.CtrlWorld.ctrl_world_client import CtrlWorldClient

                self._client = CtrlWorldClient(server_url=self.server_url)
                logger.info("Using real Ctrl-World service (local server)")
            except ImportError as e:
                logger.error(f"Failed to import Ctrl-World client: {e}")
                raise

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the initial observation image.",
                },
                "actions": {
                    "type": "array",
                    "description": (
                        "Action sequence for rollout. Each action is a 7D list: "
                        "[x, y, z, rx, ry, rz, gripper]."
                    ),
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 7,
                        "maxItems": 7,
                    },
                },
                "instruction": {
                    "type": "string",
                    "description": "Optional natural-language task instruction.",
                    "default": "",
                },
                "task_type": {
                    "type": "string",
                    "description": "Ctrl-World task type.",
                    "enum": SUPPORTED_TASK_TYPES,
                    "default": "pickplace",
                },
            },
            "required": ["image_path", "actions"],
        }

    def call(
        self,
        image_path: str,
        actions: List[List[float]],
        instruction: str = "",
        task_type: str = "pickplace",
    ) -> Dict[str, Any]:
        try:
            logger.info(
                f"Generating Ctrl-World rollout: task={task_type}, "
                f"actions={len(actions) if isinstance(actions, list) else 0}"
            )

            if not image_path.startswith("http") and not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            if task_type not in SUPPORTED_TASK_TYPES:
                return {
                    "success": False,
                    "error": (
                        f"Unknown task_type: {task_type}. "
                        f"Supported task types: {SUPPORTED_TASK_TYPES}"
                    ),
                }

            validation_error = self._validate_actions(actions)
            if validation_error:
                return {"success": False, "error": validation_error}

            result = self._client.generate_trajectory(
                image_path=image_path,
                actions=actions,
                instruction=instruction,
                task_type=task_type,
            )

            if result and result.get("success"):
                logger.info(f"Ctrl-World rollout generated: {result.get('output_path')}")
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                }

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            logger.error(f"Ctrl-World generation failed: {error_msg}")
            return {"success": False, "error": f"Ctrl-World generation failed: {error_msg}"}
        except Exception as e:
            logger.error(f"Ctrl-World tool error: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _validate_actions(actions: Any) -> str:
        if not isinstance(actions, list) or not actions:
            return "actions must be a non-empty list"

        for idx, action in enumerate(actions):
            if not isinstance(action, list):
                return f"actions[{idx}] must be a list of 7 numbers"
            if len(action) != 7:
                return f"actions[{idx}] must contain exactly 7 values"
            for value in action:
                if not isinstance(value, (int, float)):
                    return f"actions[{idx}] must contain numeric values only"
        return ""
