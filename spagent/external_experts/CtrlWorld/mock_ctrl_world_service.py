import logging
import os
import time

logger = logging.getLogger(__name__)


class MockCtrlWorldService:
    """Mock Ctrl-World service for local testing."""

    def generate_trajectory(
        self,
        image_path: str,
        actions: list,
        instruction: str = "",
        task_type: str = "pickplace",
    ) -> dict:
        logger.info(
            f"[Mock Ctrl-World] task={task_type}, "
            f"actions={len(actions) if actions else 0}"
        )

        if image_path and not image_path.startswith("http") and not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}

        if not isinstance(actions, list) or not actions:
            return {"success": False, "error": "actions must be a non-empty list"}

        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/mock_ctrl_world_{int(time.time())}.mp4"

        with open(output_path, "wb") as f:
            f.write(b"\x00" * 1024)

        return {
            "success": True,
            "output_path": output_path,
            "file_size_bytes": 1024,
            "mock": True,
            "task_type": task_type,
            "instruction": instruction,
        }
