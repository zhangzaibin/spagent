"""
Tests for CtrlWorldTool - controllable world-model rollout generation.

Usage:
  # Mock test
  python test/ctrl_world/test_ctrl_world_tool.py --use_mock --image_path assets/example.png

  # Real server test
  python test/ctrl_world/test_ctrl_world_tool.py --image_path assets/example.png --server_url http://localhost:20040
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "spagent"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_default_actions(num_waypoints: int):
    # [x, y, z, rx, ry, rz, gripper] — length of list is caller-controlled (server uses its own pred_step etc.)
    base = [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.8]
    return [base[:] for _ in range(num_waypoints)]


def main():
    parser = argparse.ArgumentParser(description="Test CtrlWorldTool")
    parser.add_argument(
        "--use_mock",
        action="store_true",
        default=False,
        help="Use mock service instead of real Ctrl-World server",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="assets/example.png",
        help="Path to initial observation image",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="pick up the blue block and place in white plate",
        help="Task instruction",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="pickplace",
        choices=[
            "replay",
            "keyboard",
            "pickplace",
            "towel_fold",
            "wipe_table",
            "tissue",
            "close_laptop",
            "stack",
            "drawer",
        ],
        help="Ctrl-World task type",
    )
    parser.add_argument(
        "--num_actions",
        type=int,
        default=5,
        help="Number of 7D poses in the test action sequence",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:20040",
        help="Ctrl-World server URL (real mode only)",
    )
    args = parser.parse_args()

    from tools.ctrl_world_tool import CtrlWorldTool

    tool = CtrlWorldTool(use_mock=args.use_mock, server_url=args.server_url)
    logger.info(f"CtrlWorldTool initialized (mock={args.use_mock}, server={args.server_url})")

    actions = build_default_actions(args.num_actions)
    result = tool.call(
        image_path=args.image_path,
        actions=actions,
        instruction=args.instruction,
        task_type=args.task_type,
    )

    if result["success"]:
        logger.info(f"[PASS] Rollout generated: {result['output_path']}")
        if result.get("result", {}).get("mock"):
            logger.info("  (mock output)")
    else:
        logger.error(f"[FAIL] {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
