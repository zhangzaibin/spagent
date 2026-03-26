"""
Tests for OrientAnythingV2Tool.

Usage:
  # Mock mode (no GPU required)
  python test/test_orient_anything_v2_tool.py --use_mock --task orientation

  # Real server (start oa_v2_server.py first)
  python test/test_orient_anything_v2_tool.py --task orientation --image_path assets/chair.jpg

  # Relative rotation between two views
  python test/test_orient_anything_v2_tool.py --use_mock --task relative_rotation \
      --image_path assets/chair.jpg --image_path2 assets/chair_side.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from spagent.tools.orient_anything_v2_tool import OrientAnythingV2Tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_orient_anything_v2(args):
    """Test OrientAnythingV2Tool with the given arguments."""
    logger.info(f"Task: {args.task}, Mock: {args.use_mock}")

    tool = OrientAnythingV2Tool(
        use_mock=args.use_mock,
        server_url=f"http://{args.host}:{args.port}",
    )

    result = tool.call(
        image_path=args.image_path,
        object_category=args.object_category,
        task=args.task,
        image_path2=args.image_path2,
    )

    logger.info(f"Result: {result}")

    if result.get("success"):
        logger.info("PASSED - tool returned success")
        logger.info(f"Result data: {result['result']}")
    else:
        logger.error(f"FAILED - {result.get('error')}")
        sys.exit(1)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock service (no GPU required)")
    parser.add_argument("--task", choices=["orientation", "symmetry", "relative_rotation"],
                        default="orientation")
    parser.add_argument("--image_path", type=str, default="assets/dog.jpeg")
    parser.add_argument("--image_path2", type=str, default=None,
                        help="Required for relative_rotation task")
    parser.add_argument("--object_category", type=str, default="dog")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=20034)
    args = parser.parse_args()

    test_orient_anything_v2(args)


if __name__ == "__main__":
    main()
