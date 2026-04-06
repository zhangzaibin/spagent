"""
Tests for SoFarTool.

Usage:
  # Mock mode (no GPU required)
  python test/test_sofar_tool.py --use_mock \
      --image_path assets/table_scene.jpg \
      --instruction "pick up the blue bottle"

  # Real server (start sofar_server.py first)
  python test/test_sofar_tool.py \
      --image_path assets/table_scene.jpg \
      --instruction "grasp the mug by its handle"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from spagent.tools.sofar_tool import SoFarTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sofar(args):
    """Test SoFarTool with the given arguments."""
    logger.info(f"Instruction: '{args.instruction}', Mock: {args.use_mock}")

    tool = SoFarTool(
        use_mock=args.use_mock,
        server_url=f"http://{args.host}:{args.port}",
    )

    result = tool.call(
        image_path=args.image_path,
        instruction=args.instruction,
    )

    logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")

    if result.get("success"):
        logger.info("PASSED - tool returned success")
        data = result["result"]
        logger.info(f"  Bbox: {data.get('bbox')}")
        logger.info(f"  Position: {data.get('position')}")
        logger.info(f"  Quaternion: {data.get('quaternion')}")
        logger.info(f"  Approach vector: {data.get('approach_vector')}")
        logger.info(f"  Spatial desc: {data.get('spatial_description')}")
        logger.info(f"  Confidence: {data.get('confidence')}")
    else:
        logger.error(f"FAILED - {result.get('error')}")
        sys.exit(1)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock service (no GPU required)")
    parser.add_argument("--image_path", type=str, default="assets/table_scene.jpg")
    parser.add_argument("--instruction", type=str, default="pick up the blue bottle")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=20036)
    args = parser.parse_args()

    test_sofar(args)


if __name__ == "__main__":
    main()
