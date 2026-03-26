"""
Tests for D4RTTool.

Usage:
  # Mock mode (no GPU required)
  python test/test_d4rt_tool.py --use_mock --task full_4d --frame_dir assets/sample_video_frames/

  # Depth and camera only
  python test/test_d4rt_tool.py --use_mock --task depth_and_camera --frame_dir assets/sample_video_frames/

  # Tracking specific points
  python test/test_d4rt_tool.py --use_mock --task tracking \
      --frame_dir assets/sample_video_frames/ --query_points "100,200;300,400"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from spagent.tools.d4rt_tool import D4RTTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_query_points(s: str):
    """Parse 'x1,y1;x2,y2' into [[x1,y1],[x2,y2]]."""
    if not s:
        return None
    points = []
    for pair in s.split(";"):
        x, y = pair.strip().split(",")
        points.append([int(x.strip()), int(y.strip())])
    return points


def test_d4rt(args):
    """Test D4RTTool with the given arguments."""
    logger.info(f"Task: {args.task}, Mock: {args.use_mock}")

    tool = D4RTTool(
        use_mock=args.use_mock,
        server_url=f"http://{args.host}:{args.port}",
    )

    query_points = parse_query_points(args.query_points)

    result = tool.call(
        frame_dir=args.frame_dir,
        task=args.task,
        query_points=query_points,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )

    logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")

    if result.get("success"):
        logger.info("PASSED - tool returned success")
        data = result["result"]
        logger.info(f"  Frames processed: {data.get('num_frames')}")
        if "depth_paths" in data:
            logger.info(f"  Depth maps: {len(data['depth_paths'])} files")
        if "camera_poses" in data:
            logger.info(f"  Camera poses: {len(data['camera_poses'])} frames")
        if "trajectories" in data:
            logger.info(f"  Trajectories: {len(data['trajectories'])} points tracked")
    else:
        logger.error(f"FAILED - {result.get('error')}")
        sys.exit(1)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock service (no GPU required)")
    parser.add_argument("--task", choices=["depth_and_camera", "tracking", "full_4d"],
                        default="full_4d")
    parser.add_argument("--frame_dir", type=str, default="assets/sample_video_frames/")
    parser.add_argument("--query_points", type=str, default=None,
                        help="Query points as 'x1,y1;x2,y2'. Required for tracking task.")
    parser.add_argument("--output_dir", type=str, default="outputs/d4rt")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=20035)
    args = parser.parse_args()

    test_d4rt(args)


if __name__ == "__main__":
    main()
