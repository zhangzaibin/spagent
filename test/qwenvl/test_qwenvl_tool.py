"""
Tests for QwenVLTool — Qwen VL 2.5 detection.

Usage:
  # Mock test
  python test/qwenvl/test_qwenvl_tool.py --use_mock --image_path test/qwenvl/example.jpg

  # Real API test (requires DASHSCOPE_API_KEY)
  python test/qwenvl/test_qwenvl_tool.py --image_path path/to/image.jpg

  # Reasoning detection mode
  python test/qwenvl/test_qwenvl_tool.py --use_mock --image_path test/qwenvl/example.jpg --task reasoning_detection --prompt "What object can be used to sit on?"
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "spagent"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test QwenVLTool")
    parser.add_argument("--use_mock", action="store_true", default=False,
                        help="Use mock service instead of real API")
    parser.add_argument("--image_path", type=str, default="test/qwenvl/example.jpg",
                        help="Path to test image")
    parser.add_argument("--prompt", type=str, default="chair",
                        help="Detection prompt / query")
    parser.add_argument("--task", type=str, default="ref_detection",
                        choices=["ref_detection", "reasoning_detection"],
                        help="Detection task type")
    parser.add_argument("--model", type=str, default="qwen-vl-max-latest",
                        help="Qwen VL model name")
    args = parser.parse_args()

    from tools.qwenvl_tool import QwenVLTool

    tool = QwenVLTool(use_mock=args.use_mock, model=args.model)
    logger.info(f"QwenVLTool initialized (mock={args.use_mock})")

    logger.info(f"Image: {args.image_path}")
    logger.info(f"Task: {args.task} | Prompt: {args.prompt}")

    result = tool.call(
        image_path=args.image_path,
        text_prompt=args.prompt,
        task=args.task,
    )

    if result["success"]:
        boxes = result.get("boxes", [])
        labels = result.get("labels", [])
        logger.info(f"[PASS] Detected {len(boxes)} objects")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            logger.info(f"  [{i}] {label}: box={box}")
        if result.get("result", {}).get("mock"):
            logger.info("  (mock output)")
    else:
        logger.error(f"[FAIL] {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
