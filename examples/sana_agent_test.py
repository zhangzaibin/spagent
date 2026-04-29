#!/usr/bin/env python3
"""
Minimal SPAgent + SanaTool test script.

Examples:
  python examples/sana_agent_test.py \
      --prompt "a small household robot organizing books in a warm study room"

  python examples/sana_agent_test.py \
      --model qwen3-vl-8b-thinking \
      --server-url http://127.0.0.1:30000 \
      --prompt "a service robot navigating a bright office hallway"

  python examples/sana_agent_test.py \
      --use-mock \
      --prompt "a warehouse robot moving a package toward a loading station"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spagent import SPAgent
from spagent.core.prompts import GENERATION_CONTINUATION_HINT, GENERATION_SYSTEM_PROMPT
from spagent.models import GPTModel
from spagent.tools import SanaTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal SPAgent + SanaTool generation test.")
    parser.add_argument(
        "--prompt",
        default="a small household robot organizing books in a warm study room",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-8b-thinking",
        help="LLM model name used by GPTModel.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:30000",
        help="Sana server URL (default: %(default)s).",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock Sana service instead of the real server.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum SPAgent tool-call iterations.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="LLM sampling seed.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="LLM nucleus sampling probability mass.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the full result JSON, including tool_calls/tool_results.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    model = GPTModel(
        model_name=args.model,
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
    )
    tools = [
        SanaTool(
            use_mock=args.use_mock,
            server_url=args.server_url,
        )
    ]

    agent = SPAgent(
        model=model,
        tools=tools,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        continuation_hint=GENERATION_CONTINUATION_HINT,
    )

    print(f"[INFO] model: {args.model}")
    print(f"[INFO] sana_mode: {'mock' if args.use_mock else 'real'}")
    if not args.use_mock:
        print(f"[INFO] sana_server: {args.server_url}")
    print(f"[INFO] prompt: {args.prompt}")
    print("[INFO] running agent...")

    start = time.time()
    result = agent.solve_problem(
        [],
        args.prompt,
        max_iterations=args.max_iterations,
    )
    elapsed = time.time() - start

    print(f"\n[OK] completed in {elapsed:.2f}s")
    print(f"[ANSWER] {result.get('answer', '')}")
    print(f"[USED_TOOLS] {result.get('used_tools', [])}")
    print(f"[ADDITIONAL_IMAGES] {result.get('additional_images', [])}")

    tool_results = result.get("tool_results", {})
    sana_result = None
    for key, value in tool_results.items():
        if isinstance(key, str) and key.startswith("image_generation_sana_tool"):
            sana_result = value
    if isinstance(sana_result, dict):
        print(f"[SANA_OUTPUT] {sana_result.get('output_path', '')}")
        print(f"[SANA_IMAGE_PATHS] {sana_result.get('image_paths', [])}")

    if args.show_raw:
        print("\n[RESULT_JSON]")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
