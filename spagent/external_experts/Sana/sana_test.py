#!/usr/bin/env python3
"""Reusable CLI test script for SanaClient."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spagent.external_experts.Sana.sana_client import SanaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reusable SanaClient test.")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:30000",
        help="Sana server URL (default: %(default)s)",
    )
    parser.add_argument("--model", default="default", help="Model name sent to API.")
    parser.add_argument(
        "--prompt",
        default="a small household robot organizing books in a warm study room",
        help="Text prompt for image generation.",
    )
    parser.add_argument("--size", default="1024x1024", help='Image size, e.g. "1024x1024".')
    parser.add_argument("--steps", type=int, default=20, help="Num inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=4.5, help="CFG guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n", type=int, default=1, help="Number of images.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sana_client",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print raw_response in JSON output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    client = SanaClient(
        server_url=args.server_url,
        model=args.model,
        timeout=args.timeout,
        output_dir=args.output_dir,
    )

    print(f"[INFO] module: {SanaClient.__module__}")
    print(f"[INFO] server: {args.server_url}")
    print(f"[INFO] output_dir: {Path(args.output_dir).resolve()}")
    print("[INFO] sending request...")

    start = time.time()
    result = client.generate_image(
        prompt=args.prompt,
        size=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        n=args.n,
        negative_prompt=args.negative_prompt,
    )
    elapsed = time.time() - start

    if result.get("success"):
        print(f"[OK] generation succeeded in {elapsed:.2f}s")
        print(f"[OK] output_path: {result.get('output_path')}")
        print(f"[OK] image_paths: {result.get('image_paths')}")
        print(f"[OK] file_size_bytes: {result.get('file_size_bytes')}")
    else:
        print(f"[ERROR] generation failed in {elapsed:.2f}s")
        print(f"[ERROR] detail: {result.get('error')}")

    safe_result = dict(result)
    if not args.raw:
        safe_result.pop("raw_response", None)

    print("[RESULT_JSON]")
    print(json.dumps(safe_result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
