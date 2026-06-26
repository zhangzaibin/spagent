"""
All-Tools SPAgent Example

Build an agent with every catalog tool and run a single step in mock mode
(no external servers or API keys required for most tools).

Usage:
    python examples/all_tools_agent.py
    python examples/all_tools_agent.py --image assets/dog.jpeg --question "What is in this image?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from spagent.core import SPAgent
from spagent.models import GPTModel
from spagent.tools.catalog import build_tools, list_catalog_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPAgent with all catalog tools")
    parser.add_argument(
        "--image",
        default="assets/dog.jpeg",
        help="Input image path (default: assets/dog.jpeg)",
    )
    parser.add_argument(
        "--question",
        default="Analyze this image. Use tools only if they would materially improve your answer.",
        help="Question/instruction for the agent",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Backbone model name",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        default=True,
        help="Use mock tool clients (default: True)",
    )
    parser.add_argument(
        "--real-services",
        action="store_true",
        help="Use real tool services instead of mock clients",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help=(
            "Comma-separated catalog keys or tool names to load "
            "(e.g. depth,segmentation,pi3x). Default: all catalog tools."
        ),
    )
    parser.add_argument(
        type=int,
        default=2,
        help="Maximum tool-call iterations",
    )
    args = parser.parse_args()

    use_mock = not args.real_services
    tool_keys = None
    if args.tools:
        tool_keys = [part.strip() for part in args.tools.split(",") if part.strip()]

    tools, skipped = build_tools(
        tool_keys=tool_keys,
        use_mock=use_mock,
        skip_unavailable=True,
    )
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    if tool_keys is None:
        print(f"Available catalog keys: {list_catalog_keys()}")
    if skipped:
        print(f"Skipped unavailable tools: {skipped}")

    agent = SPAgent(
        model=GPTModel(model_name=args.model, temperature=0.0),
        tools=tools,
        workflow_mode="all_tools",
        max_workers=4,
    )

    image_path = args.image
    if image_path and not Path(image_path).exists():
        print(f"[WARN] Image not found: {image_path}; running text-only step.")
        image_path = None

    result = agent.step(
        content=args.question,
        images=image_path,
        max_tool_iterations=args.max_iterations,
        max_images_in_context=6,
    )

    print("\n=== Answer ===")
    print(result.answer)
    print("\n=== Used tools ===")
    print(result.used_tools or "(none)")
    print("\n=== Workflow ===")
    print(result.prompts.get("workflow"))


if __name__ == "__main__":
    main()
