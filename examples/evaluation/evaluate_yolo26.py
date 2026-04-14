"""
Evaluation script for YOLO26Tool (local object detection).

Usage:
    python examples/evaluation/evaluate_yolo26.py \
        --data_path dataset/cvbench_data.jsonl \
        --image_base_path dataset \
        --model_path checkpoints/yolo26/yolo26n.pt \
        --device cpu \
        --model gpt-4o \
        --max_samples 50

Supported tool configurations (pass via --config):
    yolo26   — YOLO26 local object detection (default)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent import SPAgent
from spagent.core import DataCollector
from spagent.core.prompts import GENERAL_VISION_SYSTEM_PROMPT, GENERAL_VISION_CONTINUATION_HINT
from spagent.models import GPTModel, QwenModel
from spagent.tools import YOLO26Tool
from spagent.utils.utils import (
    load_json_data,
    extract_question_and_answer,
    normalize_answer,
    print_evaluation_results,
    validate_sample_paths,
    save_result_to_csv,
)
from spagent_evaluation import evaluate_tool_config, evaluate_single_sample


def build_tool_configs(model_path: str, device: str, conf: float, output_dir: str) -> Dict[str, List[Any]]:
    """Build YOLO26 tool configurations."""
    return {
        "yolo26": [
            YOLO26Tool(
                model_path=model_path,
                device=device,
                conf=conf,
                iou=0.45,
                max_det=100,
                save_annotated=True,
                output_dir=output_dir,
            )
        ]
    }


def print_yolo26_statistics(results: Dict[str, Any]) -> None:
    """
    Print YOLO26 detection statistics extracted from evaluation results.

    Reports class distribution, confidence statistics, and detections-per-image
    for a quick overview of what the model observed during the evaluation run.

    Args:
        results: Evaluation results dictionary returned by evaluate_tool_config.
    """
    if "yolo26_statistics" not in results:
        print("\nNo YOLO26 tool calls detected in this evaluation.")
        return

    stats = results["yolo26_statistics"]

    print(f"\n{'='*80}")
    print("YOLO26 Detection Statistics")
    print(f"{'='*80}")
    print(f"Total images processed:      {stats['total_images_processed']}")
    print(f"Total detections:            {stats['total_detections']}")
    print(f"Images with no detections:   {stats['images_with_no_detections']}")
    print(f"Avg detections / image:      {stats['avg_detections_per_image']:.2f}")
    print(f"Avg confidence (all dets):   {stats['avg_confidence']:.4f}")

    # Class distribution
    print(f"\n{'Top 10 Most Detected Classes:':<50}")
    print(f"{'Class':<25} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    for item in stats["top_10_classes"]:
        print(f"{item['class_name']:<25} {item['count']:<10} {item['percentage']:<15}")

    # Full class distribution
    if stats.get("class_distribution"):
        print(f"\n{'Full Class Distribution:':<50}")
        print(f"{'Class':<25} {'Count':<10}")
        print("-" * 80)
        for cls_name, count in stats["class_distribution"].items():
            print(f"{cls_name:<25} {count:<10}")

    print(f"{'='*80}\n")


def _collect_yolo26_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan per-sample tool call outputs and accumulate YOLO26 detection statistics.

    Args:
        results: Raw results dict from evaluate_tool_config.

    Returns:
        A statistics sub-dict ready to be merged into results.
    """
    total_images = 0
    total_detections = 0
    images_with_no_detections = 0
    all_confidences: List[float] = []
    class_counter: Counter = Counter()

    samples = results.get("samples", [])
    for sample in samples:
        for tool_call in sample.get("tool_calls", []):
            if tool_call.get("tool_name") != "yolo26_tool":
                continue
            tool_output = tool_call.get("output", {})
            if not isinstance(tool_output, dict) or not tool_output.get("success"):
                continue

            inner = tool_output.get("result", {})
            detections = inner.get("detections", [])
            num_det = inner.get("num_detections", len(detections))

            total_images += 1
            total_detections += num_det
            if num_det == 0:
                images_with_no_detections += 1

            for det in detections:
                cls_name = det.get("class_name", "unknown")
                confidence = det.get("confidence", 0.0)
                class_counter[cls_name] += 1
                all_confidences.append(float(confidence))

    if total_images == 0:
        return {}

    avg_detections = total_detections / total_images if total_images > 0 else 0.0
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    # Top-10 classes
    top_10 = class_counter.most_common(10)
    top_10_list = [
        {
            "class_name": cls,
            "count": count,
            "percentage": f"{100.0 * count / total_detections:.1f}%" if total_detections else "0.0%",
        }
        for cls, count in top_10
    ]

    return {
        "total_images_processed": total_images,
        "total_detections": total_detections,
        "images_with_no_detections": images_with_no_detections,
        "avg_detections_per_image": avg_detections,
        "avg_confidence": avg_confidence,
        "top_10_classes": top_10_list,
        "class_distribution": dict(class_counter.most_common()),
    }


def main() -> None:
    """Main entry point for YOLO26 evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate YOLO26Tool on a VQA benchmark dataset.")

    # Dataset / paths
    parser.add_argument("--data_path", type=str, default="dataset/cvbench_data.jsonl",
                        help="Path to the JSONL benchmark file (default: dataset/cvbench_data.jsonl)")
    parser.add_argument("--image_base_path", type=str, default="dataset",
                        help="Base directory for benchmark images (default: dataset)")

    # YOLO26 model settings
    parser.add_argument("--model_path", type=str, default="checkpoints/yolo26/yolo26n.pt",
                        help="Path to YOLO26 weights (default: checkpoints/yolo26/yolo26n.pt). "
                             "Can also be set via the YOLO26_MODEL_PATH environment variable.")
    parser.add_argument("--device", type=str, default=None,
                        help="Inference device: 'cpu' or 'cuda:0' (default: cpu). "
                             "Can also be set via the YOLO26_DEVICE environment variable.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO26 confidence threshold (default: 0.25)")
    parser.add_argument("--yolo_output_dir", type=str, default="outputs/yolo26",
                        help="Directory for annotated YOLO26 output images (default: outputs/yolo26)")

    # LLM / agent settings
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM to use for evaluation (default: gpt-4o)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate (default: all)")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of parallel evaluation workers (default: 4)")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Max tool-call iterations per sample (default: 3)")
    parser.add_argument("--task", type=str, default="all",
                        help="Task filter (default: all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM sampling temperature (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling probability mass (default: 1.0)")

    # Tool configuration selector
    parser.add_argument("--config", type=str, default="yolo26",
                        choices=["yolo26"],
                        help="Tool configuration to evaluate (default: yolo26)")

    # Data collection
    parser.add_argument("--enable_data_collection", action="store_true",
                        help="Enable training data collection")
    parser.add_argument("--data_output_dir", type=str, default=None,
                        help="Output directory for collected training data "
                             "(auto-generated timestamp directory if not specified)")

    args = parser.parse_args()

    # Environment variable overrides for YOLO26 settings
    model_path = os.environ.get("YOLO26_MODEL_PATH", args.model_path)
    device = os.environ.get("YOLO26_DEVICE", args.device or "cpu")

    # Validate required paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: YOLO26 weights not found at {model_path}")
        print("  Set --model_path or the YOLO26_MODEL_PATH environment variable.")
        return

    tool_configs = build_tool_configs(
        model_path=model_path,
        device=device,
        conf=args.conf,
        output_dir=args.yolo_output_dir,
    )

    # Only run the selected configuration
    selected_configs = {args.config: tool_configs[args.config]}

    all_results: Dict[str, Any] = {}
    for config_name, tools in selected_configs.items():
        # Optional data collection
        data_collector = None
        if args.enable_data_collection:
            if args.data_output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_output_dir = (
                    f"training_data/{config_name}_{args.model.replace('-', '_')}_{timestamp}"
                )
            else:
                data_output_dir = args.data_output_dir

            data_collector = DataCollector(
                output_dir=data_output_dir,
                save_images=True,
                auto_save=True,
            )
            print(f"Data collection enabled: {data_output_dir}")

        results = evaluate_tool_config(
            config_name=config_name,
            tools=tools,
            data_path=args.data_path,
            image_base_path=args.image_base_path,
            model=args.model,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            max_iterations=args.max_iterations,
            data_collector=data_collector,
            system_prompt=GENERAL_VISION_SYSTEM_PROMPT,
            continuation_hint=GENERAL_VISION_CONTINUATION_HINT,
            temperature=args.temperature,
            seed=args.seed,
            top_p=args.top_p,
        )

        # Attach YOLO26-specific statistics
        yolo26_stats = _collect_yolo26_stats(results)
        if yolo26_stats:
            results["yolo26_statistics"] = yolo26_stats

        all_results[config_name] = results

        print(f"\nResults for {config_name}:")
        print_evaluation_results(results)
        print_yolo26_statistics(results)

    # Persist results
    output_file = (
        f"spagent_evaluation_results_yolo26"
        f"_{args.model.replace('-', '_')}"
        f"_{args.max_iterations}"
        f"_{args.task}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {output_file}")


if __name__ == "__main__":
    main()
