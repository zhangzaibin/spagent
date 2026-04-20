"""
Evaluation script for the Wan (万相) video generation tool via SPAgent.

Dataset format (JSONL, one JSON object per line):
{
    "id": "unique_id",
    "prompt": "Text description of the video to generate",
    "image": [],               // optional list of reference image paths (for i2v)
    "task": "t2v",             // "t2v" (text-to-video) or "i2v" (image-to-video)
    "duration": 5,             // optional, 2-15 seconds (default 5)
    "size": "1280*720"         // optional, e.g. "1280*720" / "1920*1080" (default "1280*720")
}
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm
import argparse
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import WanTool
from spagent.utils.utils import load_json_data, save_result_to_csv, extract_question_and_answer

# ---------------------------------------------------------------------------
# Tool configurations
# ---------------------------------------------------------------------------

TOOL_CONFIGS = {
    "wan_real": [
        WanTool(use_mock=False),
    ],
    "wan_mock": [
        WanTool(use_mock=True),
    ],
}


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_single_wan_sample(
    agent: SPAgent,
    sample: Dict[str, Any],
    image_base_path: str,
    config_name: str = "wan",
    max_iterations: int = 3,
    video_num_frames: int = 4,
) -> Dict[str, Any]:
    """Evaluate a single Wan generation sample.

    Args:
        agent: SPAgent instance equipped with WanTool.
        sample: Data sample from the dataset.
        image_base_path: Base directory for resolving relative image paths.
        config_name: Configuration name used for CSV output naming.
        max_iterations: Maximum tool-call iterations allowed.
        video_num_frames: Number of frames to uniformly sample from a tool-generated video
                          and pass back to the model (default: 4).

    Returns:
        Result dictionary for this sample.
    """
    sample_id = sample.get("id", "unknown")
    task_type = sample.get("task", "t2v")

    # Support both formats:
    #   1. {"prompt": "..."}  — wan-native format
    #   2. {"conversations": [...]}  — standard VQA conversation format
    prompt = sample.get("prompt", "").strip()
    if not prompt:
        conversations = sample.get("conversations", [])
        if conversations:
            prompt, _ = extract_question_and_answer(conversations)
            prompt = (prompt or "").strip()

    if not prompt:
        return {
            "id": sample_id,
            "success": False,
            "error": "Empty prompt",
        }

    # Resolve optional reference images (image-to-video)
    raw_images = sample.get("image", [])
    image_paths: List[str] = []
    if raw_images:
        for img in raw_images:
            full = os.path.join(image_base_path, img) if not os.path.isabs(img) else img
            if not os.path.exists(full):
                return {
                    "id": sample_id,
                    "success": False,
                    "error": f"Reference image not found: {full}",
                }
            image_paths.append(full)

    # Build the agent task description
    if image_paths:
        agent_prompt = (
            f"Please generate a video based on the following description: {prompt}. "
            f"Use the provided reference image(s) for image-to-video generation."
        )
    else:
        agent_prompt = (
            f"Please generate a video based on the following description: {prompt}."
        )

    try:
        start_time = time.time()
        agent_result = agent.solve_problem(
            image_paths if image_paths else [],
            agent_prompt,
            max_iterations=max_iterations,
            video_num_frames=video_num_frames,
        )
        inference_time = time.time() - start_time

        # Extract Wan-specific information from tool calls
        wan_call_info = extract_wan_call_info(agent_result)

        # Determine overall generation success
        generation_succeeded = wan_call_info["called"] and wan_call_info["success"]

        task_data = {
            "id": sample_id,
            "prompt": prompt,
            "task": task_type,
            "wan_called": wan_call_info["called"],
            "generation_success": generation_succeeded,
            "output_path": wan_call_info.get("output_path", ""),
            "file_size_bytes": wan_call_info.get("file_size_bytes", 0),
            "inference_time": inference_time,
            "used_tools": agent_result.get("used_tools", []),
        }
        save_result_to_csv(task_data, csv_file=f"{config_name}.csv")

        return {
            "id": sample_id,
            "success": True,
            "prompt": prompt,
            "task": task_type,
            "wan_called": wan_call_info["called"],
            "generation_success": generation_succeeded,
            "output_path": wan_call_info.get("output_path", ""),
            "file_size_bytes": wan_call_info.get("file_size_bytes", 0),
            "wan_prompt_used": wan_call_info.get("wan_prompt", ""),
            "wan_duration": wan_call_info.get("duration", None),
            "wan_size": wan_call_info.get("size", None),
            "wan_error": wan_call_info.get("error", None),
            "inference_time": inference_time,
            "agent_answer": agent_result.get("answer", ""),
            "used_tools": agent_result.get("used_tools", []),
            "iterations": agent_result.get("iterations", 0),
            "tool_calls": agent_result.get("tool_calls", []),
        }

    except Exception as e:
        return {
            "id": sample_id,
            "success": False,
            "error": str(e),
        }


def extract_wan_call_info(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Wan tool call information from an agent result.

    Args:
        agent_result: Result dict returned by agent.solve_problem().

    Returns:
        Dictionary with keys: called, success, output_path, file_size_bytes,
        wan_prompt, duration, size, error.
    """
    info: Dict[str, Any] = {
        "called": False,
        "success": False,
        "output_path": "",
        "file_size_bytes": 0,
        "wan_prompt": "",
        "duration": None,
        "size": None,
        "error": None,
    }

    tool_calls = agent_result.get("tool_calls", [])
    for call in tool_calls:
        if call.get("name") == "video_generation_wan_tool":
            info["called"] = True
            args = call.get("arguments", {})
            info["wan_prompt"] = args.get("prompt", "")
            info["duration"] = args.get("duration")
            info["size"] = args.get("size")

            # The tool result is stored in the call's result field (if present)
            tool_result = call.get("result", {})
            if isinstance(tool_result, dict):
                if tool_result.get("success"):
                    info["success"] = True
                    info["output_path"] = tool_result.get("output_path", "")
                    nested = tool_result.get("result", {})
                    if isinstance(nested, dict):
                        info["file_size_bytes"] = nested.get("file_size_bytes", 0)
                    else:
                        info["file_size_bytes"] = tool_result.get("file_size_bytes", 0)
                else:
                    info["error"] = tool_result.get("error", "Unknown error")
            break  # Only inspect the first Wan call

    return info


# ---------------------------------------------------------------------------
# Config-level evaluation
# ---------------------------------------------------------------------------

def evaluate_wan_config(
    config_name: str,
    tools: List[Any],
    data_path: str,
    image_base_path: str,
    model: str = "gpt-4o",
    max_samples: int = None,
    max_workers: int = 1,
    max_iterations: int = 3,
    temperature: float = 0.0,
    seed: int = 42,
    top_p: float = 1.0,
    video_num_frames: int = 4,
) -> Dict[str, Any]:
    """Run a full Wan evaluation for a given tool configuration.

    Args:
        config_name: Identifier for this configuration (used in output filenames).
        tools: List of tool instances to equip the agent with.
        data_path: Path to the JSONL dataset file.
        image_base_path: Base directory for reference images.
        model: LLM model name passed to GPTModel.
        max_samples: Cap on the number of samples to evaluate (None = all).
        max_workers: Parallel worker threads for SPAgent.
        max_iterations: Maximum tool-call iterations per sample.
        temperature: Sampling temperature.
        seed: Random seed for reproducibility.
        top_p: Nucleus sampling probability mass.
        video_num_frames: Number of frames to uniformly sample from a tool-generated video
                          and feed back to the model (default: 4).

    Returns:
        Aggregated evaluation results dictionary.
    """
    print(f"\nEvaluating configuration: {config_name}")
    print(f"Loading data from: {data_path}")

    data = load_json_data(data_path)
    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples for evaluation")

    agent = SPAgent(
        model=GPTModel(model_name=model, temperature=temperature, seed=seed, top_p=top_p),
        tools=tools,
        max_workers=max_workers,
    )

    print(f"Evaluating {len(data)} samples with model={model}")

    results = []
    total_time = 0.0
    wan_called_count = 0
    generation_success_count = 0
    total_file_size = 0

    for sample in tqdm(data, desc="Evaluating"):
        result = evaluate_single_wan_sample(
            agent=agent,
            sample=sample,
            image_base_path=image_base_path,
            config_name=config_name,
            max_iterations=max_iterations,
            video_num_frames=video_num_frames,
        )
        results.append(result)

        if result.get("success"):
            total_time += result.get("inference_time", 0)
            if result.get("wan_called"):
                wan_called_count += 1
            if result.get("generation_success"):
                generation_success_count += 1
                total_file_size += result.get("file_size_bytes", 0)

    successful_results = [r for r in results if r.get("success")]
    failed_results = [r for r in results if not r.get("success")]

    n_success = len(successful_results)
    n_gen_success = generation_success_count
    tool_usage_rate = wan_called_count / n_success if n_success else 0.0
    generation_success_rate = n_gen_success / n_success if n_success else 0.0
    avg_inference_time = total_time / n_success if n_success else 0.0
    avg_file_size_mb = (total_file_size / n_gen_success / 1024 / 1024) if n_gen_success else 0.0

    # Per-task breakdown
    task_stats: Dict[str, Dict[str, Any]] = {}
    for r in successful_results:
        task = r.get("task", "unknown")
        if task not in task_stats:
            task_stats[task] = {"total": 0, "wan_called": 0, "generation_success": 0}
        task_stats[task]["total"] += 1
        if r.get("wan_called"):
            task_stats[task]["wan_called"] += 1
        if r.get("generation_success"):
            task_stats[task]["generation_success"] += 1

    for task in task_stats:
        t = task_stats[task]
        t["tool_usage_rate"] = t["wan_called"] / t["total"] if t["total"] else 0.0
        t["generation_success_rate"] = t["generation_success"] / t["total"] if t["total"] else 0.0

    return {
        "config_name": config_name,
        "model": model,
        "total_samples": len(data),
        "successful_samples": n_success,
        "failed_samples": len(failed_results),
        "wan_called_count": wan_called_count,
        "generation_success_count": n_gen_success,
        "tool_usage_rate": tool_usage_rate,
        "generation_success_rate": generation_success_rate,
        "average_inference_time_seconds": avg_inference_time,
        "average_file_size_mb": avg_file_size_mb,
        "task_statistics": task_stats,
        "failed_samples_details": failed_results,
        "detailed_results": results,
    }


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_wan_evaluation_results(results: Dict[str, Any]):
    """Print a formatted summary of Wan evaluation results."""
    print(f"\n{'='*70}")
    print(f"Wan Evaluation Results — {results['config_name']}")
    print(f"{'='*70}")
    print(f"Model:                    {results['model']}")
    print(f"Total samples:            {results['total_samples']}")
    print(f"Successfully processed:   {results['successful_samples']}")
    print(f"Failed to process:        {results['failed_samples']}")
    print(f"Wan tool called:          {results['wan_called_count']}  "
          f"(usage rate: {results['tool_usage_rate']:.1%})")
    print(f"Video generation success: {results['generation_success_count']}  "
          f"(success rate: {results['generation_success_rate']:.1%})")
    print(f"Avg inference time:       {results['average_inference_time_seconds']:.1f}s")
    print(f"Avg file size:            {results['average_file_size_mb']:.2f} MB")

    if results.get("task_statistics"):
        print(f"\n{'Task Breakdown':-<50}")
        print(f"{'Task':<15} {'Total':<8} {'Wan Called':<12} {'Gen Success':<14} "
              f"{'Usage Rate':<13} {'Success Rate'}")
        print("-" * 70)
        for task, stats in results["task_statistics"].items():
            print(
                f"{task:<15} {stats['total']:<8} {stats['wan_called']:<12} "
                f"{stats['generation_success']:<14} "
                f"{stats['tool_usage_rate']:<13.1%} "
                f"{stats['generation_success_rate']:.1%}"
            )

    if results.get("failed_samples_details"):
        print(f"\n{'Failed Samples':-<50}")
        for r in results["failed_samples_details"]:
            print(f"  ID={r.get('id', '?')}  error={r.get('error', 'unknown')}")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SPAgent with the Wan video generation tool."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/wan_eval_data.jsonl",
        help="Path to the evaluation JSONL dataset.",
    )
    parser.add_argument(
        "--image_base_path",
        type=str,
        default=".",
        help="Base directory for resolving relative image paths (default: project root '.').",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Parallel worker threads (default: 1; DashScope API has rate limits).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model name for the agent orchestrator (default: gpt-4o).",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum tool-call iterations per sample (default: 3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability mass (default: 1.0).",
    )
    parser.add_argument(
        "--video_num_frames",
        type=int,
        default=4,
        help="Number of frames to uniformly sample from a tool-generated video and pass back "
             "to the model in the next iteration (default: 4).",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock Wan service instead of the real DashScope API.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(TOOL_CONFIGS.keys()),
        help="Tool configuration to evaluate. Overrides --use_mock when set.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at {args.data_path}")
        return

    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        return

    # Select tool configuration
    if args.config:
        configs_to_run = {args.config: TOOL_CONFIGS[args.config]}
    else:
        key = "wan_mock" if args.use_mock else "wan_real"
        configs_to_run = {key: TOOL_CONFIGS[key]}

    all_results = {}
    for config_name, tools in configs_to_run.items():
        results = evaluate_wan_config(
            config_name=config_name,
            tools=tools,
            data_path=args.data_path,
            image_base_path=args.image_base_path,
            model=args.model,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            max_iterations=args.max_iterations,
            temperature=args.temperature,
            seed=args.seed,
            top_p=args.top_p,
            video_num_frames=args.video_num_frames,
        )
        all_results[config_name] = results
        print_wan_evaluation_results(results)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        f"wan_evaluation_results_{args.model.replace('-', '_')}_{timestamp}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        # Exclude raw tool_calls from the saved JSON to keep file size small
        for cfg_results in all_results.values():
            for r in cfg_results.get("detailed_results", []):
                r.pop("tool_calls", None)
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
