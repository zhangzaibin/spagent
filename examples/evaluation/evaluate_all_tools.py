"""
Evaluation script combining perception, 3D, video generation, and image generation tools.

Supported tool configurations (pass via --config):
  Vision / perception tools (standard VQA accuracy evaluation):
    dinosam   — GroundingDINO + SAM2
    pi3x      — Pi3X novel-view synthesis
    all_vision — dinosam + pi3x

  Video generation tools (generation-success evaluation):
    veo_real  — Veo (real Gemini API)
    veo_mock  — Veo (mock)

  Image generation tools (generation-success evaluation):
    sana_real — Sana (real SGLang server)
    sana_mock — Sana (mock)

Dataset format (JSONL):
  Vision tasks  → standard CVBench / VQA format with "conversations" field
  Video tasks   → {"id": ..., "prompt": ..., "task": "t2v"|"i2v", ...}
  Image tasks   → {"id": ..., "prompt": ..., "task": "t2i", ...}
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import argparse
from datetime import datetime
try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent import SPAgent
from spagent.core import DataCollector
from spagent.core.prompts import (
    GENERAL_VISION_SYSTEM_PROMPT,
    GENERAL_VISION_CONTINUATION_HINT,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_CONTINUATION_HINT,
)
from spagent.models import GPTModel, QwenModel
from spagent.tools import (
    SegmentationTool,
    ObjectDetectionTool,
    Pi3XTool,
    VeoTool,
    SanaTool,
)
from spagent.utils.utils import (
    load_json_data,
    extract_question_and_answer,
    normalize_answer,
    print_evaluation_results,
    validate_sample_paths,
    save_result_to_csv,
)
from spagent_evaluation import evaluate_tool_config, evaluate_single_sample

# ---------------------------------------------------------------------------
# Server URLs — edit these to match your deployment
# ---------------------------------------------------------------------------

TOOL_SERVERS = {
    "sam2":           "http://localhost:20020",
    "grounding_dino": "http://localhost:20022",
    "pi3x":           "http://localhost:20031",
    "sana":           "http://localhost:30000",
}

# ---------------------------------------------------------------------------
# Tool configurations
# ---------------------------------------------------------------------------

def build_tool_configs(configs_to_build: Optional[List[str]] = None) -> Dict[str, List[Any]]:
    """
    Build tool configurations.

    IMPORTANT: This function instantiates tool clients. To avoid requiring
    unrelated API keys (e.g. Veo) when evaluating a single config (e.g. Sana),
    you can pass `configs_to_build` to only construct the requested configs.
    """
    factories = {
        # --- vision / perception ---
        "dinosam": lambda: [
            ObjectDetectionTool(use_mock=False, server_url=TOOL_SERVERS["grounding_dino"]),
            SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["sam2"]),
        ],
        "pi3x": lambda: [
            Pi3XTool(use_mock=False, server_url=TOOL_SERVERS["pi3x"], mode="inference"),
        ],
        # --- video generation ---
        "veo_real": lambda: [VeoTool(use_mock=False)],
        "veo_mock": lambda: [VeoTool(use_mock=True)],
        # --- image generation ---
        "sana_real": lambda: [SanaTool(use_mock=False, server_url=TOOL_SERVERS["sana"])],
        "sana_mock": lambda: [SanaTool(use_mock=True)],
    }

    if configs_to_build is None:
        configs_to_build = sorted(factories.keys())

    tool_configs: Dict[str, List[Any]] = {}
    for name in configs_to_build:
        if name not in factories:
            raise KeyError(f"Unknown tool config: {name}")
        tool_configs[name] = factories[name]()
    return tool_configs


# Configs that use the standard vision VQA evaluation pipeline
VISION_CONFIGS = {"dinosam", "pi3x"}

# Configs that use the video generation evaluation pipeline
VIDEO_GEN_CONFIGS = {"veo_real", "veo_mock"}
IMAGE_GEN_CONFIGS = {"sana_real", "sana_mock"}

ALL_CONFIGS = VISION_CONFIGS | VIDEO_GEN_CONFIGS | IMAGE_GEN_CONFIGS

# Shorthand groups
CONFIG_GROUPS = {
    "all_vision": sorted(VISION_CONFIGS),
    "all_video":  sorted(VIDEO_GEN_CONFIGS),
    "all_image":  sorted(IMAGE_GEN_CONFIGS),
    "all_generation": sorted(VIDEO_GEN_CONFIGS | IMAGE_GEN_CONFIGS),
    "all":        sorted(ALL_CONFIGS),
}

# ---------------------------------------------------------------------------
# Video generation evaluation helpers (adapted from evaluate_veo / evaluate_sora)
# ---------------------------------------------------------------------------

def _extract_video_call_info(agent_result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """Extract video tool call information from an agent result."""
    info: Dict[str, Any] = {
        "called": False,
        "success": False,
        "output_path": "",
        "file_size_bytes": 0,
        "prompt_used": "",
        "duration": None,
        "aspect_ratio": None,
        "error": None,
    }

    for call in agent_result.get("tool_calls", []):
        if call.get("name") == tool_name:
            info["called"] = True
            args = call.get("arguments", {})
            info["prompt_used"] = args.get("prompt", "")
            info["duration"] = args.get("duration")
            info["aspect_ratio"] = args.get("aspect_ratio")

            tool_result = call.get("result", {})
            if isinstance(tool_result, dict):
                if tool_result.get("success"):
                    info["success"] = True
                    info["output_path"] = tool_result.get("output_path", "")
                    nested = tool_result.get("result", {})
                    info["file_size_bytes"] = (
                        nested.get("file_size_bytes", 0)
                        if isinstance(nested, dict)
                        else tool_result.get("file_size_bytes", 0)
                    )
                else:
                    info["error"] = tool_result.get("error", "Unknown error")
            break

    return info


# Maps config name → (internal tool call name, friendly label)
_VIDEO_TOOL_META = {
    "veo_real": ("video_generation_veo_tool", "Veo"),
    "veo_mock": ("video_generation_veo_tool", "Veo"),
}

_IMAGE_TOOL_META = {
    "sana_real": ("image_generation_sana_tool", "Sana"),
    "sana_mock": ("image_generation_sana_tool", "Sana"),
}


def _extract_image_call_info(agent_result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """Extract image-generation tool call information from an agent result."""
    info: Dict[str, Any] = {
        "called": False,
        "success": False,
        "output_path": "",
        "image_paths": [],
        "file_size_bytes": 0,
        "prompt_used": "",
        "size": None,
        "num_inference_steps": None,
        "guidance_scale": None,
        "seed": None,
        "negative_prompt": None,
        "error": None,
    }

    tool_results = agent_result.get("tool_results", {})
    if isinstance(tool_results, dict):
        matching_keys = [
            k for k in tool_results.keys()
            if isinstance(k, str) and k.startswith(f"{tool_name}_iter")
        ]
        if matching_keys:
            def _iter_num(key: str) -> int:
                try:
                    return int(key.split("_iter")[-1])
                except Exception:
                    return -1

            best_key = sorted(matching_keys, key=_iter_num)[-1]
            tr = tool_results.get(best_key, {})
            if isinstance(tr, dict):
                info["called"] = True
                if tr.get("success"):
                    info["success"] = True
                    info["output_path"] = tr.get("output_path", "") or ""
                    info["image_paths"] = tr.get("image_paths", []) or []
                    info["file_size_bytes"] = tr.get("file_size_bytes", 0) or 0
                else:
                    info["error"] = tr.get("error", "Unknown error")

    for call in agent_result.get("tool_calls", []):
        if call.get("name") == tool_name:
            info["called"] = True
            args = call.get("arguments", {})
            info["prompt_used"] = args.get("prompt", "")
            info["size"] = args.get("size")
            info["num_inference_steps"] = args.get("num_inference_steps")
            info["guidance_scale"] = args.get("guidance_scale")
            info["seed"] = args.get("seed")
            info["negative_prompt"] = args.get("negative_prompt")

            tool_result = call.get("result", {})
            if isinstance(tool_result, dict):
                if tool_result.get("success"):
                    info["success"] = True
                    info["output_path"] = tool_result.get("output_path", "")
                    info["image_paths"] = tool_result.get("image_paths", [])
                    nested = tool_result.get("result", {})
                    info["file_size_bytes"] = (
                        nested.get("file_size_bytes", 0)
                        if isinstance(nested, dict)
                        else tool_result.get("file_size_bytes", 0)
                    )
                else:
                    info["error"] = tool_result.get("error", "Unknown error")
            break

    return info


def evaluate_single_video_sample(
    agent: SPAgent,
    sample: Dict[str, Any],
    image_base_path: str,
    config_name: str,
    tool_call_name: str,
    max_iterations: int = 3,
    video_num_frames: int = 4,
) -> Dict[str, Any]:
    sample_id = sample.get("id", "unknown")
    task_type = sample.get("task", "t2v")

    prompt = sample.get("prompt", "").strip()
    if not prompt:
        conversations = sample.get("conversations", [])
        if conversations:
            prompt, _ = extract_question_and_answer(conversations)
            prompt = (prompt or "").strip()
    if not prompt:
        return {"id": sample_id, "success": False, "error": "Empty prompt"}

    raw_images = sample.get("image", [])
    image_paths: List[str] = []
    for img in raw_images:
        full = os.path.join(image_base_path, img) if not os.path.isabs(img) else img
        if not os.path.exists(full):
            return {"id": sample_id, "success": False, "error": f"Reference image not found: {full}"}
        image_paths.append(full)

    if image_paths:
        agent_prompt = (
            f"Please generate a video based on the following description: {prompt}. "
            f"Use the provided reference image(s) for image-to-video generation."
        )
    else:
        agent_prompt = f"Please generate a video based on the following description: {prompt}."

    try:
        start_time = time.time()
        agent_result = agent.solve_problem(
            image_paths if image_paths else [],
            agent_prompt,
            max_iterations=max_iterations,
            video_num_frames=video_num_frames,
        )
        inference_time = time.time() - start_time

        call_info = _extract_video_call_info(agent_result, tool_call_name)
        generation_succeeded = call_info["called"] and call_info["success"]

        task_data = {
            "id": sample_id,
            "prompt": prompt,
            "task": task_type,
            "tool_called": call_info["called"],
            "generation_success": generation_succeeded,
            "output_path": call_info.get("output_path", ""),
            "file_size_bytes": call_info.get("file_size_bytes", 0),
            "inference_time": inference_time,
            "used_tools": agent_result.get("used_tools", []),
        }
        save_result_to_csv(task_data, csv_file=f"{config_name}.csv")

        return {
            "id": sample_id,
            "success": True,
            "prompt": prompt,
            "task": task_type,
            "tool_called": call_info["called"],
            "generation_success": generation_succeeded,
            "output_path": call_info.get("output_path", ""),
            "file_size_bytes": call_info.get("file_size_bytes", 0),
            "prompt_used": call_info.get("prompt_used", ""),
            "duration": call_info.get("duration"),
            "aspect_ratio": call_info.get("aspect_ratio"),
            "tool_error": call_info.get("error"),
            "inference_time": inference_time,
            "agent_answer": agent_result.get("answer", ""),
            "used_tools": agent_result.get("used_tools", []),
            "iterations": agent_result.get("iterations", 0),
            "tool_calls": agent_result.get("tool_calls", []),
        }
    except Exception as e:
        return {"id": sample_id, "success": False, "error": str(e)}


def evaluate_video_config(
    config_name: str,
    tools: List[Any],
    data_path: str,
    image_base_path: str,
    model: str = "gpt-4o",
    max_samples: Optional[int] = None,
    max_workers: int = 1,
    max_iterations: int = 3,
    temperature: float = 0.0,
    seed: int = 42,
    top_p: float = 1.0,
    video_num_frames: int = 4,
) -> Dict[str, Any]:
    tool_call_name, label = _VIDEO_TOOL_META[config_name]
    print(f"\nEvaluating configuration: {config_name}  [{label}]")
    print(f"Loading data from: {data_path}")

    data = load_json_data(data_path)
    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples")

    agent = SPAgent(
        model=GPTModel(model_name=model, temperature=temperature, seed=seed, top_p=top_p),
        tools=tools,
        max_workers=max_workers,
    )
    print(f"Evaluating {len(data)} samples with model={model}")

    results, total_time = [], 0.0
    tool_called_count = gen_success_count = total_file_size = 0
    task_stats: Dict[str, Dict[str, Any]] = {}

    for sample in tqdm(data, desc="Evaluating"):
        result = evaluate_single_video_sample(
            agent=agent,
            sample=sample,
            image_base_path=image_base_path,
            config_name=config_name,
            tool_call_name=tool_call_name,
            max_iterations=max_iterations,
            video_num_frames=video_num_frames,
        )
        results.append(result)

        if result.get("success"):
            total_time += result.get("inference_time", 0)
            if result.get("tool_called"):
                tool_called_count += 1
            if result.get("generation_success"):
                gen_success_count += 1
                total_file_size += result.get("file_size_bytes", 0)

            task = result.get("task", "unknown")
            if task not in task_stats:
                task_stats[task] = {"total": 0, "tool_called": 0, "generation_success": 0}
            task_stats[task]["total"] += 1
            if result.get("tool_called"):
                task_stats[task]["tool_called"] += 1
            if result.get("generation_success"):
                task_stats[task]["generation_success"] += 1

    for t in task_stats.values():
        t["tool_usage_rate"] = t["tool_called"] / t["total"] if t["total"] else 0.0
        t["generation_success_rate"] = t["generation_success"] / t["total"] if t["total"] else 0.0

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    n_ok = len(successful)

    return {
        "config_name": config_name,
        "tool_label": label,
        "model": model,
        "total_samples": len(data),
        "successful_samples": n_ok,
        "failed_samples": len(failed),
        "tool_called_count": tool_called_count,
        "generation_success_count": gen_success_count,
        "tool_usage_rate": tool_called_count / n_ok if n_ok else 0.0,
        "generation_success_rate": gen_success_count / n_ok if n_ok else 0.0,
        "average_inference_time_seconds": total_time / n_ok if n_ok else 0.0,
        "average_file_size_mb": (total_file_size / gen_success_count / 1024 / 1024) if gen_success_count else 0.0,
        "task_statistics": task_stats,
        "failed_samples_details": failed,
        "detailed_results": results,
    }


def evaluate_single_image_sample(
    agent: SPAgent,
    sample: Dict[str, Any],
    image_base_path: str,
    config_name: str,
    tool_call_name: str,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    sample_id = sample.get("id", "unknown")
    task_type = sample.get("task", "t2i")
    prompt = sample.get("prompt", "").strip()

    if not prompt:
        conversations = sample.get("conversations", [])
        if conversations:
            prompt, _ = extract_question_and_answer(conversations)
            prompt = (prompt or "").strip()
    if not prompt:
        return {"id": sample_id, "success": False, "error": "Empty prompt"}

    raw_images = sample.get("image", [])
    image_paths: List[str] = []
    for img in raw_images:
        full = os.path.join(image_base_path, img) if not os.path.isabs(img) else img
        if not os.path.exists(full):
            return {"id": sample_id, "success": False, "error": f"Reference image not found: {full}"}
        image_paths.append(full)

    agent_prompt = (
        "Please generate an image based on the following description using the available image generation tool. "
        "Treat the generated image as a synthetic visualization rather than evidence from the original observation.\n\n"
        f"Prompt: {prompt}\n"
    )
    if sample.get("size"):
        agent_prompt += f"Preferred size: {sample['size']}\n"
    if sample.get("num_inference_steps") is not None:
        agent_prompt += f"Sampling steps: {sample['num_inference_steps']}\n"
    if sample.get("guidance_scale") is not None:
        agent_prompt += f"Guidance scale: {sample['guidance_scale']}\n"
    if sample.get("seed") is not None:
        agent_prompt += f"Seed: {sample['seed']}\n"
    if sample.get("negative_prompt"):
        agent_prompt += f"Negative prompt: {sample['negative_prompt']}\n"

    try:
        start_time = time.time()
        agent_result = agent.solve_problem(
            image_paths if image_paths else [],
            agent_prompt,
            max_iterations=max_iterations,
        )
        inference_time = time.time() - start_time

        call_info = _extract_image_call_info(agent_result, tool_call_name)
        generation_succeeded = call_info["called"] and call_info["success"]

        return {
            "id": sample_id,
            "success": True,
            "prompt": prompt,
            "task": task_type,
            "tool_called": call_info["called"],
            "generation_success": generation_succeeded,
            "output_path": call_info.get("output_path", ""),
            "image_paths": call_info.get("image_paths", []),
            "file_size_bytes": call_info.get("file_size_bytes", 0),
            "prompt_used": call_info.get("prompt_used", ""),
            "size": call_info.get("size"),
            "num_inference_steps": call_info.get("num_inference_steps"),
            "guidance_scale": call_info.get("guidance_scale"),
            "seed": call_info.get("seed"),
            "negative_prompt": call_info.get("negative_prompt"),
            "tool_error": call_info.get("error"),
            "inference_time": inference_time,
            "agent_answer": agent_result.get("answer", ""),
            "used_tools": agent_result.get("used_tools", []),
            "iterations": agent_result.get("iterations", 0),
            "tool_calls": agent_result.get("tool_calls", []),
        }
    except Exception as e:
        return {"id": sample_id, "success": False, "error": str(e)}


def evaluate_image_config(
    config_name: str,
    tools: List[Any],
    data_path: str,
    image_base_path: str,
    model: str = "gpt-4o",
    max_samples: Optional[int] = None,
    max_workers: int = 1,
    max_iterations: int = 3,
    temperature: float = 0.0,
    seed: int = 42,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    tool_call_name, label = _IMAGE_TOOL_META[config_name]
    print(f"\nEvaluating configuration: {config_name}  [{label}]")
    print(f"Loading data from: {data_path}")

    data = load_json_data(data_path)
    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples")

    agent = SPAgent(
        model=GPTModel(model_name=model, temperature=temperature, seed=seed, top_p=top_p),
        tools=tools,
        max_workers=max_workers,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        continuation_hint=GENERATION_CONTINUATION_HINT,
    )
    print(f"Evaluating {len(data)} samples with model={model}")

    results, total_time = [], 0.0
    tool_called_count = gen_success_count = total_file_size = 0
    task_stats: Dict[str, Dict[str, Any]] = {}

    for sample in tqdm(data, desc="Evaluating"):
        result = evaluate_single_image_sample(
            agent=agent,
            sample=sample,
            image_base_path=image_base_path,
            config_name=config_name,
            tool_call_name=tool_call_name,
            max_iterations=max_iterations,
        )
        results.append(result)

        if result.get("success"):
            total_time += result.get("inference_time", 0)
            if result.get("tool_called"):
                tool_called_count += 1
            if result.get("generation_success"):
                gen_success_count += 1
                total_file_size += result.get("file_size_bytes", 0)

            task = result.get("task", "unknown")
            if task not in task_stats:
                task_stats[task] = {"total": 0, "tool_called": 0, "generation_success": 0}
            task_stats[task]["total"] += 1
            if result.get("tool_called"):
                task_stats[task]["tool_called"] += 1
            if result.get("generation_success"):
                task_stats[task]["generation_success"] += 1

    for t in task_stats.values():
        t["tool_usage_rate"] = t["tool_called"] / t["total"] if t["total"] else 0.0
        t["generation_success_rate"] = t["generation_success"] / t["total"] if t["total"] else 0.0

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    n_ok = len(successful)

    return {
        "config_name": config_name,
        "tool_label": label,
        "model": model,
        "total_samples": len(data),
        "successful_samples": n_ok,
        "failed_samples": len(failed),
        "tool_called_count": tool_called_count,
        "generation_success_count": gen_success_count,
        "tool_usage_rate": tool_called_count / n_ok if n_ok else 0.0,
        "generation_success_rate": gen_success_count / n_ok if n_ok else 0.0,
        "average_inference_time_seconds": total_time / n_ok if n_ok else 0.0,
        "average_file_size_mb": (total_file_size / gen_success_count / 1024 / 1024) if gen_success_count else 0.0,
        "task_statistics": task_stats,
        "failed_samples_details": failed,
        "detailed_results": results,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_video_results(results: Dict[str, Any]):
    label = results.get("tool_label", results["config_name"])
    print(f"\n{'='*70}")
    print(f"{label} Evaluation Results — {results['config_name']}")
    print(f"{'='*70}")
    print(f"Model:                    {results['model']}")
    print(f"Total samples:            {results['total_samples']}")
    print(f"Successfully processed:   {results['successful_samples']}")
    print(f"Failed to process:        {results['failed_samples']}")
    print(f"Tool called:              {results['tool_called_count']}  "
          f"(usage rate: {results['tool_usage_rate']:.1%})")
    print(f"Video generation success: {results['generation_success_count']}  "
          f"(success rate: {results['generation_success_rate']:.1%})")
    print(f"Avg inference time:       {results['average_inference_time_seconds']:.1f}s")
    print(f"Avg file size:            {results['average_file_size_mb']:.2f} MB")

    if results.get("task_statistics"):
        print(f"\n{'Task Breakdown':-<60}")
        print(f"{'Task':<15} {'Total':<8} {'Called':<10} {'Gen OK':<10} "
              f"{'Usage%':<10} {'SuccRate%'}")
        print("-" * 70)
        for task, s in results["task_statistics"].items():
            print(f"{task:<15} {s['total']:<8} {s['tool_called']:<10} "
                  f"{s['generation_success']:<10} "
                  f"{s['tool_usage_rate']:<10.1%} {s['generation_success_rate']:.1%}")

    if results.get("failed_samples_details"):
        print(f"\n{'Failed Samples':-<60}")
        for r in results["failed_samples_details"]:
            print(f"  ID={r.get('id', '?')}  error={r.get('error', 'unknown')}")
    print(f"{'='*70}\n")


def print_image_results(results: Dict[str, Any]):
    label = results.get("tool_label", results["config_name"])
    print(f"\n{'='*70}")
    print(f"{label} Evaluation Results — {results['config_name']}")
    print(f"{'='*70}")
    print(f"Model:                    {results['model']}")
    print(f"Total samples:            {results['total_samples']}")
    print(f"Successfully processed:   {results['successful_samples']}")
    print(f"Failed to process:        {results['failed_samples']}")
    print(f"Tool called:              {results['tool_called_count']}  "
          f"(usage rate: {results['tool_usage_rate']:.1%})")
    print(f"Image generation success: {results['generation_success_count']}  "
          f"(success rate: {results['generation_success_rate']:.1%})")
    print(f"Avg inference time:       {results['average_inference_time_seconds']:.1f}s")
    print(f"Avg file size:            {results['average_file_size_mb']:.2f} MB")

    if results.get("task_statistics"):
        print(f"\n{'Task Breakdown':-<60}")
        print(f"{'Task':<15} {'Total':<8} {'Called':<10} {'Gen OK':<10} "
              f"{'Usage%':<10} {'SuccRate%'}")
        print("-" * 70)
        for task, s in results["task_statistics"].items():
            print(f"{task:<15} {s['total']:<8} {s['tool_called']:<10} "
                  f"{s['generation_success']:<10} "
                  f"{s['tool_usage_rate']:<10.1%} {s['generation_success_rate']:.1%}")

    if results.get("failed_samples_details"):
        print(f"\n{'Failed Samples':-<60}")
        for r in results["failed_samples_details"]:
            print(f"  ID={r.get('id', '?')}  error={r.get('error', 'unknown')}")
    print(f"{'='*70}\n")


def print_pi3_statistics(results: Dict[str, Any]):
    """Print Pi3 tool parameter statistics if present in results."""
    if "pi3_statistics" not in results:
        return

    pi3_stats = results["pi3_statistics"]
    print(f"\n{'='*80}")
    print("Pi3 Tool Parameter Statistics")
    print(f"{'='*80}")
    print(f"Total Pi3 calls:              {pi3_stats['total_pi3_calls']}")
    print(f"Unique angle combinations:    {pi3_stats['unique_angle_combinations']}")

    print(f"\n{'Top 5 Most Used Angle Combinations:':<50}")
    print(f"{'Angle (azimuth, elevation)':<35} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    for item in pi3_stats["top_5_angle_combinations"]:
        print(f"{item['angle']:<35} {item['count']:<10} {item['percentage']:<15}")

    print(f"\n{'Rotation Reference Camera Usage:':<50}")
    print(f"{'Camera':<20} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    for camera, count in pi3_stats["rotation_reference_camera_usage"].items():
        pct = pi3_stats["rotation_reference_camera_percentage"][camera]
        print(f"{camera:<20} {count:<10} {pct:<15}")

    print(f"\n{'Camera View Mode Usage:':<50}")
    print(f"{'Mode':<20} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    for mode, count in pi3_stats["camera_view_usage"].items():
        pct = pi3_stats["camera_view_percentage"][mode]
        display = "Enabled (True)" if "true" in mode else "Disabled (False)"
        print(f"{display:<20} {count:<10} {pct:<15}")

    print(f"\n{'Full Angle Distribution:':<50}")
    print(f"{'Angle (azimuth, elevation)':<35} {'Count':<10}")
    print("-" * 80)
    for angle, count in pi3_stats["angle_distribution"].items():
        print(f"{angle:<35} {count:<10}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_config_choices = sorted(ALL_CONFIGS) + list(CONFIG_GROUPS.keys())

    parser = argparse.ArgumentParser(
        description="Unified SPAgent evaluation script covering all tools.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- required / data ---
    parser.add_argument("--data_path", type=str, default="dataset/cvbench_data.jsonl",
                        help="Path to the evaluation JSONL dataset.")
    parser.add_argument("--image_base_path", type=str, default="dataset",
                        help="Base directory for resolving relative image paths.")

    # --- tool selection ---
    parser.add_argument(
        "--config", type=str, default=None,
        choices=all_config_choices,
        help=(
            "Tool configuration to evaluate. Options:\n"
            "  Vision:  dinosam | pi3x\n"
            "  Video:   veo_real | veo_mock\n"
            "  Image:   sana_real | sana_mock\n"
            "  Groups:  all_vision | all_video | all_image | all_generation | all\n"
            "Defaults to 'dinosam' if not specified."
        ),
    )

    # --- model ---
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model name for the agent orchestrator (default: gpt-4o).")

    # --- sampling ---
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (default: all).")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Parallel worker threads (default: 4).")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum tool-call iterations per sample (default: 3).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling probability mass (default: 1.0).")
    parser.add_argument("--task", type=str, default="all",
                        help="Task filter label used in the output filename (default: all).")

    # --- video-specific ---
    parser.add_argument("--video_num_frames", type=int, default=4,
                        help="Frames sampled from generated video to feed back to model (default: 4).")

    # --- data collection ---
    parser.add_argument("--enable_data_collection", action="store_true",
                        help="Enable training data collection for vision configs.")
    parser.add_argument("--data_output_dir", type=str, default=None,
                        help="Directory for training data (auto-generated timestamp dir if not set).")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at {args.data_path}")
        return
    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        return

    # Resolve config → list of config names
    config_arg = args.config or "dinosam"
    if config_arg in CONFIG_GROUPS:
        configs_to_run = CONFIG_GROUPS[config_arg]
    else:
        configs_to_run = [config_arg]

    print(f"\nRunning evaluation for: {configs_to_run}")

    tool_configs = build_tool_configs(configs_to_build=configs_to_run)
    all_results: Dict[str, Any] = {}

    for config_name in configs_to_run:
        tools = tool_configs[config_name]

        # ---- VIDEO GENERATION configs ----
        if config_name in VIDEO_GEN_CONFIGS:
            results = evaluate_video_config(
                config_name=config_name,
                tools=tools,
                data_path=args.data_path,
                image_base_path=args.image_base_path,
                model=args.model,
                max_samples=args.max_samples,
                max_workers=min(args.max_workers, 2),  # rate-limit safety
                max_iterations=args.max_iterations,
                temperature=args.temperature,
                seed=args.seed,
                top_p=args.top_p,
                video_num_frames=args.video_num_frames,
            )
            all_results[config_name] = results
            print_video_results(results)

        # ---- IMAGE GENERATION configs ----
        elif config_name in IMAGE_GEN_CONFIGS:
            results = evaluate_image_config(
                config_name=config_name,
                tools=tools,
                data_path=args.data_path,
                image_base_path=args.image_base_path,
                model=args.model,
                max_samples=args.max_samples,
                max_workers=min(args.max_workers, 2),
                max_iterations=args.max_iterations,
                temperature=args.temperature,
                seed=args.seed,
                top_p=args.top_p,
            )
            all_results[config_name] = results
            print_image_results(results)

        # ---- VISION / PERCEPTION configs ----
        else:
            data_collector = None
            if args.enable_data_collection:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = (
                    args.data_output_dir
                    or f"training_data/{config_name}_{args.model.replace('-', '_')}_{ts}"
                )
                data_collector = DataCollector(
                    output_dir=out_dir,
                    save_images=True,
                    auto_save=True,
                )
                print(f"Data collection enabled: {out_dir}")

            extra_kwargs: Dict[str, Any] = {}
            use_vision_prompts = True
            if use_vision_prompts:
                extra_kwargs["system_prompt"] = GENERAL_VISION_SYSTEM_PROMPT
                extra_kwargs["continuation_hint"] = GENERAL_VISION_CONTINUATION_HINT

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
                temperature=args.temperature,
                seed=args.seed,
                top_p=args.top_p,
                **extra_kwargs,
            )
            all_results[config_name] = results

            print(f"\nResults for {config_name}:")
            print_evaluation_results(results)
            print_pi3_statistics(results)

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_tag = args.config or "dinosam"
    output_file = (
        f"evaluation_results_{config_tag}_{args.model.replace('-', '_')}_"
        f"iter{args.max_iterations}_{args.task}_{timestamp}.json"
    )

    # Strip raw tool_calls from video results to keep file size manageable
    for cfg_results in all_results.values():
        for r in cfg_results.get("detailed_results", []):
            r.pop("tool_calls", None)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to: {output_file}")


if __name__ == "__main__":
    main()
