"""
Evaluation script for the Sana image generation tool via SPAgent.

Dataset format (JSONL, one JSON object per line):
{
    "id": "unique_id",
    "prompt": "Text description of the image to generate",
    "task": "t2i",
    "size": "1024x1024",
    "num_inference_steps": 20,
    "guidance_scale": 4.5,
    "seed": 42
}
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent import SPAgent
from spagent.core.prompts import GENERATION_CONTINUATION_HINT, GENERATION_SYSTEM_PROMPT
from spagent.models import GPTModel
from spagent.tools import SanaTool


def load_jsonl_data(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file without extra dependencies."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


TOOL_CONFIGS = {
    "sana_real": [
        SanaTool(use_mock=False, server_url="http://127.0.0.1:30000"),
    ],
    "sana_mock": [
        SanaTool(use_mock=True),
    ],
}


def extract_sana_call_info(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Sana tool call information from an agent result."""
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
            if isinstance(k, str) and k.startswith("image_generation_sana_tool_iter")
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
        if call.get("name") == "image_generation_sana_tool":
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
                    if isinstance(nested, dict):
                        info["file_size_bytes"] = nested.get(
                            "file_size_bytes",
                            tool_result.get("file_size_bytes", 0),
                        )
                    else:
                        info["file_size_bytes"] = tool_result.get("file_size_bytes", 0)
                else:
                    info["error"] = tool_result.get("error", "Unknown error")
            break

    return info


def evaluate_single_sana_sample(
    agent: SPAgent,
    sample: Dict[str, Any],
    config_name: str = "sana",
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """Evaluate a single Sana generation sample."""
    sample_id = sample.get("id", "unknown")
    prompt = sample.get("prompt", "").strip()
    if not prompt:
        return {
            "id": sample_id,
            "success": False,
            "error": "Empty prompt",
        }

    size = sample.get("size", "1024x1024")
    num_inference_steps = sample.get("num_inference_steps", 20)
    guidance_scale = sample.get("guidance_scale", 4.5)
    seed = sample.get("seed", 42)
    negative_prompt = sample.get("negative_prompt")
    task_type = sample.get("task", "t2i")

    agent_prompt = (
        "Generate an image using Sana based on the following description. "
        "Treat the result as a synthetic visualization, not as evidence from a real observation.\n\n"
        f"Prompt: {prompt}\n"
        f"Preferred size: {size}\n"
        f"Sampling steps: {num_inference_steps}\n"
        f"Guidance scale: {guidance_scale}\n"
        f"Seed: {seed}\n"
    )
    if negative_prompt:
        agent_prompt += f"Negative prompt: {negative_prompt}\n"

    try:
        start_time = time.time()
        agent_result = agent.solve_problem(
            [],
            agent_prompt,
            max_iterations=max_iterations,
        )
        inference_time = time.time() - start_time

        sana_call_info = extract_sana_call_info(agent_result)
        generation_success = sana_call_info["called"] and sana_call_info["success"]

        return {
            "id": sample_id,
            "success": True,
            "prompt": prompt,
            "task": task_type,
            "sana_called": sana_call_info["called"],
            "generation_success": generation_success,
            "output_path": sana_call_info.get("output_path", ""),
            "image_paths": sana_call_info.get("image_paths", []),
            "file_size_bytes": sana_call_info.get("file_size_bytes", 0),
            "sana_prompt_used": sana_call_info.get("prompt_used", ""),
            "sana_size": sana_call_info.get("size"),
            "sana_num_inference_steps": sana_call_info.get("num_inference_steps"),
            "sana_guidance_scale": sana_call_info.get("guidance_scale"),
            "sana_seed": sana_call_info.get("seed"),
            "sana_negative_prompt": sana_call_info.get("negative_prompt"),
            "sana_error": sana_call_info.get("error"),
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


def evaluate_sana_config(
    config_name: str,
    tools: List[Any],
    data_path: str,
    model: str = "gpt-4o",
    max_samples: Optional[int] = None,
    max_workers: int = 1,
    max_iterations: int = 3,
    temperature: float = 0.0,
    seed: int = 42,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    """Run a full Sana evaluation for a given tool configuration."""
    print(f"\nEvaluating configuration: {config_name}")
    print(f"Loading data from: {data_path}")

    data = load_jsonl_data(data_path)
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

    results = []
    start_total = time.time()
    for sample in tqdm(data, desc=f"Evaluating {config_name}"):
        result = evaluate_single_sana_sample(
            agent=agent,
            sample=sample,
            config_name=config_name,
            max_iterations=max_iterations,
        )
        results.append(result)
    total_time = time.time() - start_total

    successful = [r for r in results if r.get("success")]
    generated = [r for r in successful if r.get("generation_success")]

    return {
        "config_name": config_name,
        "model": model,
        "total_samples": len(results),
        "successful_samples": len(successful),
        "generation_success_count": len(generated),
        "generation_success_rate": (len(generated) / len(successful)) if successful else 0.0,
        "average_inference_time_seconds": (
            sum(r.get("inference_time", 0.0) for r in successful) / len(successful)
            if successful else 0.0
        ),
        "total_runtime_seconds": total_time,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SanaTool via SPAgent")
    parser.add_argument(
        "--config",
        choices=sorted(TOOL_CONFIGS.keys()),
        default="sana_mock",
        help="Tool configuration to evaluate.",
    )
    parser.add_argument(
        "--data_path",
        default="dataset/sana_cases_sample.jsonl",
        help="Path to the Sana JSONL dataset.",
    )
    parser.add_argument("--model", default="gpt-4o", help="LLM model name.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    results = evaluate_sana_config(
        config_name=args.config,
        tools=TOOL_CONFIGS[args.config],
        data_path=args.data_path,
        model=args.model,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
    )

    output_file = f"sana_evaluation_results_{args.config}_{args.model.replace('-', '_')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(
        {
            "config_name": results["config_name"],
            "total_samples": results["total_samples"],
            "successful_samples": results["successful_samples"],
            "generation_success_count": results["generation_success_count"],
            "generation_success_rate": results["generation_success_rate"],
            "average_inference_time_seconds": results["average_inference_time_seconds"],
            "output_file": output_file,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
