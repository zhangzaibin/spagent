"""
SPAgentVLMEvalModel
===================
A VLMEvalKit-compatible model wrapper around SPAgent.

Usage
-----
This class inherits from ``vlmeval.vlm.base.BaseModel`` and implements
``generate_inner(msgs, dataset=None)``.  VLMEvalKit drives the entire
evaluation loop (data loading, answer scoring, writing xlsx results);
we only need to plug in the SPAgent inference and write per-sample
trace files so ``analyze_failures.py`` can do failure attribution later.

Trace format (one JSON per sample under ``<trace_dir>/<dataset>/<index>.json``)
::
    {
      "index":        int,
      "dataset":      str,
      "question":     str,
      "image_paths":  list[str],
      "answer":       str,
      "used_tools":   list[str],
      "tool_calls":   list[dict],
      "tool_results": dict,   # image fields stripped to bool flags
      "iterations":   int
    }
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from vlmeval.vlm.base import BaseModel

# We import SPAgent lazily inside __init__ so this file can be imported
# even when the spagent package is not yet on sys.path (tests, etc.)


# ── Helpers borrowed from examples/evaluation/spagent_evaluation.py ──────────

_IMAGE_FIELDS = {
    'image', 'images', 'img',
    'camera_views', 'camera_view',
    'output_path', 'vis_path', 'visualization_path',
    'frames', 'frame',
    'visualization', 'rendered_image', 'rendered_images',
    'depth_map', 'depth_image',
    'mask', 'masks', 'segmentation_mask',
    'image_path', 'image_paths', 'img_path',
}


def _clean_dict(data):
    """Recursively strip image data from a dict, replacing with bool flags."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if k in _IMAGE_FIELDS:
                out[f'has_{k}'] = True
            elif isinstance(v, dict):
                out[k] = _clean_dict(v)
            elif isinstance(v, list):
                out[k] = [_clean_dict(i) if isinstance(i, dict) else i for i in v]
            else:
                out[k] = v
        return out
    return data


# ── Main wrapper class ────────────────────────────────────────────────────────

class SPAgentVLMEvalModel(BaseModel):
    """VLMEvalKit model wrapper that delegates to SPAgent."""

    INSTALL_REQ = False
    INTERLEAVE = True      # supports multiple images per question (needed for BLINK)

    # Thread-safe counter for trace file naming within a dataset run
    _counter_lock = threading.Lock()

    def __init__(
        self,
        agent,                          # SPAgent instance
        trace_dir: str = "outputs/spagent_traces",
        dataset_tag: str = "unknown",
        max_iterations: int = 3,
    ):
        """
        Parameters
        ----------
        agent:          A fully-configured ``SPAgent`` instance.
        trace_dir:      Root directory for per-sample JSON traces.
        dataset_tag:    Identifier appended to trace sub-directory name
                        (typically ``f"{model_name}_{config_name}"``).
        max_iterations: Maximum tool-call iterations passed to ``solve_problem``.
        """
        # BaseModel.__init__ usually sets self.model_name; we override so
        # VLMEvalKit names the output files correctly.
        self.model_name = dataset_tag   # used by VLMEvalKit for output file names
        self.agent = agent
        self.trace_dir = Path(trace_dir)
        self.dataset_tag = dataset_tag
        self.max_iterations = max_iterations
        self._sample_counter: Dict[str, int] = {}  # dataset_name -> count

    # ── VLMEvalKit interface ──────────────────────────────────────────────────

    def generate_inner(self, msgs: List[Dict[str, Any]], dataset: Optional[str] = None) -> str:
        """
        Called by VLMEvalKit for each sample.

        Parameters
        ----------
        msgs:    Interleaved list of ``{"type": "image"|"text", "value": ...}``.
        dataset: Dataset name string passed by VLMEvalKit (e.g. ``"MMStar"``).

        Returns
        -------
        str: Raw prediction string (VLMEvalKit will parse A/B/C/D or run a judge).
        """
        # Split into images and text
        image_paths: List[str] = [m['value'] for m in msgs if m['type'] == 'image']
        text_parts: List[str]  = [m['value'] for m in msgs if m['type'] == 'text']
        question = "\n".join(text_parts).strip()

        # Run SPAgent
        result = self.agent.solve_problem(
            image_path=image_paths if image_paths else [],
            question=question,
            max_iterations=self.max_iterations,
        )

        # Write trace (non-blocking; errors here must not affect scoring)
        try:
            self._dump_trace(question, image_paths, result, dataset)
        except Exception as exc:
            import warnings
            warnings.warn(f"[SPAgentVLMEvalModel] trace dump failed: {exc}")

        return result["answer"]

    # ── Trace helpers ─────────────────────────────────────────────────────────

    def _next_index(self, dataset_name: str) -> int:
        with self._counter_lock:
            idx = self._sample_counter.get(dataset_name, 0)
            self._sample_counter[dataset_name] = idx + 1
            return idx

    def _dump_trace(
        self,
        question: str,
        image_paths: List[str],
        result: Dict[str, Any],
        dataset_name: Optional[str],
    ) -> None:
        """Write a cleaned trace JSON for one sample."""
        dataset_name = dataset_name or "unknown"
        idx = self._next_index(dataset_name)

        # Build output path: <trace_dir>/<dataset_tag>/<dataset>/<index:05d>.json
        out_dir = self.trace_dir / self.dataset_tag / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{idx:05d}.json"

        # Clean tool_results of any image blobs
        raw_tool_results = result.get("tool_results", {})
        cleaned_tool_results = {
            k: _clean_dict(v) if isinstance(v, dict) else v
            for k, v in raw_tool_results.items()
        }

        trace = {
            "index":        idx,
            "dataset":      dataset_name,
            "question":     question,
            "image_paths":  image_paths,
            "answer":       result.get("answer", ""),
            "used_tools":   result.get("used_tools", []),
            "tool_calls":   [_clean_dict(tc) for tc in result.get("tool_calls", [])],
            "tool_results": cleaned_tool_results,
            "iterations":   result.get("iterations", 0),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
