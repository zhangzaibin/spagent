"""
quick_eval.py  —  SPAgent empirical study: quick & direct evaluation script
=================================================================[]===========
Bypasses vlmeval's `infer_data_job` machinery entirely.
Instead: iterate dataset rows → SPAgent.solve_problem → write xlsx → call dataset.evaluate().

Usage
-----
# Smoke test: 5 samples on VStarBench, no tools
    python scripts/quick_eval.py \
    --model  gpt-4.1-mini \
    --datasets VStarBench \
    --limit 5

# No-tools baseline across 5 benchmarks
python scripts/quick_eval.py \
    --model  gpt-4.1-mini \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
    --limit 50

# Perception tools only (detection + segmentation + depth)
python scripts/quick_eval.py \
    --model  gpt-4.1-mini \
    --tools detection segmentation depth \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
    --limit 50 \
    --detection-url  http://localhost:20022 \
    --segmentation-url http://localhost:20020 \
    --depth-url http://localhost:20019

# Detection + depth only (custom combo)
python scripts/quick_eval.py \
    --model  gpt-4.1-mini \
    --tools detection depth \
    --datasets MMStar \
    --limit 50

All 15 mentor benchmarks:
  MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI
  MMBench_dev_en RealWorldQA ScienceQA_VAL
  HRBench4K HRBench8K
  MathVerse_MINI WeMath LogicVista MMMU_Pro_10c DynaMath

Local datasets (no VLMEvalKit registration needed):
  MindCube  — multi-image spatial reasoning
    Prepare: python spagent/utils/download_mindcube.py
    Eval:    python scripts/quick_eval.py --model gpt-4.1-mini --datasets MindCube
    Custom:  ... --mindcube-path /path/to/custom.jsonl
             ... --mindcube-path dataset/mindcube/data/raw/MindCube_tinybench.jsonl

  VSIBench  — video spatial reasoning
    Prepare: python spagent/utils/download_vsibench.py
    Eval:    python scripts/quick_eval.py --model gpt-4.1-mini --datasets VSIBench
    Custom:  ... --vsibench-path /path/to/VSI_Bench.jsonl

  Any custom JSONL (same schema):
    python scripts/quick_eval.py --model gpt-4.1-mini --datasets MyDataset --data-path /path/to/data.jsonl
"""

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("LMUData", str(Path.home() / "LMUData"))
warnings.filterwarnings("ignore", message=".*LiteLLM.*")

# ── Load .env if present (project root or current directory) ──────────────────
def _load_dotenv():
    """Load .env file from project root or CWD if dotenv is available."""
    try:
        from dotenv import load_dotenv
        for candidate in [_ROOT / ".env", Path(".env")]:
            if candidate.exists():
                load_dotenv(candidate, override=False)
                return
    except ImportError:
        pass  # python-dotenv not installed; rely on shell environment

_load_dotenv()

# ── SPAgent ───────────────────────────────────────────────────────────────────
from spagent.core import SPAgent, GENERAL_VISION_CONTINUATION_HINT
from spagent.models import GPTModel, QwenModel, QwenVLLMModel


# ── Model factory ─────────────────────────────────────────────────────────────

def _build_model(model_name: str, backend: str, temperature: float, seed: int):
    """
    Instantiate the right Model subclass.

    backend:
      auto  — infer from model_name (default)
      gpt   — always use GPTModel (OpenAI-compatible API)
      qwen  — always use QwenModel (DashScope API)
      qwen-vllm — always use QwenVLLMModel (local vLLM server)
    """
    resolved = backend
    if backend == "auto":
        lower = model_name.lower()
        if lower.startswith("qwen"):
            resolved = "qwen"
        else:
            resolved = "gpt"

    if resolved == "gpt":
        return GPTModel(model_name=model_name, temperature=temperature, seed=seed)
    elif resolved == "qwen":
        return QwenModel(model_name=model_name, temperature=temperature)
    elif resolved == "qwen-vllm":
        return QwenVLLMModel(model_name=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model backend: {resolved!r}")

# ── vlmeval dataset helpers ───────────────────────────────────────────────────
from vlmeval.dataset import build_dataset
from vlmeval.smp.vlm import decode_base64_to_image_file

import pandas as pd

# ── image-field cleaner (reuse from wrapper) ──────────────────────────────────
_IMAGE_FIELDS = {
    'image', 'images', 'img', 'camera_views', 'camera_view',
    'output_path', 'vis_path', 'frames', 'frame',
    'depth_map', 'depth_image', 'mask', 'masks',
    'image_path', 'image_paths', 'img_path',
}

def _json_default(obj):
    """Fallback serializer for json.dump: handles numpy arrays, PIL images, etc."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            return f"<PIL.Image size={obj.size} mode={obj.mode}>"
    except ImportError:
        pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return f"<{obj.__class__.__name__}>"
    return str(obj)


def _clean(data):
    """Recursively strip image fields and convert non-JSON-serializable objects."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if k in _IMAGE_FIELDS:
                out[f'has_{k}'] = True
            else:
                out[k] = _clean(v)
        return out
    if isinstance(data, (list, tuple)):
        return [_clean(i) for i in data]
    # numpy / other non-serialisable scalars
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            return f"<ndarray shape={data.shape} dtype={data.dtype}>"
        if isinstance(data, (np.integer,)):
            return int(data)
        if isinstance(data, (np.floating,)):
            return float(data)
    except ImportError:
        pass
    return data


# ── Tool factory ──────────────────────────────────────────────────────────────

# Maps CLI tool name → (import class name, args attribute for server_url or None)
_TOOL_SPECS: Dict[str, tuple] = {
    "detection":    ("ObjectDetectionTool",   "detection_url"),
    "segmentation": ("SegmentationTool",      "segmentation_url"),
    "depth":        ("DepthEstimationTool",   "depth_url"),
    "pi3x":         ("Pi3XTool",             "pi3x_url"),
    "pi3":          ("Pi3Tool",              "pi3_url"),
    "vggt":         ("VGGTTool",             "vggt_url"),
    "mapanything":  ("MapAnythingTool",      "mapanything_url"),
    "yoloe":        ("YOLOETool",            "yoloe_url"),
    "supervision":  ("SupervisionTool",      "supervision_url"),
    "moondream":    ("MoondreamTool",        "moondream_url"),
    "molmo2":       ("Molmo2Tool",           "molmo2_url"),
    "orient":       ("OrientAnythingV2Tool", "orient_url"),
    "vace":         ("VaceTool",             "vace_url"),
    "sana":         ("SanaTool",             "sana_url"),
    "qwenvl":       ("QwenVLTool",           None),
    "veo":          ("VeoTool",              None),
    "sora":         ("SoraTool",             None),
    "wan":          ("WanTool",              None),
}

ALL_TOOL_NAMES = list(_TOOL_SPECS.keys())


def make_tools(tool_names: List[str], args) -> List[Any]:
    if not tool_names:
        return []

    import spagent.tools as _tools_module
    tools = []
    for name in tool_names:
        spec = _TOOL_SPECS.get(name)
        if spec is None:
            print(f"  [WARN] Unknown tool name: {name!r}, skipping")
            continue
        cls_name, url_attr = spec
        cls = getattr(_tools_module, cls_name, None)
        if cls is None:
            print(f"  [WARN] {cls_name} not found in spagent.tools, skipping")
            continue
        try:
            if url_attr is not None:
                server_url = getattr(args, url_attr.replace("-", "_"), None)
                kwargs = dict(use_mock=False, server_url=server_url)
                if name == "vace":
                    kwargs["timeout_seconds"] = getattr(args, "vace_timeout", 480)
                tools.append(cls(**kwargs))
            else:
                tools.append(cls(use_mock=False))
        except Exception as e:
            print(f"  [WARN] {cls_name}: {e}")
    return tools


# ── Dataset sub-sampling ──────────────────────────────────────────────────────

DEFAULT_LIMIT = {
    "MMStar": 200, "VStarBench": None, "BLINK": 200,
    "MMMU_DEV_VAL": 150, "MathVista_MINI": 200,
    "MMBench_dev_en": 200, "RealWorldQA": 200, "ScienceQA_VAL": 200,
    "HRBench4K": 200, "HRBench8K": 200, "MathVerse_MINI": 200,
    "WeMath": 200, "LogicVista": 200, "MMMU_Pro_10c": 150, "DynaMath": 200,
    # Local datasets
    "MindCube": None, "VSIBench": None,
}

# ── Local JSONL dataset wrapper ───────────────────────────────────────────────
#
# Follows the same loading pattern as evaluate_pi3x.py / spagent_evaluation.py:
#   • raw_items   - original dicts from JSONL (conversations / image / video fields)
#   • image_base_path - prepended to relative paths exactly like validate_sample_paths()
#   • data        - minimal DataFrame for resume bookkeeping only

class _LocalDataset:
    """
    Wrapper around a local JSONL file (MindCube / VSIBench schema).

    JSONL schema:
      id, image (list[str]), video (list[str]),
      conversations [{from: human, value: question}, {from: gpt, value: answer}]
      Optional: task, output_type, data_source, others
    """

    def __init__(
        self,
        name: str,
        jsonl_path: str,
        image_base_path: Optional[str] = None,
    ):
        self.name = name
        self.jsonl_path = jsonl_path
        # image paths in JSONL are relative to this directory (same as evaluate_pi3x.py)
        self.image_base_path: str = image_base_path or str(_ROOT / "dataset")

        self.raw_items: List[Dict] = []
        rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.raw_items.append(item)

                # Extract question / answer from conversations for bookkeeping
                question, answer = "", ""
                for turn in item.get("conversations", []):
                    if turn.get("from") == "human":
                        question = turn.get("value", "")
                    elif turn.get("from") == "gpt":
                        answer = turn.get("value", "")

                rows.append({
                    "index":       item.get("id", str(i)),
                    "question":    question,
                    "answer":      answer,
                    "task":        item.get("task", ""),
                    "output_type": item.get("output_type", "MCQ"),
                    "data_source": item.get("data_source", name),
                })
        self.data = pd.DataFrame(rows)


# Maps dataset name → (default JSONL path, image_base_path)
LOCAL_DATASET_PATHS: Dict[str, str] = {
    "MindCube": "dataset/MindCube_data.jsonl",
    "VSIBench":  "dataset/VSI_Bench.jsonl",
}


def load_dataset(name: str, limit: Optional[int], local_path: Optional[str] = None):
    # ── Local JSONL (MindCube, VSIBench, or any explicit --data-path) ──────────
    jsonl_path = local_path or LOCAL_DATASET_PATHS.get(name)
    if jsonl_path:
        full = Path(_ROOT) / jsonl_path if not Path(jsonl_path).is_absolute() else Path(jsonl_path)
        if not full.exists():
            raise FileNotFoundError(
                f"Local JSONL for {name!r} not found: {full}\n"
                f"Prepare it with: python spagent/utils/download_mindcube.py  (MindCube)\n"
                f"                 python spagent/utils/download_vsibench.py   (VSIBench)"
            )
        print(f"  Loading local JSONL: {full}")
        ds = _LocalDataset(name, str(full))
        if limit and len(ds.raw_items) > limit:
            ds.raw_items = ds.raw_items[:limit]
            ds.data = ds.data.head(limit).reset_index(drop=True)
        return ds

    # ── VLMEvalKit datasets ────────────────────────────────────────────────────
    ds = build_dataset(name)
    df = ds.data
    if limit and len(df) > limit:
        if name == "BLINK" and "category" in df.columns:
            cats = df["category"].unique()
            per_cat = max(1, limit // len(cats))
            df = (df.groupby("category", group_keys=False)
                    .apply(lambda g: g.head(per_cat))
                    .reset_index(drop=True)
                    .head(limit))
        else:
            df = df.head(limit)
        ds.data = df
    return ds


# ── Image decode helper (VLMEvalKit datasets only) ────────────────────────────

_img_tmp_dir = Path(tempfile.mkdtemp(prefix="spagent_eval_imgs_"))


def decode_images(line: Dict, ds_name: str) -> List[str]:
    """Decode base64-encoded images from a VLMEvalKit dataset row."""
    raw = line.get("image", "")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []

    entries = raw if isinstance(raw, list) else [raw]
    paths = []
    for i, entry in enumerate(entries):
        entry = str(entry)
        if len(entry) < 512 and Path(entry).exists():
            paths.append(entry)
            continue
        idx = str(line.get("index", "unknown"))
        tmp_path = str(_img_tmp_dir / f"{ds_name}_{idx}_{i}.jpg")
        try:
            decode_base64_to_image_file(entry, tmp_path)
            paths.append(tmp_path)
        except Exception as exc:
            print(f"  [WARN] image decode failed for index={idx}: {exc}")
    return paths


# ── Single-sample inference ───────────────────────────────────────────────────

def _build_conversation_log(result: Dict, question: str, image_paths: List[str]) -> Dict:
    """
    Build a structured conversation log from memory_entries for debugging.

    Returns a dict with:
      - prompts:       system_prompt, user_prompt, workflow
      - conversation:  chronological list of turns (role / text / images / metadata)
    """
    entries = result.get("memory_entries", [])
    turns = []
    for e in entries:
        turn: Dict[str, Any] = {
            "role":       e["role"],
            "type":       e["entry_type"],
            "text":       e.get("text"),
            "images":     e.get("images", []),
        }
        meta = e.get("metadata", {})
        if meta:
            turn["metadata"] = meta
        turns.append(turn)

    return {
        "prompts":      result.get("prompts", {}),
        "conversation": turns,
    }


def infer_sample(
    agent: SPAgent,
    line: Dict,
    ds_name: str,
    max_iterations: int,
    trace_dir: Optional[Path],
    model_tag: str,
    position: int,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run SPAgent on one VLMEvalKit dataset sample. Returns result dict."""
    image_paths = decode_images(line, ds_name)

    question = str(line.get("question", ""))
    # VLMEvalKit MCQ datasets have options in separate A/B/C/D/E columns
    for opt in ["A", "B", "C", "D", "E"]:
        if opt in line and pd.notna(line[opt]):
            question += f"\n({opt}) {line[opt]}"

    t0 = time.time()
    try:
        result = agent.solve_problem(
            image_path=image_paths or [],
            question=question,
            max_iterations=max_iterations,
        )
        prediction = result.get("answer", "")
        error = None
    except Exception as exc:
        prediction = f"INFER_FAIL: {exc}"
        result = {}
        error = str(exc)
    elapsed = time.time() - t0

    # Write trace
    if trace_dir is not None:
        td = trace_dir / model_tag / ds_name
        td.mkdir(parents=True, exist_ok=True)
        trace = {
            "index":        position,
            "dataset":      ds_name,
            "question":     question,
            "image_paths":  image_paths,
            "answer":       prediction,
            "used_tools":   result.get("used_tools", []),
            "tool_calls":   [_clean(tc) for tc in result.get("tool_calls", [])],
            "tool_results": {k: _clean(v) for k, v in result.get("tool_results", {}).items()},
            "iterations":   result.get("iterations", 0),
            "elapsed_s":    round(elapsed, 2),
            "error":        error,
        }
        with open(td / f"{position:05d}.json", "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2, default=_json_default)

        # Debug: save full conversation log (prompts + model I/O per iteration)
        if debug and result:
            conv_log = _build_conversation_log(result, question, image_paths)
            conv_log["index"]      = position
            conv_log["dataset"]    = ds_name
            conv_log["question"]   = question
            conv_log["image_paths"] = image_paths
            conv_log["answer"]     = prediction
            conv_log["elapsed_s"]  = round(elapsed, 2)
            with open(td / f"{position:05d}_conv.json", "w", encoding="utf-8") as f:
                json.dump(conv_log, f, ensure_ascii=False, indent=2, default=_json_default)

    return {
        "index":      line.get("index", position),
        "prediction": prediction,
        "elapsed":    elapsed,
    }


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def run_dataset(
    ds_name: str,
    ds,
    agent: SPAgent,
    model_tag: str,
    work_dir: Path,
    trace_dir: Path,
    judge_model: str,
    max_iterations: int,
    nproc: int,
    debug: bool = False,
) -> Dict[str, Any]:
    from tqdm import tqdm

    out_dir = work_dir / model_tag / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"{model_tag}_{ds_name}.xlsx"

    print(f"  Running {len(ds.data)} samples ...")
    predictions = {}

    # If xlsx already exists, load prior results to allow resuming
    if xlsx_path.exists():
        prior = pd.read_excel(xlsx_path)
        if "index" in prior.columns and "prediction" in prior.columns:
            predictions = dict(zip(prior["index"].astype(str), prior["prediction"]))
            print(f"  Resuming: {len(predictions)} prior predictions loaded")

    rows = list(ds.data.iterrows())
    for pos, (_, line) in enumerate(tqdm(rows, desc=f"  {ds_name}")):
        idx_key = str(line.get("index", pos))
        if idx_key in predictions:
            continue   # already done
        r = infer_sample(agent, dict(line), ds_name, max_iterations, trace_dir, model_tag, pos,
                         debug=debug)
        predictions[str(r["index"])] = r["prediction"]

    # Merge predictions back into dataset df and write xlsx
    # Drop 'image' column (base64 blobs) to keep xlsx readable
    df = ds.data.copy()
    df["prediction"] = [predictions.get(str(idx), "INFER_FAIL: missing") for idx in df["index"]]
    df_save = df.drop(columns=[c for c in ["image"] if c in df.columns])
    df_save.to_excel(xlsx_path, index=False)
    print(f"  Saved predictions → {xlsx_path}")

    # Score via VLMEvalKit judge
    scores: Dict[str, Any] = {}
    print(f"  Scoring (judge={judge_model}) ...")
    try:
        # vlmeval's OpenAIWrapper expects a *full* URL ending in /chat/completions,
        # while the OpenAI SDK (used by SPAgent) accepts a base URL without that suffix.
        # Normalise: read OPENAI_API_BASE first (vlmeval's own var), fall back to
        # OPENAI_BASE_URL and append the missing path segment if needed.
        api_base = (
            os.environ.get("OPENAI_API_BASE", "")
            or os.environ.get("OPENAI_BASE_URL", "")
        )
        if api_base and not api_base.rstrip("/").endswith("/chat/completions"):
            api_base = api_base.rstrip("/") + "/chat/completions"
        judge_kwargs: Dict[str, Any] = {"model": judge_model, "nproc": nproc, "verbose": True}
        if api_base:
            judge_kwargs["api_base"] = api_base
        print(f"  Judge API base: {api_base or '(default openai.com)'}")
        raw = ds.evaluate(str(xlsx_path), **judge_kwargs)
        if raw is None:
            scores = {}
        elif isinstance(raw, pd.DataFrame):
            scores = raw.to_dict(orient="list")
        elif isinstance(raw, dict):
            scores = raw
        else:
            scores = {"result": str(raw)}
    except Exception as exc:
        print(f"  [WARN] evaluate() failed: {exc}")
        if "answer" in df.columns:
            import re as _re
            def _letter(s):
                if not isinstance(s, str): return ""
                m = _re.search(r'\b([A-E])\b', s)
                return m.group(1) if m else s.strip()[:1].upper()
            df["_gt"]   = df["answer"].apply(_letter)
            df["_pred"] = df["prediction"].apply(_letter)
            acc = (df["_gt"] == df["_pred"]).mean()
            scores = {"accuracy": round(acc, 4), "note": "fallback simple MCQ match"}
            print(f"  Fallback accuracy: {acc:.2%}")

    return {
        "dataset":   ds_name,
        "n_samples": len(df),
        "xlsx":      str(xlsx_path),
        "scores":    scores,
    }


# ── Local dataset evaluation (MindCube / VSIBench) ───────────────────────────
#
# Follows the same pattern as examples/evaluation/evaluate_pi3x.py:
#   validate_sample_paths → agent.solve_problem → normalize_answer → accuracy
#
# This keeps the raw conversations / image / video fields intact and uses
# image_base_path (= dataset/) to resolve relative paths, exactly like
# the existing spagent_evaluation.py helpers do.

def _extract_video_frames(video_path: str, num_frames: int = 7) -> List[str]:
    """Uniformly sample *num_frames* frames from a video; return tmp image paths."""
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // num_frames)

    tmp_dir = Path(tempfile.mkdtemp(prefix="spagent_video_frames_"))
    stem = Path(video_path).stem
    paths: List[str] = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            p = str(tmp_dir / f"{stem}_frame_{i}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
    cap.release()
    return paths


def _normalize_answer_local(answer: str) -> str:
    """Extract MCQ letter from <answer>...</answer> or raw text (same as utils.normalize_answer)."""
    import re
    s = answer.strip()
    a_start, a_end = s.find("<answer>"), s.find("</answer>")
    if a_start != -1 and a_end > a_start:
        s = s[a_start + 8 : a_end].strip()
    m = re.search(r'\(([A-E])\)|([A-E])\.', s)
    if m:
        return m.group(1) or m.group(2)
    for ch in s:
        if ch in "ABCDE":
            return ch
    return s


def run_local_dataset(
    ds_name: str,
    ds: "_LocalDataset",
    agent: "SPAgent",
    model_tag: str,
    work_dir: Path,
    trace_dir: Path,
    max_iterations: int,
    debug: bool = False,
    num_video_frames: int = 7,
) -> Dict[str, Any]:
    """
    Evaluate a local JSONL dataset (MindCube / VSIBench) following the same
    pattern as examples/evaluation/evaluate_pi3x.py:

      • validate paths with os.path.join(image_base_path, rel_path)
      • run agent.solve_problem(image_paths, question, max_iterations)
      • score with normalize_answer (letter match, no judge model needed)
    """
    from tqdm import tqdm
    from spagent.utils.utils import validate_sample_paths

    image_base_path = ds.image_base_path
    out_dir = work_dir / model_tag / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: load prior predictions ────────────────────────────────────────
    pred_cache_path = out_dir / f"{model_tag}_{ds_name}_predictions.json"
    predictions: Dict[str, str] = {}
    if pred_cache_path.exists():
        with open(pred_cache_path, encoding="utf-8") as f:
            predictions = json.load(f)
        print(f"  Resuming: {len(predictions)} prior predictions loaded")

    # ── Trace dir ─────────────────────────────────────────────────────────────
    td = trace_dir / model_tag / ds_name
    td.mkdir(parents=True, exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"  Running {len(ds.raw_items)} samples (image_base_path={image_base_path}) ...")
    results: List[Dict] = []
    frame_dirs: List[Path] = []  # track temp dirs for cleanup

    for pos, sample in enumerate(tqdm(ds.raw_items, desc=f"  {ds_name}")):
        sample_id = str(sample.get("id", pos))
        if sample_id in predictions:
            # Already inferred – reconstruct minimal result for stats
            results.append({
                "id":        sample_id,
                "success":   True,
                "prediction": predictions[sample_id],
                "answer":    "",   # will be filled from DataFrame later
                "task":      sample.get("task", ""),
                "skipped":   True,
            })
            continue

        has_image = bool(sample.get("image"))
        has_video = bool(sample.get("video"))
        input_type = "image" if has_image else "video"

        # validate_sample_paths mirrors what evaluate_pi3x.py does
        is_valid, path_result = validate_sample_paths(sample, image_base_path, input_type)
        if not is_valid:
            results.append({"id": sample_id, "success": False,
                             "error": path_result.get("error", "path error"), "task": sample.get("task", "")})
            continue

        image_paths = path_result["path"]
        question    = path_result["question"]
        ground_truth = path_result["ground_truth"]

        # ── For video samples: extract frames before calling the agent ─────────
        extracted_frame_dir: Optional[Path] = None
        if has_video and image_paths:
            frames = _extract_video_frames(image_paths[0], num_frames=num_video_frames)
            if frames:
                extracted_frame_dir = Path(frames[0]).parent
                frame_dirs.append(extracted_frame_dir)
                agent_question = (
                    f"Based on these {len(frames)} uniformly sampled frames from a video, "
                    f"please answer: {question}"
                )
                image_paths = frames
            else:
                results.append({"id": sample_id, "success": False,
                                 "error": "Failed to extract video frames", "task": sample.get("task", "")})
                continue
        else:
            agent_question = question

        import time as _time
        t0 = _time.time()
        try:
            agent_result = agent.solve_problem(
                image_path=image_paths,
                question=agent_question,
                max_iterations=max_iterations,
            )
            prediction = agent_result.get("answer", "")
            error = None
        except Exception as exc:
            prediction = f"INFER_FAIL: {exc}"
            agent_result = {}
            error = str(exc)
        elapsed = _time.time() - t0

        predictions[sample_id] = prediction

        # ── Write per-sample trace ─────────────────────────────────────────────
        trace = {
            "index":       pos,
            "id":          sample_id,
            "dataset":     ds_name,
            "question":    question,
            "image_paths": image_paths,
            "answer":      prediction,
            "ground_truth": ground_truth,
            "used_tools":  agent_result.get("used_tools", []),
            "tool_calls":  [_clean(tc) for tc in agent_result.get("tool_calls", [])],
            "iterations":  agent_result.get("iterations", 0),
            "elapsed_s":   round(elapsed, 2),
            "error":       error,
        }
        with open(td / f"{pos:05d}.json", "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2, default=_json_default)

        if debug and agent_result:
            conv_log = _build_conversation_log(agent_result, question, image_paths)
            conv_log.update({"index": pos, "id": sample_id, "dataset": ds_name,
                             "question": question, "answer": prediction, "elapsed_s": round(elapsed, 2)})
            with open(td / f"{pos:05d}_conv.json", "w", encoding="utf-8") as f:
                json.dump(conv_log, f, ensure_ascii=False, indent=2, default=_json_default)

        results.append({
            "id":          sample_id,
            "success":     error is None,
            "prediction":  prediction,
            "answer":      ground_truth,
            "task":        sample.get("task", ""),
            "elapsed":     elapsed,
            "error":       error,
        })

        # Checkpoint predictions after each sample
        with open(pred_cache_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False)

    # ── Cleanup video frame temp dirs ──────────────────────────────────────────
    import shutil
    for d in frame_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    # ── Fill ground_truth for resumed (skipped) samples from DataFrame ─────────
    df_meta = ds.data.set_index("index")
    for r in results:
        if r.get("skipped") and not r.get("answer"):
            r["answer"] = str(df_meta.at[r["id"], "answer"]) if r["id"] in df_meta.index else ""

    # ── Scoring: normalize_answer letter match (no judge model needed) ─────────
    correct = 0
    task_stats: Dict[str, Dict[str, int]] = {}
    successful = [r for r in results if r.get("success")]

    for r in successful:
        norm_pred = _normalize_answer_local(r.get("prediction", ""))
        norm_gt   = _normalize_answer_local(r.get("answer", ""))
        r["normalized_prediction"] = norm_pred
        r["normalized_ground_truth"] = norm_gt
        r["is_correct"] = (norm_pred == norm_gt)
        if r["is_correct"]:
            correct += 1
        task = r.get("task", "unknown") or "unknown"
        task_stats.setdefault(task, {"correct": 0, "total": 0})
        task_stats[task]["total"] += 1
        if r["is_correct"]:
            task_stats[task]["correct"] += 1

    for task in task_stats:
        t = task_stats[task]
        t["accuracy"] = round(t["correct"] / t["total"], 4) if t["total"] else 0.0

    overall_acc = round(correct / len(successful), 4) if successful else 0.0
    scores = {"Overall": overall_acc, **{t: v["accuracy"] for t, v in task_stats.items()}}

    print(f"  Overall accuracy: {overall_acc:.2%}  ({correct}/{len(successful)})")
    for task, s in sorted(task_stats.items()):
        print(f"    {task:25s}: {s['accuracy']:.2%} ({s['correct']}/{s['total']})")

    # ── Save xlsx for inspection ───────────────────────────────────────────────
    xlsx_path = out_dir / f"{model_tag}_{ds_name}.xlsx"
    df_out = pd.DataFrame([{
        "id":                    r["id"],
        "task":                  r.get("task", ""),
        "answer":                r.get("answer", ""),
        "prediction":            r.get("prediction", ""),
        "normalized_prediction": r.get("normalized_prediction", ""),
        "normalized_gt":         r.get("normalized_ground_truth", ""),
        "is_correct":            r.get("is_correct", False),
        "elapsed_s":             r.get("elapsed", 0),
        "error":                 r.get("error", ""),
    } for r in results])
    df_out.to_excel(xlsx_path, index=False)
    print(f"  Saved results → {xlsx_path}")

    # ── Save full JSON results ─────────────────────────────────────────────────
    json_path = out_dir / f"{model_tag}_{ds_name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset":         ds_name,
            "total_samples":   len(ds.raw_items),
            "successful":      len(successful),
            "failed":          len(results) - len(successful),
            "overall_accuracy": overall_acc,
            "task_statistics": task_stats,
            "scores":          scores,
            "detailed_results": results,
        }, f, ensure_ascii=False, indent=2, default=_json_default)

    return {
        "dataset":   ds_name,
        "n_samples": len(ds.raw_items),
        "xlsx":      str(xlsx_path),
        "scores":    scores,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--model",   default="gpt-4.1-mini",
                        help="Model name (e.g. gpt-4.1-mini, Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--model-backend", default="auto",
                        choices=["auto", "gpt", "qwen", "qwen-vllm"],
                        help=(
                            "Model backend to use. "
                            "'auto' detects from model name (Qwen* → qwen, else → gpt). "
                            "'gpt' uses GPTModel (OpenAI-compatible API, needs OPENAI_API_KEY). "
                            "'qwen' uses QwenModel (DashScope API, needs DASHSCOPE_API_KEY). "
                            "'qwen-vllm' uses QwenVLLMModel (local vLLM server)."
                        ))
    parser.add_argument("--tools",   nargs="*", default=[],
                        choices=ALL_TOOL_NAMES, metavar="TOOL",
                        help=(
                            "Tools to enable (space-separated). "
                            f"Available: {', '.join(ALL_TOOL_NAMES)}. "
                            "Default: no tools."
                        ))
    parser.add_argument("--datasets", nargs="+",
                        default=["VStarBench"])
    parser.add_argument("--limit",   type=int, default=None,
                        help="Override sample limit (default: per-dataset defaults)")
    parser.add_argument("--trace-dir", default="outputs/spagent_traces")
    parser.add_argument("--work-dir",  default="outputs/vlmeval_runs")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--nproc",    type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--no-score", action="store_true",
                        help="Skip the scoring step (just run inference)")
    parser.add_argument("--data-path", default=None,
                        help=(
                            "Path to a local JSONL file. Use when evaluating a custom "
                            "dataset not registered in VLMEvalKit (e.g. MindCube, VSIBench). "
                            "Only valid when --datasets has exactly one entry."
                        ))
    parser.add_argument("--mindcube-path", default="dataset/MindCube_data.jsonl",
                        help="Path to the MindCube JSONL file (default: dataset/MindCube_data.jsonl)")
    parser.add_argument("--vsibench-path", default="dataset/VSI_Bench.jsonl",
                        help="Path to the VSIBench JSONL file (default: dataset/VSI_Bench.jsonl)")
    parser.add_argument("--debug", action="store_true",
                        help=(
                            "Save full conversation log (system prompt, per-iteration "
                            "prompts and model responses) as *_conv.json alongside each trace"
                        ))
    # Tool server URLs (used only when the corresponding tool is enabled)
    parser.add_argument("--detection-url",   default="http://localhost:20022")
    parser.add_argument("--segmentation-url", default="http://localhost:20020")
    parser.add_argument("--depth-url",        default="http://localhost:20019")
    parser.add_argument("--pi3x-url",         default="http://localhost:20031")
    parser.add_argument("--pi3-url",          default="http://localhost:20030")
    parser.add_argument("--vggt-url",         default="http://localhost:20022")
    parser.add_argument("--mapanything-url",  default="http://localhost:20022")
    parser.add_argument("--yoloe-url",        default="http://0.0.0.0:8000")
    parser.add_argument("--supervision-url",  default="http://0.0.0.0:8000")
    parser.add_argument("--moondream-url",    default="http://localhost:20024")
    parser.add_argument("--molmo2-url",       default="http://localhost:20025")
    parser.add_argument("--orient-url",       default="http://localhost:20034")
    parser.add_argument("--vace-url",         default="http://localhost:20034")
    parser.add_argument("--vace-timeout",     type=int, default=480,
                        help="VACE inference timeout in seconds (default: 480)")
    parser.add_argument("--sana-url",         default="http://127.0.0.1:30000")
    args = parser.parse_args()

    # Allow overriding local JSONL paths via CLI
    LOCAL_DATASET_PATHS["MindCube"] = args.mindcube_path
    LOCAL_DATASET_PATHS["VSIBench"] = args.vsibench_path

    work_dir  = Path(args.work_dir)
    trace_dir = Path(args.trace_dir)

    model_safe = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
    tools_tag  = "_".join(args.tools) if args.tools else "no_tools"
    model_tag  = f"{model_safe}_{tools_tag}"

    print(f"\n{'='*65}")
    print(f"  SPAgent Quick Eval")
    print(f"  Model    : {args.model}  (backend={args.model_backend})")
    print(f"  Tools    : {args.tools or ['(none)']}")
    print(f"  Datasets : {args.datasets}")
    print(f"  Limit    : {args.limit or 'defaults'}")
    print(f"  Tag      : {model_tag}")
    print(f"{'='*65}\n")

    # Build model (auto-detect backend from name, or use explicit --model-backend)
    try:
        model_instance = _build_model(args.model, args.model_backend, args.temperature, args.seed)
    except Exception as exc:
        print(f"[ERROR] Failed to build model '{args.model}' (backend={args.model_backend}): {exc}")
        sys.exit(1)

    tools = make_tools(args.tools, args)
    print(f"Tools: {[t.name for t in tools] if tools else ['(none)']}\n")

    agent = SPAgent(
        model=model_instance,
        tools=tools,
        max_workers=4,
        system_prompt=(
            "You are a helpful multimodal assistant. Analyze the image(s) carefully "
            "and answer the question. Use available tools when they help you perceive "
            "fine details, detect objects, or understand spatial relationships. "
            "Always put your final answer inside <answer></answer> tags."
        ),
        continuation_hint=GENERAL_VISION_CONTINUATION_HINT,
    )

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n{'─'*55}")
        print(f"  {ds_name}")
        limit = args.limit if args.limit is not None else DEFAULT_LIMIT.get(ds_name)

        # Explicit --data-path overrides the default local JSONL path
        local_path = args.data_path if args.data_path and len(args.datasets) == 1 else None

        try:
            ds = load_dataset(ds_name, limit, local_path=local_path)
        except Exception as exc:
            print(f"  [ERROR] Load failed: {exc}")
            all_results[ds_name] = {"error": str(exc)}
            continue

        print(f"  {len(ds.data)} rows loaded")

        # Local datasets (MindCube / VSIBench) use their own evaluation path
        # that mirrors examples/evaluation/evaluate_pi3x.py.
        if isinstance(ds, _LocalDataset):
            result = run_local_dataset(
                ds_name=ds_name,
                ds=ds,
                agent=agent,
                model_tag=model_tag,
                work_dir=work_dir,
                trace_dir=trace_dir,
                max_iterations=args.max_iterations,
                debug=args.debug,
            )
        else:
            result = run_dataset(
                ds_name=ds_name,
                ds=ds,
                agent=agent,
                model_tag=model_tag,
                work_dir=work_dir,
                trace_dir=trace_dir,
                judge_model=args.judge_model,
                max_iterations=args.max_iterations,
                nproc=args.nproc,
                debug=args.debug,
            )
        all_results[ds_name] = result

        scores = result.get("scores", {})
        if scores:
            print(f"\n  ✓ {ds_name} scores: {scores}")

    # Save summary
    summary_path = work_dir / f"{model_tag}_quick_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n\nSummary → {summary_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
