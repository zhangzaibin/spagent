#!/usr/bin/env bash
# ============================================================
#  eval_pi3x_only.sh
#  Run quick_eval.py with the Pi3X 3D reconstruction tool only.
#
#  Tools enabled:
#    pi3x — pi3x_tool: 3D reconstruction from images, generates
#            point clouds and visualizations from custom viewpoints
#
#  Default datasets  : MindCube (50/category) + VSIBench
#  Default prompt    : spatial  (SPATIAL_3D_ROLE + SPATIAL_3D_WORKFLOW)
#  Default per-cat   : 50  (samples per MindCube task category)
#
#  Usage:
#    bash scripts/eval_pi3x_only.sh
#    MODEL=gpt-4.1 bash scripts/eval_pi3x_only.sh
#    DATASETS="MindCube" PER_CATEGORY=100 bash scripts/eval_pi3x_only.sh
#    DATASETS="VStarBench" PROMPT=general LIMIT=50 bash scripts/eval_pi3x_only.sh
#
#  VSI-Bench data prep (run once before evaluating VSIBench):
#    python spagent/utils/download_vsibench.py
#  (Requires raw dataset at dataset/VSI-Bench/ from HF nyu-visionx/VSI-Bench)
# ============================================================
set -euo pipefail

# ── Project root ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Configurable via env-var overrides ──────────────────────
MODEL="${MODEL:-gpt-4.1}"
MODEL_BACKEND="${MODEL_BACKEND:-auto}"
DATASETS="${DATASETS:-MindCube VSIBench}"
PROMPT="${PROMPT:-spatial}"
PER_CATEGORY="${PER_CATEGORY:-1000}"
LIMIT="${LIMIT:-}"
MAX_ITER="${MAX_ITER:-3}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
WORK_DIR="${WORK_DIR:-outputs/vlmeval_runs}"
TRACE_DIR="${TRACE_DIR:-outputs/spagent_traces}"

# ── Pi3X server URL ──────────────────────────────────────────
PI3X_URL="${PI3X_URL:-http://10.7.8.94:20031}"

# ── Build optional flags ─────────────────────────────────────
# --limit is only forwarded when explicitly set (non-empty).
# For local datasets (MindCube / VSIBench), --per-category takes precedence.
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

PER_CATEGORY_FLAG=""
if [ -n "${PER_CATEGORY}" ]; then
    PER_CATEGORY_FLAG="--per-category ${PER_CATEGORY}"
fi

# ── Print config ─────────────────────────────────────────────
echo "========================================================"
echo "  SPAgent eval_pi3x_only.sh"
echo "  Model     : ${MODEL}  (backend=${MODEL_BACKEND})"
echo "  Datasets  : ${DATASETS}"
echo "  Prompt    : ${PROMPT}"
echo "  Per-cat   : ${PER_CATEGORY:-none}"
echo "  Limit     : ${LIMIT:-none}"
echo "  Max iter  : ${MAX_ITER}"
echo "  Video frm : ${NUM_VIDEO_FRAMES}"
echo "  Work dir  : ${WORK_DIR}"
echo "  Tools     : pi3x"
echo "  Pi3X URL  : ${PI3X_URL}"
echo "========================================================"
echo ""

# ── VSI-Bench data check ──────────────────────────────────────
if echo "${DATASETS}" | grep -qw "VSIBench"; then
    VSIBENCH_JSONL="${PROJECT_ROOT}/dataset/VSI_Bench.jsonl"
    if [ ! -s "${VSIBENCH_JSONL}" ]; then
        echo "  [WARN] VSI_Bench.jsonl is missing or empty: ${VSIBENCH_JSONL}"
        echo "  Run: python spagent/utils/download_vsibench.py"
        echo "  (Needs raw data at dataset/VSI-Bench/ from HF nyu-visionx/VSI-Bench)"
        echo ""
    fi
fi

# ── Run ──────────────────────────────────────────────────────
python scripts/quick_eval.py \
    --model            "${MODEL}" \
    --model-backend    "${MODEL_BACKEND}" \
    --datasets         ${DATASETS} \
    --tools            pi3x \
    --prompt           "${PROMPT}" \
    ${PER_CATEGORY_FLAG} \
    ${LIMIT_FLAG} \
    --max-iterations   "${MAX_ITER}" \
    --num-video-frames "${NUM_VIDEO_FRAMES}" \
    --temperature      "${TEMPERATURE}" \
    --seed             "${SEED}" \
    --work-dir         "${WORK_DIR}" \
    --trace-dir        "${TRACE_DIR}" \
    --pi3x-url         "${PI3X_URL}" \
    "$@"
