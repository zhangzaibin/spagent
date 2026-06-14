#!/usr/bin/env bash
# ============================================================
#  eval_detection_pi3x.sh
#  Run quick_eval.py with both GroundingDINO detection tools
#  and Pi3X 3D reconstruction tool.
#
#  Tools enabled:
#    zoom      — zoom_object_tool: crop close-up for attribute inspection
#    localize  — localize_object_tool: draw bboxes for spatial/counting
#    pi3x      — pi3x_tool: 3D reconstruction with custom viewpoints
#
#  Usage:
#    bash scripts/eval_detection_pi3x.sh
#    MODEL=gpt-4.1 LIMIT=50 bash scripts/eval_detection_pi3x.sh
#    DATASETS="MindCube VStarBench" bash scripts/eval_detection_pi3x.sh
#    TOOLS="zoom pi3x" bash scripts/eval_detection_pi3x.sh   # subset of tools
# ============================================================
set -euo pipefail

# ── Project root ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Configurable via env-var overrides ──────────────────────
MODEL="${MODEL:-gpt-4.1}"
MODEL_BACKEND="${MODEL_BACKEND:-auto}"
DATASETS="${DATASETS:-MMStar}"
LIMIT="${LIMIT:-50}"
MAX_ITER="${MAX_ITER:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
WORK_DIR="${WORK_DIR:-outputs/vlmeval_runs}"
TRACE_DIR="${TRACE_DIR:-outputs/spagent_traces}"

# ── Tool server URLs ─────────────────────────────────────────
DETECTION_URL="${DETECTION_URL:-http://10.7.8.94:20022}"
PI3X_URL="${PI3X_URL:-http://10.8.131.51:31927}"

# ── Tools to enable (space-separated) ────────────────────────
# zoom     = zoom_object_tool
# localize = localize_object_tool
# pi3x     = pi3x_tool
TOOLS="${TOOLS:-zoom localize pi3x}"

# ── Build --limit flag (empty = omit) ───────────────────────
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

# ── Print config ─────────────────────────────────────────────
echo "========================================================"
echo "  SPAgent eval_detection_pi3x.sh"
echo "  Model     : ${MODEL}  (backend=${MODEL_BACKEND})"
echo "  Datasets  : ${DATASETS}"
echo "  Limit     : ${LIMIT:-none}"
echo "  Max iter  : ${MAX_ITER}"
echo "  Work dir  : ${WORK_DIR}"
echo "  Tools     : ${TOOLS}"
echo "  DetURL    : ${DETECTION_URL}"
echo "  Pi3X URL  : ${PI3X_URL}"
echo "========================================================"
echo ""

# ── Run ──────────────────────────────────────────────────────
python scripts/quick_eval.py \
    --model            "${MODEL}" \
    --model-backend    "${MODEL_BACKEND}" \
    --datasets         ${DATASETS} \
    --tools            ${TOOLS} \
    ${LIMIT_FLAG} \
    --max-iterations   "${MAX_ITER}" \
    --temperature      "${TEMPERATURE}" \
    --seed             "${SEED}" \
    --work-dir         "${WORK_DIR}" \
    --trace-dir        "${TRACE_DIR}" \
    --detection-url    "${DETECTION_URL}" \
    --pi3x-url         "${PI3X_URL}" \
    "$@"
