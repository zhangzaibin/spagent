#!/usr/bin/env bash
# ============================================================
#  eval_pi3x_only.sh
#  Run quick_eval.py with the Pi3X 3D reconstruction tool only.
#
#  Tools enabled:
#    pi3x — pi3x_tool: 3D reconstruction from images, generates
#            point clouds and visualizations from custom viewpoints
#
#  Usage:
#    bash scripts/eval_pi3x_only.sh
#    MODEL=gpt-4.1 LIMIT=50 bash scripts/eval_pi3x_only.sh
#    DATASETS="MindCube VStarBench" bash scripts/eval_pi3x_only.sh
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

# ── Pi3X server URL ──────────────────────────────────────────
PI3X_URL="${PI3X_URL:-http://10.8.131.51:31927}"

# ── Build --limit flag (empty = omit) ───────────────────────
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

# ── Print config ─────────────────────────────────────────────
echo "========================================================"
echo "  SPAgent eval_pi3x_only.sh"
echo "  Model     : ${MODEL}  (backend=${MODEL_BACKEND})"
echo "  Datasets  : ${DATASETS}"
echo "  Limit     : ${LIMIT:-none}"
echo "  Max iter  : ${MAX_ITER}"
echo "  Work dir  : ${WORK_DIR}"
echo "  Tools     : pi3x"
echo "  Pi3X URL  : ${PI3X_URL}"
echo "========================================================"
echo ""

# ── Run ──────────────────────────────────────────────────────
python scripts/quick_eval.py \
    --model            "${MODEL}" \
    --model-backend    "${MODEL_BACKEND}" \
    --datasets         ${DATASETS} \
    --tools            pi3x \
    ${LIMIT_FLAG} \
    --max-iterations   "${MAX_ITER}" \
    --temperature      "${TEMPERATURE}" \
    --seed             "${SEED}" \
    --work-dir         "${WORK_DIR}" \
    --trace-dir        "${TRACE_DIR}" \
    --pi3x-url         "${PI3X_URL}" \
    "$@"
