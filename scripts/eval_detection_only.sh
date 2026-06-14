#!/usr/bin/env bash
# ============================================================
#  eval_detection_only.sh
#  Run quick_eval.py with GroundingDINO tools on:
#    VStarBench | HRBench4K | HRBench8K
#
#  Tools enabled:
#    zoom      — zoom_object_tool: crop close-up for attribute inspection (color, texture…)
#    localize  — localize_object_tool: draw bboxes on full image for spatial/counting
#
#  Usage:
#    bash scripts/eval_detection_only.sh
#    MODEL=gpt-4.1 LIMIT=50 bash scripts/eval_detection_only.sh
#    TOOLS="zoom" bash scripts/eval_detection_only.sh          # zoom only
#    TOOLS="zoom localize" bash scripts/eval_detection_only.sh # both (default)
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

# ── Tool server URL ──────────────────────────────────────────
DETECTION_URL="${DETECTION_URL:-http://10.7.8.94:20022}"

# ── Tools to enable (space-separated) ────────────────────────
# zoom     = zoom_object_tool   (crop close-up for attribute questions)
# localize = localize_object_tool (bbox annotation for spatial/counting)
TOOLS="${TOOLS:-zoom localize}"

# ── Build --limit flag (empty = omit) ───────────────────────
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

# ── Print config ─────────────────────────────────────────────
echo "========================================================"
echo "  SPAgent eval_detection_only.sh"
echo "  Model     : ${MODEL}  (backend=${MODEL_BACKEND})"
echo "  Datasets  : ${DATASETS}"
echo "  Limit     : ${LIMIT:-none}"
echo "  Max iter  : ${MAX_ITER}"
echo "  Work dir  : ${WORK_DIR}"
echo "  Tools     : ${TOOLS}"
echo "  DetURL    : ${DETECTION_URL}"
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
    "$@"
