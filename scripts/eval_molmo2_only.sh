#!/usr/bin/env bash
# ============================================================
#  eval_molmo2_only.sh
#  Run quick_eval.py with the Molmo2 pointing tool only.
#
#  Tools enabled:
#    molmo2 — molmo2_tool: reasoning-oriented point grounding
#             with annotated output overlays
#
#  Usage:
#    bash scripts/eval_molmo2_only.sh
#    MODEL=gpt-4.1 LIMIT=50 bash scripts/eval_molmo2_only.sh
#    DATASETS="VStarBench MMStar" bash scripts/eval_molmo2_only.sh
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

# ── Molmo2 server URL ────────────────────────────────────────
MOLMO2_URL="${MOLMO2_URL:-http://10.8.131.51:30094}"

# ── Molmo2 annotated image output dir ────────────────────────
export MOLMO2_OUTPUT_DIR="${MOLMO2_OUTPUT_DIR:-outputs/molmo2_annotations}"

# ── Build --limit flag (empty = omit) ───────────────────────
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

# ── Print config ─────────────────────────────────────────────
echo "========================================================"
echo "  SPAgent eval_molmo2_only.sh"
echo "  Model     : ${MODEL}  (backend=${MODEL_BACKEND})"
echo "  Datasets  : ${DATASETS}"
echo "  Limit     : ${LIMIT:-none}"
echo "  Max iter  : ${MAX_ITER}"
echo "  Work dir  : ${WORK_DIR}"
echo "  Tools     : molmo2"
echo "  Molmo2URL : ${MOLMO2_URL}"
echo "  OutputDir : ${MOLMO2_OUTPUT_DIR}"
echo "========================================================"
echo ""

# ── Run ──────────────────────────────────────────────────────
python scripts/quick_eval.py \
    --model            "${MODEL}" \
    --model-backend    "${MODEL_BACKEND}" \
    --datasets         ${DATASETS} \
    --tools            molmo2 \
    ${LIMIT_FLAG} \
    --max-iterations   "${MAX_ITER}" \
    --temperature      "${TEMPERATURE}" \
    --seed             "${SEED}" \
    --work-dir         "${WORK_DIR}" \
    --trace-dir        "${TRACE_DIR}" \
    --molmo2-url       "${MOLMO2_URL}" \
    "$@"
