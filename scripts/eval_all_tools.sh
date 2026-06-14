#!/usr/bin/env bash
# ============================================================
#  eval_all_tools.sh
#  Run quick_eval.py with the full tool stack:
#    Pi3X | GroundingDINO | SAM2 | Depth Anything
#    Molmo2 | Veo | VACE
#  (Orient Anything V2 / Sana temporarily disabled)
#
#  Usage:
#    bash scripts/eval_all_tools.sh                         # defaults below
#    MODEL=gpt-4.1 DATASETS="MindCube MMStar" bash scripts/eval_all_tools.sh
#
#  Tool name mapping (quick_eval.py --tools):
#    pi3x            → Pi3XTool             (port 20031)
#    detection       → ObjectDetectionTool   (port 20022)
#    segmentation    → SegmentationTool      (port 20020)
#    depth           → DepthEstimationTool   (port 20019)
#    molmo2          → Molmo2Tool            (port 20025)
#    veo             → VeoTool               (Gemini API)
#    vace            → VaceTool              (port 20035)
#  [disabled]
#    orient          → OrientAnythingV2Tool  (port 20034)
#    sana            → SanaTool              (port 30000)
# ============================================================
set -euo pipefail

# ── Project root ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Configurable via env-var overrides ──────────────────────
MODEL="${MODEL:-gpt-4.1-mini}"
MODEL_BACKEND="${MODEL_BACKEND:-auto}"
DATASETS="${DATASETS:-MindCube VSIBench MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI MMBench_dev_en RealWorldQA ScienceQA_VAL HRBench4K HRBench8K MathVerse_MINI WeMath LogicVista MMMU_Pro_10c DynaMath}"
LIMIT="${LIMIT:-50}"          # samples per dataset (empty string = no limit)
MAX_ITER="${MAX_ITER:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
WORK_DIR="${WORK_DIR:-outputs/vlmeval_runs}"
TRACE_DIR="${TRACE_DIR:-outputs/spagent_traces}"
MINDCUBE_PATH="${MINDCUBE_PATH:-dataset/MindCube_data.jsonl}"

# ── Tool server URLs ─────────────────────────────────────────
PI3X_URL="${PI3X_URL:-http://10.8.131.51:31537}"
DETECTION_URL="${DETECTION_URL:-http://10.7.8.94:20022}"
SEGMENTATION_URL="${SEGMENTATION_URL:-http://10.7.8.94:20020}"
DEPTH_URL="${DEPTH_URL:-http://10.7.8.94:20019}"
MOLMO2_URL="${MOLMO2_URL:-http://10.8.131.51:31108}"
VACE_URL="${VACE_URL:-http://10.8.131.51:30014}"
VACE_TIMEOUT="${VACE_TIMEOUT:-1000}"   # seconds (8 min); override e.g. VACE_TIMEOUT=600
# Veo uses Gemini API — no URL needed, set GOOGLE_API_KEY in your environment

# ── Build --limit flag (empty = omit) ───────────────────────
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

# ── Print config ─────────────────────────────────────────────
echo "========================================================"
echo "  SPAgent eval_all_tools.sh"
echo "  Model     : ${MODEL}  (backend=${MODEL_BACKEND})"
echo "  Datasets  : ${DATASETS}"
echo "  Limit     : ${LIMIT:-none}"
echo "  Max iter  : ${MAX_ITER}"
echo "  Work dir  : ${WORK_DIR}"
echo "========================================================"
echo ""

# ── Run ──────────────────────────────────────────────────────
python scripts/quick_eval.py \
    --model            "${MODEL}" \
    --model-backend    "${MODEL_BACKEND}" \
    --datasets         ${DATASETS} \
    --tools            pi3x detection segmentation depth molmo2 vace \
    ${LIMIT_FLAG} \
    --max-iterations   "${MAX_ITER}" \
    --temperature      "${TEMPERATURE}" \
    --seed             "${SEED}" \
    --work-dir         "${WORK_DIR}" \
    --trace-dir        "${TRACE_DIR}" \
    --mindcube-path    "${MINDCUBE_PATH}" \
    --pi3x-url         "${PI3X_URL}" \
    --detection-url    "${DETECTION_URL}" \
    --segmentation-url "${SEGMENTATION_URL}" \
    --depth-url        "${DEPTH_URL}" \
    --molmo2-url       "${MOLMO2_URL}" \
    --vace-url         "${VACE_URL}" \
    --vace-timeout     "${VACE_TIMEOUT}" \
    "$@"
