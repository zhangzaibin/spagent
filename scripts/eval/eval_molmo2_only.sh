#!/usr/bin/env bash
# ============================================================
#  eval_molmo2_only.sh
#  Run quick_eval.py with the Molmo2 point-grounding tool only.
#
#  Tools enabled:
#    molmo2 — molmo2_tool: reasoning-oriented point grounding
#             with annotated output overlays
#
#  Usage:
#    bash scripts/eval/eval_molmo2_only.sh
#    MODEL=gpt-4.1 LIMIT=50 bash scripts/eval/eval_molmo2_only.sh
#    DATASETS="VStarBench MMStar" bash scripts/eval/eval_molmo2_only.sh
# ============================================================

source "$(dirname "${BASH_SOURCE[0]}")/_eval_common.sh"

# ── Script-specific defaults ────────────────────────────────
DATASETS="${DATASETS:-MMStar}"
LIMIT="${LIMIT:-50}"

# Annotated point-grounding overlays are written here.
export MOLMO2_OUTPUT_DIR="${MOLMO2_OUTPUT_DIR:-outputs/molmo2_annotations}"

eval_print_header "eval_molmo2_only.sh" \
    "Tools     : molmo2" \
    "Molmo2    : ${MOLMO2_URL}" \
    "OutputDir : ${MOLMO2_OUTPUT_DIR}"

python scripts/quick_eval.py \
    --model            "${MODEL}" \
    --model-backend    "${MODEL_BACKEND}" \
    --datasets         ${DATASETS} \
    --tools            molmo2 \
    $(eval_limit_flag) \
    --max-iterations   "${MAX_ITER}" \
    --temperature      "${TEMPERATURE}" \
    --seed             "${SEED}" \
    --work-dir         "${WORK_DIR}" \
    --trace-dir        "${TRACE_DIR}" \
    --mindcube-path    "${MINDCUBE_PATH}" \
    --vsibench-path    "${VSIBENCH_PATH}" \
    --mmsi-path        "${MMSI_PATH}" \
    --omnispatial-path "${OMNISPATIAL_PATH}" \
    --molmo2-url       "${MOLMO2_URL}" \
    "$@"
