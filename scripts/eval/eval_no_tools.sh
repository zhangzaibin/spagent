#!/usr/bin/env bash
# ============================================================
#  eval_no_tools.sh
#  Baseline VLM evaluation with NO tools enabled.
#
#  Usage:
#    bash scripts/eval/eval_no_tools.sh
#    MODEL=gpt-4.1 DATASETS="MindCube MMStar BLINK" bash scripts/eval/eval_no_tools.sh
# ============================================================

# Baseline runs a single pass (no tool iterations) by default.
MAX_ITER="${MAX_ITER:-1}"

source "$(dirname "${BASH_SOURCE[0]}")/_eval_common.sh"

# ── Script-specific defaults ────────────────────────────────
DATASETS="${DATASETS:-MindCube VSIBench MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI MMBench_dev_en RealWorldQA ScienceQA_VAL HRBench4K HRBench8K MathVerse_MINI WeMath LogicVista MMMU_Pro_10c DynaMath}"
LIMIT="${LIMIT:-100000}"

eval_print_header "eval_no_tools.sh  (baseline, no tools)"

python scripts/quick_eval.py \
    --model          "${MODEL}" \
    --model-backend  "${MODEL_BACKEND}" \
    --datasets       ${DATASETS} \
    $(eval_limit_flag) \
    --max-iterations "${MAX_ITER}" \
    --temperature    "${TEMPERATURE}" \
    --seed           "${SEED}" \
    --work-dir       "${WORK_DIR}" \
    --trace-dir      "${TRACE_DIR}" \
    --mindcube-path    "${MINDCUBE_PATH}" \
    --vsibench-path    "${VSIBENCH_PATH}" \
    --mmsi-path        "${MMSI_PATH}" \
    --omnispatial-path "${OMNISPATIAL_PATH}" \
    "$@"
