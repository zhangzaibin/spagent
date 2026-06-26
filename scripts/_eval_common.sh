#!/usr/bin/env bash
# ============================================================
#  _eval_common.sh  —  shared setup for SPAgent eval wrappers
# ============================================================
# This file is meant to be *sourced* (not executed) from the
# scripts/eval_*.sh wrappers. It:
#   1. Resolves the project root and cd's into it.
#   2. Applies common env-var defaults (model, iterations, dirs).
#   3. Defines localhost defaults for every tool-server URL.
#   4. Exposes two helpers: eval_limit_flag and eval_print_header.
#
# Every value uses `${VAR:-default}`, so any setting can be
# overridden from the command line, e.g.:
#   MODEL=gpt-4.1 LIMIT=100 bash scripts/eval_all_tools.sh
#   DETECTION_URL=http://10.7.8.94:20022 bash scripts/eval_detection_only.sh
#
# Wrappers that need a different *default* for a knob (e.g. a
# baseline with MAX_ITER=1) should set it BEFORE sourcing this
# file; the `:-` defaults below will then leave it untouched.
# ------------------------------------------------------------

set -euo pipefail

# ── Project root (this file lives in scripts/) ──────────────
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_COMMON_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Common run configuration ────────────────────────────────
MODEL="${MODEL:-gpt-4.1-mini}"
MODEL_BACKEND="${MODEL_BACKEND:-auto}"   # auto | gpt | qwen | qwen-vllm
MAX_ITER="${MAX_ITER:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
WORK_DIR="${WORK_DIR:-outputs/vlmeval_runs}"
TRACE_DIR="${TRACE_DIR:-outputs/spagent_traces}"
MINDCUBE_PATH="${MINDCUBE_PATH:-dataset/MindCube_data.jsonl}"

# ── Tool-server URLs (default to localhost) ─────────────────
# Ports match docs/Tool/TOOL_USING.md. Override for remote
# deployments, e.g.  PI3X_URL=http://10.8.131.51:31537 ...
DETECTION_URL="${DETECTION_URL:-http://localhost:20022}"    # GroundingDINO (zoom + localize)
SEGMENTATION_URL="${SEGMENTATION_URL:-http://localhost:20020}"  # SAM2
DEPTH_URL="${DEPTH_URL:-http://localhost:20019}"            # Depth Anything V2
PI3X_URL="${PI3X_URL:-http://localhost:20031}"              # Pi3X 3D reconstruction
PI3_URL="${PI3_URL:-http://localhost:20030}"               # Pi3 3D reconstruction
MOONDREAM_URL="${MOONDREAM_URL:-http://localhost:20024}"    # Moondream VLM
MOLMO2_URL="${MOLMO2_URL:-http://localhost:20025}"          # Molmo2 point grounding
ORIENT_URL="${ORIENT_URL:-http://localhost:20034}"          # Orient Anything V2
VACE_URL="${VACE_URL:-http://localhost:20034}"             # VACE local video gen
VACE_TIMEOUT="${VACE_TIMEOUT:-600}"                         # seconds

# ── Helper: echo "--limit N" only when LIMIT is non-empty ────
# Usage:  python quick_eval.py ... $(eval_limit_flag) ...
eval_limit_flag() {
    if [ -n "${LIMIT:-}" ]; then
        printf -- "--limit %s" "${LIMIT}"
    fi
}

# ── Helper: print a standard config banner ──────────────────
# Usage:  eval_print_header "eval_all_tools.sh" "Tools : pi3x zoom" "Pi3X URL : ${PI3X_URL}"
eval_print_header() {
    local label="$1"; shift
    echo "========================================================"
    echo "  SPAgent ${label}"
    echo "  Model     : ${MODEL}  (backend=${MODEL_BACKEND})"
    echo "  Datasets  : ${DATASETS:-(default)}"
    echo "  Limit     : ${LIMIT:-none}"
    echo "  Max iter  : ${MAX_ITER}"
    echo "  Work dir  : ${WORK_DIR}"
    local line
    for line in "$@"; do
        echo "  ${line}"
    done
    echo "========================================================"
    echo ""
}
