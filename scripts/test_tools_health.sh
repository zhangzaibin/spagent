#!/usr/bin/env bash
# ============================================================
#  test_tools_health.sh
#  Quick health-check for all currently deployed tool servers.
#
#  Step 1: curl /health and /test on every HTTP server
#  Step 2: real inference test for pi3x and molmo2 via test_tool.py
#  Step 3: veo mock test (no API key needed)
#
#  Usage:
#    bash scripts/test_tools_health.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Server URLs (localhost defaults; override for remote, e.g.
#    PI3X_URL=http://10.8.131.51:31537 bash scripts/test_tools_health.sh) ──
PI3X_URL="${PI3X_URL:-http://localhost:20031}"
DETECTION_URL="${DETECTION_URL:-http://localhost:20022}"
SEGMENTATION_URL="${SEGMENTATION_URL:-http://localhost:20020}"
DEPTH_URL="${DEPTH_URL:-http://localhost:20019}"
MOLMO2_URL="${MOLMO2_URL:-http://localhost:20025}"
VACE_URL="${VACE_URL:-http://localhost:20034}"

TEST_IMAGE="${TEST_IMAGE:-assets/dog.jpeg}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
TIMEOUT="${TIMEOUT:-10}"    # seconds per curl /health /test request
VACE_TIMEOUT="${VACE_TIMEOUT:-600}"  # 5 minutes for VACE video generation

PASS=0
FAIL=0
SKIP=0

# ── Helpers ───────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✅ PASS${NC}  $*"; ((PASS++)); }
fail() { echo -e "  ${RED}❌ FAIL${NC}  $*"; ((FAIL++)); }
skip() { echo -e "  ${YELLOW}⏭  SKIP${NC}  $*"; ((SKIP++)); }

check_http() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X "${method}" --max-time "${TIMEOUT}" "${url}" 2>/dev/null)
    if [ "${http_code}" = "200" ]; then
        ok "${name}  →  ${url}  (HTTP ${http_code})"
    else
        fail "${name}  →  ${url}  (HTTP ${http_code:-connection failed})"
    fi
}

# Run python test_tool.py and check for ✅ anywhere in output
run_infer_test() {
    local label="$1"; shift
    local out
    out=$(python test/test_tool.py "$@" 2>&1)
    echo "${out}" | tail -5
    if echo "${out}" | grep -q "✅"; then
        ok "${label}"
    else
        fail "${label}  (see above)"
    fi
}

# ── Banner ────────────────────────────────────────────────────
echo "========================================================"
echo "  SPAgent Tool Health Check"
echo "  Test image   : ${TEST_IMAGE}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo "  Timeout      : ${TIMEOUT}s (curl)  /  ${VACE_TIMEOUT}s (VACE inference)"
echo "========================================================"
echo ""

# ──────────────────────────────────────────────────────────────
# 1. Pi3X
# ──────────────────────────────────────────────────────────────
echo "── Pi3X ─────────────────────────────────────────────────"
check_http "pi3x  /health" "${PI3X_URL}/health"
check_http "pi3x  /test  " "${PI3X_URL}/test"
if [ -f "${TEST_IMAGE}" ]; then
    echo "  Running inference test..."
    run_infer_test "pi3x  inference" \
        --tool pi3x \
        --image "${TEST_IMAGE}" \
        --azimuth 45 --elevation 0 \
        --server_url "${PI3X_URL}" \
        --output_dir "${OUTPUT_DIR}"
else
    skip "pi3x  inference  (test image not found: ${TEST_IMAGE})"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# 2. Detection (GroundingDINO)
# ──────────────────────────────────────────────────────────────
echo "── Detection (GroundingDINO) ────────────────────────────"
check_http "detection  /health" "${DETECTION_URL}/health"
check_http "detection  /test  " "${DETECTION_URL}/test"
if [ -f "${TEST_IMAGE}" ]; then
    echo "  Running inference test..."
    run_infer_test "detection  inference" \
        --tool detection \
        --image "${TEST_IMAGE}" \
        --prompt "dog" \
        --server_url "${DETECTION_URL}" \
        --output_dir "${OUTPUT_DIR}"
else
    skip "detection  inference  (test image not found: ${TEST_IMAGE})"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# 3. Segmentation (SAM2)
# ──────────────────────────────────────────────────────────────
echo "── Segmentation (SAM2) ──────────────────────────────────"
check_http "segmentation  /health" "${SEGMENTATION_URL}/health"
check_http "segmentation  /test  " "${SEGMENTATION_URL}/test"
if [ -f "${TEST_IMAGE}" ]; then
    echo "  Running inference test..."
    run_infer_test "segmentation  inference" \
        --tool segmentation \
        --image "${TEST_IMAGE}" \
        --server_url "${SEGMENTATION_URL}" \
        --output_dir "${OUTPUT_DIR}"
else
    skip "segmentation  inference  (test image not found: ${TEST_IMAGE})"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# 4. Depth (Depth Anything V2)
# ──────────────────────────────────────────────────────────────
echo "── Depth (Depth Anything V2) ────────────────────────────"
check_http "depth  /health" "${DEPTH_URL}/health"
check_http "depth  /test  " "${DEPTH_URL}/test"
if [ -f "${TEST_IMAGE}" ]; then
    echo "  Running inference test..."
    run_infer_test "depth  inference" \
        --tool depth \
        --image "${TEST_IMAGE}" \
        --server_url "${DEPTH_URL}" \
        --output_dir "${OUTPUT_DIR}"
else
    skip "depth  inference  (test image not found: ${TEST_IMAGE})"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# 5. Molmo2
# ──────────────────────────────────────────────────────────────
echo "── Molmo2 ───────────────────────────────────────────────"
check_http "molmo2  /health" "${MOLMO2_URL}/health"
check_http "molmo2  /test  " "${MOLMO2_URL}/test"
if [ -f "${TEST_IMAGE}" ]; then
    echo "  Running inference test (task=point → saves annotated image)..."
    run_infer_test "molmo2  inference" \
        --tool molmo2 \
        --image "${TEST_IMAGE}" \
        --task point \
        --prompt "Point to the dog." \
        --save_annotated \
        --server_url "${MOLMO2_URL}" \
        --output_dir "${OUTPUT_DIR}"
else
    skip "molmo2  inference  (test image not found: ${TEST_IMAGE})"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# 6. VACE
# ──────────────────────────────────────────────────────────────
echo "── VACE ─────────────────────────────────────────────────"
check_http "vace  /health" "${VACE_URL}/health"
check_http "vace  /test  " "${VACE_URL}/test" "POST"   # VACE /test is POST
if [ -f "${TEST_IMAGE}" ]; then
    echo "  Running inference test (timeout: ${VACE_TIMEOUT}s, this may take a few minutes)..."
    out=$(timeout "${VACE_TIMEOUT}" python test/test_tool.py \
        --tool vace \
        --image "${TEST_IMAGE}" \
        --prompt "The dog slowly turns its head and looks around." \
        --server_url "${VACE_URL}" \
        --output_dir "${OUTPUT_DIR}" 2>&1)
    echo "${out}" | tail -5
    if echo "${out}" | grep -q "✅"; then
        ok "vace  inference"
    else
        fail "vace  inference  (see above)"
    fi
else
    skip "vace  inference  (test image not found: ${TEST_IMAGE})"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# 7. Veo (Gemini API)
# ──────────────────────────────────────────────────────────────
echo "── Veo (Gemini API) ─────────────────────────────────────"
echo "  Running mock test (no real API call)..."
run_infer_test "veo  mock" \
    --tool veo \
    --image dummy \
    --prompt "A dog running on a beach." \
    --use_mock \
    --output_dir "${OUTPUT_DIR}"
echo ""

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
echo "========================================================"
echo -e "  ${GREEN}PASS${NC}: ${PASS}   ${RED}FAIL${NC}: ${FAIL}   ${YELLOW}SKIP${NC}: ${SKIP}"
echo "========================================================"
if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
