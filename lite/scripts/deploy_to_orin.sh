#!/bin/bash
# Sync the Orion-Lite deployment pipeline to a Jetson AGX Orin host.
#
# Usage:
#   ORION_HOST=<user@host> ORION_DIR=<path-on-orin> bash deploy_to_orin.sh
#   bash deploy_to_orin.sh                 # prompts for host if ORION_HOST unset
#
# Syncs this repository's lite/ tree (code only) to the Orin host; ONNX
# artifacts and checkpoints must be synced separately (see lite/README table
# in the top-level README "Get started" section).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ORIN_HOST="${ORION_HOST:-}"
ORIN_DIR="${ORION_DIR:-$HOME/orion-trt-llm-edge}"

if [ -z "$ORIN_HOST" ]; then
    echo "error: ORION_HOST is not set (e.g. export ORION_HOST=user@192.168.x.x)"
    exit 1
fi

echo "sync: ${REPO_ROOT}/lite -> ${ORIN_HOST}:${ORIN_DIR}/lite"

rsync -avz --partial \
    --exclude='__pycache__' \
    --exclude='*.engine' \
    --exclude='*.onnx' \
    --exclude='*.pth' \
    --exclude='*.npz' \
    --exclude='*.log' \
    "${REPO_ROOT}/lite/" \
    "${ORIN_HOST}:${ORIN_DIR}/lite/"

echo ""
echo "done. Next steps on the Orin:"
echo ""
echo "  # 1. sync artifacts (ONNX + student ckpt) separately, e.g.:"
echo "  rsync -avz --partial <repo>/lite/step2_onnx_export/output/*.onnx ${ORIN_HOST}:${ORIN_DIR}/lite/step2_onnx_export/output/"
echo ""
echo "  # 2. build engines (FP16 or INT8):"
echo "  cd ${ORIN_DIR}/lite/step3_trt_compile_orin"
echo "  python build_int8_trt10.py --onnx <vision.onnx> --calib <calib-dir> --output <vision.engine>  # INT8"
echo ""
echo "  # 3. benchmark:"
echo "  cd ${ORIN_DIR}/lite/step4_inference_eval_orin"
echo "  python real_data_benchmark.py --help"
echo ""
