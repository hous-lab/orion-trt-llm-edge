#!/usr/bin/env bash
# Export the Orion vision module to ONNX (mixed-precision-marked: image backbone
# FP16, map/pts-bbox heads kept FP32; onnxsim-simplified with external data).
#
# Usage:
#   bash trtllm_7b/adzoo_orion/export_vision.sh <config> <checkpoint> [out_dir]
#   e.g. bash trtllm_7b/adzoo_orion/export_vision.sh \
#          $ORION_PROJECT_ROOT/adzoo/orion/configs/orion_stage3_fp16.py \
#          $ORION_PROJECT_ROOT/ckpts/Orion.pth
#
# Output: <out_dir>/<config-basename>.onnx (default under work_dirs/vision_onnx/).
# Next:   trtllm_7b/scripts/build_vision_engine.sh turns the ONNX into the engine
#         run_trt_eval.sh expects (work_dirs/vision_mixed_precision.engine).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${1:?usage: export_vision.sh <config> <checkpoint> [out_dir]}"
CHECKPOINT="${2:?usage: export_vision.sh <config> <checkpoint> [out_dir]}"
OUT_DIR="${3:-$REPO_ROOT/work_dirs/vision_onnx}"

for f in "$CONFIG" "$CHECKPOINT"; do
    [ -e "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
mkdir -p "$OUT_DIR"

PYTHON=${PYTHON:-python}
PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH" "$PYTHON" "$SCRIPT_DIR/export_vision_orion.py" \
    "$CONFIG" "$CHECKPOINT" "$OUT_DIR/"

echo ""
echo "ONNX exported to: $OUT_DIR"
echo "Next: bash trtllm_7b/scripts/build_vision_engine.sh"
