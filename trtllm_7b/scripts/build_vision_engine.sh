#!/usr/bin/env bash
# Build the production vision engine (FP16) from the exported ONNX with trtexec.
#
# The ONNX carries the mixed-precision intent (backbone FP16-capable, map/pts heads
# kept FP32 by the fp16 wrap in the export script); trtexec --fp16 realizes it.
# Output lands at work_dirs/vision_mixed_precision.engine, the default path for
# run_trt_eval.sh / run_verify.sh / verify_hs_compare.py.
#
# Env overrides:
#   ORION_VISION_ONNX    input ONNX (default: newest *.onnx in work_dirs/vision_onnx/)
#   ORION_VISION_ENGINE  output engine path
#   TRTEXEC              trtexec binary (default: PATH or TensorRT-10.4 source tree)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ONNX="${ORION_VISION_ONNX:-$(ls -t "$REPO_ROOT"/work_dirs/vision_onnx/*.onnx 2>/dev/null | head -1 || true)}"
if [ -z "${ONNX:-}" ] || [ ! -f "$ONNX" ]; then
    echo "ERROR: no vision ONNX found - run trtllm_7b/adzoo_orion/export_vision.sh first"
    echo "       (or set ORION_VISION_ONNX=/path/to/onnx)"
    exit 1
fi

ENGINE="${ORION_VISION_ENGINE:-$REPO_ROOT/work_dirs/vision_mixed_precision.engine}"
TRTEXEC="${TRTEXEC:-$(command -v trtexec || echo /usr/src/TensorRT-10.4.0.26/targets/x86_64-linux-gnu/bin/trtexec)}"
if [ ! -x "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at: $TRTEXEC"
    echo "       set TRTEXEC=/path/to/trtexec"
    exit 1
fi

mkdir -p "$(dirname "$ENGINE")"
echo "ONNX:    $ONNX"
echo "Engine:  $ENGINE"
echo "trtexec: $TRTEXEC"
echo ""
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16

echo ""
echo "Done: $ENGINE ($(du -h "$ENGINE" | cut -f1))"
echo "Next: bash trtllm_7b/run_trt_eval.sh --llm-precision w8abf16_patched"
