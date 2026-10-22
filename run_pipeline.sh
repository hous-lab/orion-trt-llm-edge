#!/usr/bin/env bash
# Orion-Lite deployment pipeline driver.
#
# Drives the Lite pipeline from a fresh clone: guard checks (fail-fast on
# missing upstream artifacts) -> optional distillation -> ONNX export ->
# TRT engine build -> benchmark. Each step is idempotent; step toggles:
#   RUN_STEP1=1  also run distillation (teacher data collection + student training)
#   RUN_STEP2=0  skip ONNX export
#   RUN_STEP3=0  skip engine build
#   RUN_STEP4=0  skip benchmark
#   PLATFORM=x86|orin
#
# Usage:
#   source trtllm_7b/env_template.sh    # or export the ORION_* vars yourself
#   bash run_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RUN_STEP1="${RUN_STEP1:-0}"
RUN_STEP2="${RUN_STEP2:-1}"
RUN_STEP3="${RUN_STEP3:-1}"
RUN_STEP4="${RUN_STEP4:-1}"
PLATFORM="${PLATFORM:-x86}"
IMG_SIZE="${IMG_SIZE:-512}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/lite/artifacts}"
PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Step 0: guards (fail-fast, no work)
# ---------------------------------------------------------------------------
echo "=== step 0: guards ==="
missing=0
fail() { echo "  MISSING: $1"; missing=1; }

: "${ORION_LITE_PROJECT_ROOT:=}"
: "${ORION_PROJECT_ROOT:=}"
: "${FUSED_CKPT:=}"

[ -n "$ORION_LITE_PROJECT_ROOT" ] || fail "ORION_LITE_PROJECT_ROOT unset - clone the upstream training repo (Lite variant), see ckpts/README.md"
[ -n "$ORION_PROJECT_ROOT" ] || fail "ORION_PROJECT_ROOT unset - see ckpts/README.md"
[ -d "$ORION_LITE_PROJECT_ROOT/adzoo" ] || fail "$ORION_LITE_PROJECT_ROOT/adzoo not found (not a training-repo clone?)"
[ -d "$ORION_PROJECT_ROOT/adzoo" ] || fail "$ORION_PROJECT_ROOT/adzoo not found (not a training-repo clone?)"

QFORMER="${ORION_QFORMER_PATH:-$REPO_ROOT/ckpts/hf_models/pretrain_qformer}"
[ -d "$QFORMER" ] || fail "QFormer not found at $QFORMER - download per ckpts/README.md"

if [ "$RUN_STEP1" = "1" ]; then
    [ -f "$ORION_PROJECT_ROOT/ckpts/Orion.pth" ] || fail "Orion.pth missing - download per ckpts/README.md (teacher, required for distillation)"
    [ -d "data/bench2drive" ] || fail "data/bench2drive missing - see data/README.md (needed for distill data collection)"
fi

if [ "$RUN_STEP2" = "1" ]; then
    [ -n "$FUSED_CKPT" ] || fail "FUSED_CKPT unset - the fused distillation checkpoint (contains upstream weights, not redistributed); regenerate via RUN_STEP1=1, see ckpts/README.md"
    [ -f "$FUSED_CKPT" ] || fail "FUSED_CKPT=$FUSED_CKPT not found"
fi

if [ "$missing" = "1" ]; then
    echo ""
    echo "step 0 failed: fix the items above and re-run."
    exit 1
fi
echo "  all guards passed"

# ---------------------------------------------------------------------------
# Step 1 (optional, RUN_STEP1=1): teacher data collection + student training
# ---------------------------------------------------------------------------
if [ "$RUN_STEP1" = "1" ]; then
    echo "=== step 1: distillation ==="
    mkdir -p distill_data
    ( cd "$ORION_LITE_PROJECT_ROOT" && \
      "$PYTHON" "$REPO_ROOT/lite/step1_distill/run_orion_collect.py" \
        --config adzoo/orion/configs/orion_stage3_infer.py \
        --checkpoint "$ORION_PROJECT_ROOT/ckpts/Orion.pth" \
        --save-dir "$REPO_ROOT/distill_data/train" \
        --split train --fp16 )
    ( cd lite/step1_distill && \
      "$PYTHON" train_student_with_orion_loss.py \
        --train-dir "$REPO_ROOT/distill_data/train" \
        --val-dir "$REPO_ROOT/distill_data/val" \
        --orion-ckpt "$ORION_PROJECT_ROOT/ckpts/Orion.pth" \
        --use-feature-mimic-loss --epochs 20 --batch-size 64 )
fi

# ---------------------------------------------------------------------------
# Step 2: ONNX export (vision module, platform independent)
# ---------------------------------------------------------------------------
if [ "$RUN_STEP2" = "1" ]; then
    echo "=== step 2: ONNX export (img_size=$IMG_SIZE) ==="
    mkdir -p "$ARTIFACT_DIR"
    ( cd "$ORION_LITE_PROJECT_ROOT" && \
      ORION_QFORMER_PATH="$QFORMER" \
      "$PYTHON" "$REPO_ROOT/lite/step2_onnx_export/export_vision_onnx.py" \
        adzoo/orion/configs/orion_distill_new_agent_fused.py \
        "$FUSED_CKPT" \
        "$ARTIFACT_DIR" \
        --img-size "$IMG_SIZE" \
        --cfg-options model.img_backbone.img_size="$IMG_SIZE" \
        --save-test-data )
fi

# ---------------------------------------------------------------------------
# Step 3: TensorRT engine build (FP16; INT8 optional via BUILD_INT8=1)
# ---------------------------------------------------------------------------
ONNX_FILE="$ARTIFACT_DIR/orion_lite_vision_sim_mixed.onnx"

if [ "$RUN_STEP3" = "1" ]; then
    echo "=== step 3: engine build (platform=$PLATFORM) ==="
    if [ "$PLATFORM" = "orin" ]; then
        STEP3_DIR="$REPO_ROOT/lite/step3_trt_compile_orin"
    else
        STEP3_DIR="$REPO_ROOT/lite/step3_trt_compile_x86"
    fi
    if [ "${BUILD_INT8:-0}" = "1" ]; then
        "$PYTHON" "$STEP3_DIR/build_int8_trt10.py" \
            --onnx "$ONNX_FILE" \
            --calib-dir "$ARTIFACT_DIR/calibration_samples" \
            --output "$ARTIFACT_DIR/vision_int8_calibrated_trt10.engine"
    else
        "$PYTHON" "$STEP3_DIR/build_fp16.py" \
            --onnx "$ONNX_FILE" \
            --output "$ARTIFACT_DIR/vision_fp16.engine"
    fi
fi

# ---------------------------------------------------------------------------
# Step 4: real-data benchmark + per-layer profile
# ---------------------------------------------------------------------------
if [ "$RUN_STEP4" = "1" ]; then
    echo "=== step 4: benchmark (platform=$PLATFORM) ==="
    if [ "$PLATFORM" = "orin" ]; then
        STEP4_DIR="$REPO_ROOT/lite/step4_inference_eval_orin"
    else
        STEP4_DIR="$REPO_ROOT/lite/step4_inference_eval_x86"
    fi
    ENGINE_FILE="$ARTIFACT_DIR/vision_fp16.engine"
    [ "${BUILD_INT8:-0}" = "1" ] && ENGINE_FILE="$ARTIFACT_DIR/vision_int8_calibrated_trt10.engine"
    "$PYTHON" "$STEP4_DIR/real_data_benchmark.py" \
        --test-data-dir "$ARTIFACT_DIR/test_data" \
        --engine-dir "$ARTIFACT_DIR"
    "$PYTHON" "$STEP4_DIR/profile_layers.py" \
        --engine "$ENGINE_FILE" \
        --test-data-dir "$ARTIFACT_DIR/test_data" \
        --output "$ARTIFACT_DIR/layer_profile.json"
fi

echo "pipeline finished."
