#!/usr/bin/env bash
# ============================================================================
# Orion TRT-LLM 引擎快速验证脚本
#
# 在单样本上运行快速推理，验证 vision engine + LLM engine 的输出正确性，
# 确认 full_hidden_states 可正常提取。
#
# 用法:
#   bash run_verify.sh                    # 默认 w8abf16 engine
#   LLM_PRECISION=bfp16 bash run_verify.sh
# ============================================================================

set -euo pipefail

# ---- 项目根路径 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${ORION_PROJECT_ROOT:?error: ORION_PROJECT_ROOT is not set (see ckpts/README.md)}"

# ---- 路径配置 ----
LLM_PRECISION="${LLM_PRECISION:-w8abf16}"
ENGINE_STEM="${LLM_PRECISION%_patched}"
CONFIG="${ORION_CONFIG:-$ORION_PROJECT_ROOT/adzoo/orion/configs/orion_stage3.py}"
TOKENIZER="${ORION_QFORMER_PATH:-$REPO_ROOT/ckpts/hf_models/pretrain_qformer/}"
VISION_ENGINE="${ORION_VISION_ENGINE:-$REPO_ROOT/work_dirs/vision_mixed_precision.engine}"
# Engine dir follows run_convert_and_build.sh output naming; _patched maps to the
# same engine (all release builds use the vocab-patched checkpoint).
LLM_ENGINE="${ORION_LLM_ENGINE:-$REPO_ROOT/work_dirs/llm_safetensor/x86_1gpu_${ENGINE_STEM}_fp32attn}"
SAVE_PATH="${ORION_VERIFY_OUT:-$REPO_ROOT/work_dirs/verify_$(date +%Y%m%d_%H%M%S)/}"

# ---- 检查文件 ----
MISSING=0
for f in "$CONFIG" "$VISION_ENGINE" "${TOKENIZER}/config.json"; do
    if [ ! -e "$f" ]; then
        echo "  [MISSING] $f"
        MISSING=1
    fi
done
if [ ! -d "$LLM_ENGINE" ] || [ ! -f "${LLM_ENGINE}/rank0.engine" ]; then
    echo "  [MISSING] LLM engine: $LLM_ENGINE"
    MISSING=1
fi
if [ $MISSING -eq 1 ]; then
    exit 1
fi

mkdir -p "$SAVE_PATH"

# ---- 环境变量 ----
export PYTHONPATH="${ORION_PROJECT_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export TRT_HOME="${TRT_HOME:-/usr/src/TensorRT-10.4.0.26}"
export LD_LIBRARY_PATH="${TRT_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3/envs/orion}"
export ORION_QFORMER_PATH="$TOKENIZER"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export QT_QPA_PLATFORM="offscreen"

PYTHON="$CONDA_PREFIX/bin/python"

echo "========================================================================"
echo "  Orion TRT-LLM 快速验证"
echo "========================================================================"
echo "  Vision:  $VISION_ENGINE"
echo "  LLM:     $LLM_ENGINE  (precision: $LLM_PRECISION)"
echo "  Save:    $SAVE_PATH"
echo "========================================================================"
echo ""

$PYTHON "$REPO_ROOT/trtllm_7b/verify_hidden_states.py" \
    --engine_dir "$LLM_ENGINE" \
    --output_dir "$SAVE_PATH"

echo ""
echo "Done. Results saved to: $SAVE_PATH"