#!/usr/bin/env bash
# ============================================================================
# Orion TRT-LLM 端到端推理脚本 (单GPU)
#
# 加载 Vision engine + LLM engine (含 full_hidden_states 输出)，
# 在 Bench2Drive 验证集上运行开环推理，评估 planning L2 精度。
#
# 用法:
#   bash run_trt_eval.sh                          # 默认使用 bfp16 LLM
#   bash run_trt_eval.sh --llm-precision bfp16    # 使用 bfp16 LLM engine（默认）
#   bash run_trt_eval.sh --llm-precision w8abf16  # 使用 w8abf16 LLM engine
#   bash run_trt_eval.sh --llm-precision w4a16    # 使用 w4a16 LLM engine
#   bash run_trt_eval.sh --llm-precision w8abf16_patched  # 使用方案A修复后的 W8A16 engine (vocab_size=32001)
#   bash run_trt_eval.sh --ego-engine /path/to/ego_fut_pred_sim.engine  # 提供 ego engine
#
# 环境要求:
#   - orion conda env (Python 3.8, TRT-LLM 0.13.0)
#   - TensorRT 10.4.0
# ============================================================================

set -euo pipefail

# ---- 解析参数 ----
LLM_PRECISION="bfp16"
EGO_ENGINE_PTH=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm-precision)
            LLM_PRECISION="$2"
            shift 2
            ;;
        --ego-engine)
            EGO_ENGINE_PTH="$2"
            shift 2
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# 验证 LLM precision
case "$LLM_PRECISION" in
    bfp16|w8abf16|w4a16|w8abf16_patched|bfp16_patched)
        ;;
    *)
        echo "ERROR: Unsupported LLM precision: $LLM_PRECISION"
        echo "       Supported: bfp16, w8abf16, w4a16, w8abf16_patched, bfp16_patched"
        exit 1
        ;;
esac

# ---- 项目根路径 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${ORION_PROJECT_ROOT:?error: ORION_PROJECT_ROOT is not set - point it at a clone of the upstream Orion training repo (see ckpts/README.md)}"
echo "========================================================================"
echo "  Orion TRT-LLM end-to-end eval"
echo "========================================================================"
echo "  Repo root:     $REPO_ROOT"

# ---- 路径配置 (overridable via env; defaults per ckpts/README.md layout) ----
CONFIG="${ORION_CONFIG:-$ORION_PROJECT_ROOT/adzoo/orion/configs/orion_stage3.py}"
TOKENIZER="${ORION_QFORMER_PATH:-$REPO_ROOT/ckpts/hf_models/pretrain_qformer/}"

# Vision engine (ONNX->TRT, mixed precision; built by adzoo_orion/export_vision.sh)
VISION_ENGINE="${ORION_VISION_ENGINE:-$REPO_ROOT/work_dirs/vision_mixed_precision.engine}"

# LLM engine (TRT-LLM, with full_hidden_states output)
# Engine dir follows run_convert_and_build.sh output naming; _patched variants map
# to the same engine (all release builds use the vocab-patched checkpoint).
ENGINE_STEM="${LLM_PRECISION%_patched}"
LLM_ENGINE="${ORION_LLM_ENGINE:-$REPO_ROOT/work_dirs/llm_safetensor/x86_1gpu_${ENGINE_STEM}_fp32attn}"

# Orion full weights (ego prediction fallback)
CHECKPOINT="${ORION_CHECKPOINT:-$ORION_PROJECT_ROOT/ckpts/Orion.pth}"

# Eval outputs (gitignored)
SAVE_PATH="${ORION_EVAL_OUT:-$REPO_ROOT/work_dirs/trt_eval_${LLM_PRECISION}/}"

# ---- 检查文件是否存在 ----
echo "-----------------------------------------------------------------------"
echo "  Checking files..."
echo "-----------------------------------------------------------------------"

MISSING=0

# Config
if [ ! -f "$CONFIG" ]; then
    echo "  [MISSING] Config: $CONFIG"
    MISSING=1
else
    echo "  [OK]      Config: $CONFIG"
fi

# Tokenizer
if [ ! -f "${TOKENIZER}/config.json" ]; then
    echo "  [MISSING] Tokenizer: ${TOKENIZER}/config.json"
    MISSING=1
else
    echo "  [OK]      Tokenizer: $TOKENIZER"
fi

# Vision engine
if [ ! -f "$VISION_ENGINE" ]; then
    echo "  [MISSING] Vision engine: $VISION_ENGINE"
    MISSING=1
else
    echo "  [OK]      Vision engine: $VISION_ENGINE"
fi

# LLM engine
if [ ! -d "$LLM_ENGINE" ] || [ ! -f "${LLM_ENGINE}/rank0.engine" ]; then
    echo "  [MISSING] LLM engine: $LLM_ENGINE"
    echo "           Available LLM engines:"
    for d in "$REPO_ROOT/work_dirs/"llm_*/; do
        if [ -f "${d}/rank0.engine" ]; then
            echo "             - $(basename "$d")"
        fi
    done
    MISSING=1
else
    echo "  [OK]      LLM engine: $LLM_ENGINE"
fi

# Ego engine (optional)
if [ -n "$EGO_ENGINE_PTH" ]; then
    if [ ! -f "$EGO_ENGINE_PTH" ]; then
        echo "  [MISSING] Ego engine: $EGO_ENGINE_PTH"
        echo "           (optional, will disable planning evaluation)"
        EGO_ENGINE_PTH=""
    else
        echo "  [OK]      Ego engine: $EGO_ENGINE_PTH"
    fi
else
    echo "  [SKIP]    Ego engine: not provided (planning evaluation disabled)"
fi

# Data files
DATA_ROOT="${ORION_DATA_ROOT:-$REPO_ROOT/data/bench2drive}"
ANN_FILE="${ORION_ANN_FILE:-$REPO_ROOT/data/infos-tiny/b2d_infos_val.pkl}"
MAP_FILE="${ORION_MAP_FILE:-$REPO_ROOT/data/infos-tiny/b2d_map_infos.pkl}"
MAP_ROOT="${ORION_MAP_ROOT:-$REPO_ROOT/data/bench2drive/maps}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "  [MISSING] Data root: $DATA_ROOT"
    MISSING=1
else
    echo "  [OK]      Data root: $DATA_ROOT"
fi
if [ ! -f "$ANN_FILE" ]; then
    echo "  [MISSING] Ann file: $ANN_FILE"
    MISSING=1
else
    echo "  [OK]      Ann file: $ANN_FILE"
fi
if [ ! -f "$MAP_FILE" ]; then
    echo "  [MISSING] Map file: $MAP_FILE"
    MISSING=1
else
    echo "  [OK]      Map file: $MAP_FILE"
fi
if [ ! -d "$MAP_ROOT" ]; then
    echo "  [MISSING] Map root: $MAP_ROOT"
    MISSING=1
else
    echo "  [OK]      Map root: $MAP_ROOT"
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Some files are missing. Please check the paths above."
    exit 1
fi
echo "  All files checked OK."
echo ""

# ---- 创建输出目录 ----
mkdir -p "$SAVE_PATH"

# ---- 环境变量 ----
# 与 VSCode launch.json 保持一致
export PYTHONPATH="${ORION_PROJECT_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export TRT_HOME="/usr/src/TensorRT-10.4.0.26"
export LD_LIBRARY_PATH="${TRT_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3/envs/orion}"
export ORION_QFORMER_PATH="$TOKENIZER"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export QT_QPA_PLATFORM="offscreen"

# ---- Python 解释器 ----
PYTHON="$CONDA_PREFIX/bin/python"

# ---- 输出配置信息 ----
echo "========================================================================"
echo "  配置摘要"
echo "========================================================================"
echo "  Config:         $CONFIG"
echo "  Vision engine:  $VISION_ENGINE"
echo "  LLM engine:     $LLM_ENGINE  (precision: $LLM_PRECISION)"
echo "  Tokenizer:      $TOKENIZER"
echo "  Ego engine:     ${EGO_ENGINE_PTH:-<not provided, planning eval disabled>}"
echo "  Save path:      $SAVE_PATH"
echo "  Python:         $PYTHON"
echo "  CUDA_VISIBLE:   $CUDA_VISIBLE_DEVICES"
echo "  TRT_HOME:       $TRT_HOME"
echo "========================================================================"
echo ""

# ---- 构建参数列表 ----
CMD_ARGS=(
    --engine_pth "$VISION_ENGINE"
    --eval bbox
    --config "$CONFIG"
    --launcher none
    --llm_engine_pth "$LLM_ENGINE"
    --tokenizer_pth "$TOKENIZER"
    --qa_save_path "$SAVE_PATH"     --checkpoint "$CHECKPOINT"
    --cfg-options
        data.test.data_root="$DATA_ROOT"
        data.test.ann_file="$ANN_FILE"
        data.test.map_root="$MAP_ROOT"
        data.test.map_file="$MAP_FILE"
)

# 添加可选的 ego engine
if [ -n "$EGO_ENGINE_PTH" ]; then
    CMD_ARGS+=(--ego_engine_pth "$EGO_ENGINE_PTH")
fi

# 添加额外参数
if [ ${#OTHER_ARGS[@]} -gt 0 ]; then
    CMD_ARGS+=("${OTHER_ARGS[@]}")
fi

# ---- 运行推理 ----
echo "========================================================================"
echo "  开始推理"
echo "========================================================================"
echo ""

# 切换到 PROJECT_ROOT：config 文件中 llm_path='ckpts/pretrain_qformer/' 等为相对路径，
# 必须在 PROJECT_ROOT 下运行 python 才能正确解析。
cd "$ORION_PROJECT_ROOT"

$PYTHON "${ORION_PROJECT_ROOT}/adzoo/orion/test_trt.py" "${CMD_ARGS[@]}"

echo ""
echo "========================================================================"
echo "  推理完成"
echo "========================================================================"
echo "  Results saved to: $SAVE_PATH"
echo "========================================================================"