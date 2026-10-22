#!/usr/bin/env bash
# ============================================================================
# Orion TRT-LLM Checkpoint Convert + Engine Build 脚本
#
# 把 patched HF safetensors (vocab_size=32001, 含真实 waypoint embedding)
# 转为 TRT-LLM checkpoint，再编译成含 full_hidden_states 输出的 engine。
#
# 这是精度修复方案A的核心执行脚本：
#   根因：旧 engine vocab_size=32000，waypoint_token(32000) 越界，
#         被替换为 pad_embedding，导致 hidden_states 计算错误 → plan_L2 恶化。
#   修复：patched checkpoint vocab_size=32001，waypoint_token 用真实 embedding。
#
# 用法:
#   bash run_convert_and_build.sh                          # 默认 w8abf16
#   bash run_convert_and_build.sh --precision w8abf16      # W8A16 量化 (默认)
#   bash run_convert_and_build.sh --precision bfp16        # bfloat16 全精度
#   bash run_convert_and_build.sh --precision w4a16        # W4A16 量化
#   bash run_convert_and_build.sh --skip-convert           # 跳过 convert，只 build
#   bash run_convert_and_build.sh --skip-build             # 跳过 build，只 convert
#
# 环境要求:
#   - orion conda env (Python 3.8, TRT-LLM 0.13.0)
#   - TensorRT 10.4.0.26
#   - 已修改的 tensorrt_llm (modeling_utils.py:777 mark_output full_hidden_states)
# ============================================================================

set -euo pipefail

# ---- 解析参数 ----
PRECISION="w8abf16"
SKIP_CONVERT=0
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --skip-convert)
            SKIP_CONVERT=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Usage: bash run_convert_and_build.sh [--precision w8abf16|bfp16|w4a16] [--skip-convert] [--skip-build]"
            exit 1
            ;;
    esac
done

# 验证精度选项
case "$PRECISION" in
    w8abf16|bfp16|w4a16|fp16|fp8|w8abf16_kv8|bfp16_kv8|w8abf16_fp8kv|bfp16_fp8kv)
        ;;
    *)
        echo "ERROR: Unsupported precision: $PRECISION"
        echo "       Supported: w8abf16, bfp16, w4a16, fp16, fp8, *kv8 (INT8 KV cache), *fp8kv (FP8 KV cache)"
        exit 1
        ;;
esac

# ---- 项目根路径 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${ORION_PROJECT_ROOT:?error: ORION_PROJECT_ROOT is not set - point it at a clone of the upstream Orion training repo (see ckpts/README.md)}"

echo "========================================================================"
echo "  Orion TRT-LLM Convert + Build (方案A: patched vocab_size=32001)"
echo "========================================================================"
echo "  Repo root:     $REPO_ROOT"
echo "  Precision:     $PRECISION"
echo "  Skip convert:  $SKIP_CONVERT"
echo "  Skip build:    $SKIP_BUILD"

# ---- 路径配置 ----
# patched checkpoint (含正确 embed_tokens + lm_head, vocab_size=32001)
HF_MODEL_DIR="${ORION_HF_MODEL_DIR:-${ORION_PROJECT_ROOT}/ckpts/llm_clean_no_lora_patched}"
TRT_CKPT_DIR="${HF_MODEL_DIR}/trt"
# engine 输出（命名带 patched 后缀，避免覆盖旧 engine）
ENGINE_DIR="${ORION_LLM_ENGINE_BASE:-$REPO_ROOT/work_dirs/llm_safetensor}/x86_1gpu_${PRECISION}_fp32attn"

# ---- 环境变量（与 run_trt_eval.sh 完全一致，避免重复试错）----
# 所有路径/环境变量集中在此，后续调用一律通过本脚本，不再裸调 python。
export PYTHONPATH="${ORION_PROJECT_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export TRT_HOME="${TRT_HOME:-/usr/src/TensorRT-10.4.0.26}"
export LD_LIBRARY_PATH="${TRT_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3/envs/orion}"
export ORION_QFORMER_PATH="${ORION_QFORMER_PATH:-$REPO_ROOT/ckpts/hf_models/pretrain_qformer}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export QT_QPA_PLATFORM="offscreen"

PYTHON="$CONDA_PREFIX/bin/python"
TRTLLM_BUILD="$CONDA_PREFIX/bin/trtllm-build"

# ---- 检查文件存在 ----
echo "-----------------------------------------------------------------------"
echo "  Checking files..."
echo "-----------------------------------------------------------------------"

MISSING=0

if [ ! -f "${HF_MODEL_DIR}/config.json" ]; then
    echo "  [MISSING] HF model dir: ${HF_MODEL_DIR}/config.json"
    echo "           请先运行 patch_llm_checkpoint.py 生成 patched checkpoint"
    MISSING=1
else
    # 验证 vocab_size=32001
    VOCAB_SIZE=$(python3 -c "import json; print(json.load(open('${HF_MODEL_DIR}/config.json'))['vocab_size'])" 2>/dev/null || echo "unknown")
    if [ "$VOCAB_SIZE" != "32001" ]; then
        echo "  [ERROR]   HF model vocab_size=${VOCAB_SIZE}, 期望 32001"
        echo "           请重新运行 patch_llm_checkpoint.py"
        MISSING=1
    else
        echo "  [OK]      HF model: ${HF_MODEL_DIR} (vocab_size=${VOCAB_SIZE})"
    fi
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: 文件检查失败，请先完成 patch 步骤。"
    exit 1
fi
echo "  All files checked OK."
echo ""

# ============================================================================
# Step 1: Convert HF checkpoint → TRT-LLM checkpoint
# ============================================================================
if [ $SKIP_CONVERT -eq 0 ]; then
    echo "========================================================================"
    echo "  Step 1: Convert HF checkpoint → TRT-LLM checkpoint"
    echo "========================================================================"
    echo "  HF model dir:  $HF_MODEL_DIR"
    echo "  TRT ckpt dir:  $TRT_CKPT_DIR"
    echo "  Precision:     $PRECISION"

    # 根据精度设置 convert 参数
    CONVERT_ARGS=(
        --model_dir "$HF_MODEL_DIR"
        --output_dir "$TRT_CKPT_DIR"
        --log_level info
    )

    case "$PRECISION" in
        w8abf16|w8abf16_kv8|w8abf16_fp8kv)
            CONVERT_ARGS+=(--dtype float16 --use_weight_only --weight_only_precision int8)
            ;;
        w4a16)
            CONVERT_ARGS+=(--dtype float16 --use_weight_only --weight_only_precision int4)
            ;;
        bfp16|bfp16_kv8|bfp16_fp8kv)
            # 全精度 bfloat16
            CONVERT_ARGS+=(--dtype bfloat16)
            ;;
        fp16)
            # 全精度 float16
            CONVERT_ARGS+=(--dtype float16)
            ;;
        fp8)
            # FP8 权重量化
            CONVERT_ARGS+=(--dtype float16 --use_fp8)
            ;;
    esac

    # KV cache 量化（可叠加在任何权重精度上）
    case "$PRECISION" in
        *kv8)
            CONVERT_ARGS+=(--int8_kv_cache)
            ;;
        *fp8kv)
            CONVERT_ARGS+=(--fp8_kv_cache)
            ;;
    esac

    echo "  Convert args: ${CONVERT_ARGS[*]}"
    echo ""

    CONV_TIK=$(date +%s)
    $PYTHON "${ORION_PROJECT_ROOT}/adzoo/orion/convert_llm_checkpoint.py" "${CONVERT_ARGS[@]}"
    CONV_TOK=$(date +%s)
    echo ""
    echo "  Convert 耗时: $((CONV_TOK - CONV_TIK))s"
    echo ""

    # 验证 convert 输出
    if [ ! -f "${TRT_CKPT_DIR}/config.json" ]; then
        echo "ERROR: Convert 失败，${TRT_CKPT_DIR}/config.json 不存在"
        exit 1
    fi
    TRT_VOCAB=$(python3 -c "import json; c=json.load(open('${TRT_CKPT_DIR}/config.json')); print(c.get('vocab_size', c.get('pretrained_config',{}).get('vocab_size','unknown')))" 2>/dev/null || echo "unknown")
    echo "  [OK] TRT checkpoint vocab_size=${TRT_VOCAB}"
    if [ "$TRT_VOCAB" != "32001" ]; then
        echo "  [WARN] vocab_size=${TRT_VOCAB}, 期望 32001，请检查 convert 是否读取了 patched config"
    fi
else
    echo "  [SKIP] Step 1: Convert (用户指定 --skip-convert)"
fi
echo ""

# ============================================================================
# Step 2: Build TRT engine (含 full_hidden_states 输出)
# ============================================================================
if [ $SKIP_BUILD -eq 0 ]; then
    echo "========================================================================"
    echo "  Step 2: Build TRT engine (含 full_hidden_states 输出)"
    echo "========================================================================"
    echo "  TRT ckpt dir:  $TRT_CKPT_DIR"
    echo "  Engine dir:    $ENGINE_DIR"

    if [ ! -f "${TRT_CKPT_DIR}/config.json" ]; then
        echo "ERROR: ${TRT_CKPT_DIR}/config.json 不存在，请先运行 convert"
        exit 1
    fi

    mkdir -p "$ENGINE_DIR"

    # Build 参数说明：
    #   --max_batch_size 1:        Orion 单帧推理，batch=1
    #   --max_input_len 600:       输入序列长度（文本+视觉 token，约 599）
    #   --max_num_tokens 4096:     总 token 数上限
    #   --max_prompt_embedding_table_size 4096:
    #                              支持 p-tuning 视觉 token（视觉 token ID >= vocab_size，
    #                              通过 prompt_table 注入 embedding）。旧版 hs engine 设为 0
    #                              导致 prompt_table 无法生效——这是另一个隐藏 bug。
    #   --gemm_plugin float16:     GEMM 插件精度
    #   --enable_debug_output:     开启 debug 输出（full_hidden_states 通过 debug_buffer 读取）
    #   --remove_input_padding disable:
    #                              disable 以禁用 attention 插件兼容性检查。
    #                              enable 与 gpt_attention_plugin disable 不兼容
    #                              （assertion error: remove_input_padding requires
    #                              gpt_attention_plugin）。disable 模式下
    #                              full_hidden_states 仍输出 3D [1, ?, 4096]
    #                              (batch=1 时 total_tokens=seq_len)。
    #   --logits_dtype float16:    logits 精度
    #
    # 注意：full_hidden_states 输出由 modeling_utils.py:777 的 mark_output 实现，
    #       不需要额外 build 参数，只要用已修改的 tensorrt_llm 代码 build 即可。
    # 根据精度设置 gemm_plugin（注意：bfp16 必须与 checkpoint dtype 一致，否则类型不匹配）
    # 之前尝试 bfp16 + float16 gemm_plugin 导致 layer 0 残差连接类型冲突
    # （vocab_embedding bfloat16 vs GEMM plugin float16）
    case "$PRECISION" in
        bfp16)
            GEMM_PLUGIN="bfloat16"
            ;;
        *)
            GEMM_PLUGIN="float16"
            ;;
    esac

    # 决定是否使用 attention plugin
    # 当 USE_NO_PLUGIN=1 时，禁用 attention plugin，使用 Python 级 attention 实现（含 fp32 softmax）
    USE_NO_PLUGIN=${USE_NO_PLUGIN:-0}
    if [ "$USE_NO_PLUGIN" = "1" ]; then
        ENGINE_DIR="${ORION_LLM_ENGINE_BASE:-$REPO_ROOT/work_dirs/llm_safetensor}/x86_1gpu_${PRECISION}_noplugin"
        GPT_ATTN_PLUGIN="disable"
        PAGED_KV_CACHE="disable"
        REMOVE_INPUT_PADDING="disable"
        CONTEXT_FMHA="disable"
        echo "  [NO-PLUGIN mode] Using plain TensorRT attention (fp32 softmax)"
    else
        GPT_ATTN_PLUGIN="auto"
        PAGED_KV_CACHE="enable"
        REMOVE_INPUT_PADDING="enable"
        CONTEXT_FMHA="enable"
    fi

    BUILD_ARGS=(
        --checkpoint_dir "$TRT_CKPT_DIR"
        --output_dir "$ENGINE_DIR"
        --max_batch_size 1
        --max_input_len 600
        --max_num_tokens 4096
        --max_prompt_embedding_table_size 4096
        --gemm_plugin "$GEMM_PLUGIN"
        --gpt_attention_plugin "$GPT_ATTN_PLUGIN"
        --paged_kv_cache "$PAGED_KV_CACHE"
        --enable_debug_output
        --bert_context_fmha_fp32_acc enable
        --remove_input_padding "$REMOVE_INPUT_PADDING"
        --context_fmha "$CONTEXT_FMHA"
        --logits_dtype float16
    )

    echo "  Build args: ${BUILD_ARGS[*]}"
    echo ""

    BUILD_TIK=$(date +%s)
    $TRTLLM_BUILD "${BUILD_ARGS[@]}"
    BUILD_TOK=$(date +%s)
    echo ""
    echo "  Build 耗时: $((BUILD_TOK - BUILD_TIK))s"
    echo ""

    # 验证 engine 输出
    if [ ! -f "${ENGINE_DIR}/rank0.engine" ]; then
        echo "ERROR: Build 失败，${ENGINE_DIR}/rank0.engine 不存在"
        exit 1
    fi
    echo "  [OK] Engine: ${ENGINE_DIR}/rank0.engine"
    ENGINE_SIZE=$(du -h "${ENGINE_DIR}/rank0.engine" | cut -f1)
    echo "  [OK] Engine size: ${ENGINE_SIZE}"
else
    echo "  [SKIP] Step 2: Build (用户指定 --skip-build)"
fi
echo ""

# ============================================================================
# Step 3: 验证 engine 输出（full_hidden_states binding）
# ============================================================================
echo "========================================================================"
echo "  Step 3: 验证 engine 输出"
echo "========================================================================"

if [ -f "${ENGINE_DIR}/rank0.engine" ]; then
    echo "  检查 full_hidden_states binding..."
    $PYTHON -c "
import sys
sys.path.insert(0, '${ORION_PROJECT_ROOT}')
import tensorrt as trt
import tensorrt_llm
# 注册 TRT-LLM plugin（WeightOnlyQuantMatmul 等），否则反序列化 W8A16 engine 失败
from tensorrt_llm.plugin import _load_plugin_lib
_load_plugin_lib()

TRITON_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRITON_LOGGER)
with open('${ENGINE_DIR}/rank0.engine', 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

if engine is None:
    print('  [FAIL] 无法反序列化 engine')
    sys.exit(1)

# 列出所有输出 binding
output_names = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.OUTPUT:
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        output_names.append(name)
        print(f'  [OUT] {name}: shape={shape}, dtype={dtype}')

if 'full_hidden_states' in output_names:
    print()
    print('  ✅ full_hidden_states 已正确导出为 engine 输出')
else:
    print()
    print('  ❌ full_hidden_states 未找到！')
    print('     请确认 tensorrt_llm/models/modeling_utils.py:777 已插入 mark_output')
    sys.exit(1)

# 检查 vocab_size
import json
with open('${ENGINE_DIR}/config.json') as f:
    cfg = json.load(f)
vocab = cfg.get('pretrained_config', {}).get('vocab_size', 'unknown')
print(f'  vocab_size = {vocab}')
if vocab == 32001:
    print('  ✅ vocab_size=32001，waypoint_token(32000) 将使用真实 embedding')
else:
    print(f'  ❌ vocab_size={vocab}，期望 32001')
    sys.exit(1)
" 2>&1 | tail -20
else
    echo "  [SKIP] Engine 不存在，跳过验证"
fi
echo ""

echo "========================================================================"
echo "  完成！"
echo "========================================================================"
echo "  Engine: ${ENGINE_DIR}"
echo ""
echo "  下一步：运行 run_trt_eval.sh 评估精度"
echo "    bash ${SCRIPT_DIR}/run_trt_eval.sh --llm-precision ${PRECISION}_patched"
echo "  （需在 run_trt_eval.sh 的 precision case 里添加 ${PRECISION}_patched 选项）"
echo "========================================================================"
