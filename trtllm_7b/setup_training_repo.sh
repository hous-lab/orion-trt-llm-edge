#!/usr/bin/env bash
# Integrate this repository's deployment files into the upstream training repos.
#
# The model definitions (mmcv fork, adzoo/orion model code) are NOT redistributed:
# they live in the upstream Orion training repo you clone separately. This script
# copies our deployment files into that repo and applies the two patches.
# Idempotent: safe to re-run.
#
# Usage:
#   source trtllm_7b/env_template.sh   # sets ORION_PROJECT_ROOT / CONDA_PREFIX
#   bash trtllm_7b/setup_training_repo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${ORION_PROJECT_ROOT:?error: ORION_PROJECT_ROOT is not set - point it at a clone of the upstream Orion training repo (see ckpts/README.md)}"
: "${CONDA_PREFIX:?error: ORION_PROJECT_ROOT unset message above; CONDA_PREFIX must point at the env with TRT-LLM installed}"

echo "repo root:          ${REPO_ROOT}"
echo "ORION_PROJECT_ROOT: ${ORION_PROJECT_ROOT}"
echo "CONDA_PREFIX:       ${CONDA_PREFIX}"
echo ""

# ---------------------------------------------------------------------------
# 1. TRT-LLM modeling_utils patch (full_hidden_states export, v0.x backends)
# ---------------------------------------------------------------------------
PY="$CONDA_PREFIX/bin/python"
MU="$($PY -c 'import tensorrt_llm, os; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), "models", "modeling_utils.py"))')"
echo "[1/4] TRT-LLM modeling_utils: ${MU}"

if grep -q 'full_hidden_states' "$MU"; then
    echo "      already patched, skipping"
else
    (cd "$REPO_ROOT/trtllm_7b/patches" && patch "$MU" modeling_utils_v0x.patch)
    echo "      patched"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. adzoo/orion/apis/test.py patch (single_gpu_test_trt)
# ---------------------------------------------------------------------------
TEST_APIS="$ORION_PROJECT_ROOT/adzoo/orion/apis/test.py"
echo "[2/4] apis/test.py: ${TEST_APIS}"
if grep -q 'single_gpu_test_trt' "$TEST_APIS"; then
    echo "      already patched, skipping"
else
    (cd "$ORION_PROJECT_ROOT" && patch -p1 < "$REPO_ROOT/trtllm_7b/patches/apis_test_trt.patch")
    echo "      patched"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Copy our deployment files into the training repo (adzoo/orion/)
# ---------------------------------------------------------------------------
echo "[3/4] copy deployment files -> ${ORION_PROJECT_ROOT}/adzoo/orion/"
cp -v "$REPO_ROOT"/trtllm_7b/adzoo_orion/* "$ORION_PROJECT_ROOT/adzoo/orion/"
# Single-file helpers imported by test_trt.py at runtime (same directory).
cp -v "$REPO_ROOT"/trtllm_7b/canbus_fixer.py "$ORION_PROJECT_ROOT/adzoo/orion/"
cp -v "$REPO_ROOT"/trtllm_7b/lightweight_ego.py "$ORION_PROJECT_ROOT/adzoo/orion/"
# Checkpoint-prep chain: LoRA merge -> clean Llama checkpoint. merge_lora_completely
# imports load_model from save_llm_checkpoint; both must sit in adzoo/orion/.
cp -v "$REPO_ROOT"/trtllm_7b/scripts/merge_lora_completely.py "$ORION_PROJECT_ROOT/adzoo/orion/"
cp -v "$REPO_ROOT"/trtllm_7b/scripts/save_llm_checkpoint.py "$ORION_PROJECT_ROOT/adzoo/orion/"
# onnxsim_large_model (MIT, vendored): the vision ONNX export simplifies >2GB graphs
# through it; the import `from onnxsim_large_model...` must resolve from adzoo/orion/.
cp -rv "$REPO_ROOT"/trtllm_7b/onnxsim_large_model "$ORION_PROJECT_ROOT/adzoo/orion/"
# Eval config used by the checkpoint-prep chain (our variant of upstream orion_stage3).
cp -v "$REPO_ROOT"/trtllm_7b/configs/orion_stage3_fp16.py "$ORION_PROJECT_ROOT/adzoo/orion/configs/"
echo ""

# ---------------------------------------------------------------------------
# 4. Copy the Lite distill config into the Lite training repo (if set)
# ---------------------------------------------------------------------------
if [ -n "${ORION_LITE_PROJECT_ROOT:-}" ] && [ -d "${ORION_LITE_PROJECT_ROOT}/adzoo/orion/configs" ]; then
    echo "[4/4] copy lite/configs/* -> ${ORION_LITE_PROJECT_ROOT}/adzoo/orion/configs/"
    cp -v "$REPO_ROOT"/lite/configs/*.py "$ORION_LITE_PROJECT_ROOT/adzoo/orion/configs/"
else
    echo "[4/4] ORION_LITE_PROJECT_ROOT not set - skipping Lite config copy"
fi
echo ""
echo "done. Training repos now carry the deployment files; run the pipeline scripts
from trtllm_7b/ (they must execute with cwd = ORION_PROJECT_ROOT, which they handle
internally via cd)."
