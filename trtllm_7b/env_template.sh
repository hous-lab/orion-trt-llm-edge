#!/usr/bin/env bash
# ============================================================================
# Environment template for the 7B TRT-LLM toolchain (source me, then override).
#
# Usage:
#   source trtllm_7b/env_template.sh
#
# Override on the command line or in your shell; nothing here is machine
# specific except the fallback defaults, which you should adapt:
#   export ORION_PROJECT_ROOT=/path/to/orion-training-repo     # required
#   export ORION_LITE_PROJECT_ROOT=/path/to/lite-training-repo # Lite pipeline only
#   export CONDA_PREFIX=$HOME/miniconda3/envs/<env>            # your conda env
#   export TRT_HOME=/path/to/TensorRT-10.x.x.x                 # TRT tarball root
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ORION_TRTLLM_EDGE_ROOT="${SCRIPT_DIR}/.."

# Training repos are NOT redistributed by this repository; the scripts import
# model definitions from them. See ckpts/README.md for what to clone.
export ORION_PROJECT_ROOT="${ORION_PROJECT_ROOT:-}"
export ORION_LITE_PROJECT_ROOT="${ORION_LITE_PROJECT_ROOT:-}"

# Conda env with the training repo installed (see docs/environment.md).
export CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3/envs/orion}"
export PYTHON="${CONDA_PREFIX}/bin/python"

# QFormer weights: download per ckpts/README.md into the release tree.
export ORION_QFORMER_PATH="${ORION_QFORMER_PATH:-${ORION_TRTLLM_EDGE_ROOT}/ckpts/hf_models/pretrain_qformer}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export QT_QPA_PLATFORM="offscreen"
export PYTHONUNBUFFERED=1

# TensorRT: point TRT_HOME at your tarball (10.x) or /usr/src/TensorRT-<ver>.
export TRT_HOME="${TRT_HOME:-/usr/src/TensorRT-10.4.0.26}"
export LD_LIBRARY_PATH="${TRT_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[env] ORION_PROJECT_ROOT=${ORION_PROJECT_ROOT:-<unset - required>}"
echo "[env] ORION_LITE_PROJECT_ROOT=${ORION_LITE_PROJECT_ROOT:-<Lite pipeline only>}"
echo "[env] CONDA_PREFIX=${CONDA_PREFIX}"
echo "[env] TRT_HOME=${TRT_HOME}"
echo "[env] ORION_QFORMER_PATH=${ORION_QFORMER_PATH}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
