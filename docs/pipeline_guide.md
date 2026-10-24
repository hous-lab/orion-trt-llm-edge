# Pipeline guide

Two independent chains; each starts with fail-fast checks that point at
[ckpts/README.md](../ckpts/README.md) when an upstream artifact is missing.

## 7B chain (x86, 24 GB+ GPU)

One-time env setup:

```bash
export ORION_PROJECT_ROOT=/path/to/orion-training-repo
export ORION_LITE_PROJECT_ROOT=/path/to/lite-training-repo   # Lite pipeline only
source trtllm_7b/env_template.sh
bash trtllm_7b/setup_training_repo.sh
```

Setup copies the deployment files into the training repo (test_trt.py plus the
canbus_fixer / lightweight_ego helpers, the LoRA-merge scripts, the eval config) and
patches two files: the TRT-LLM modeling_utils.py (full_hidden_states export) and the
training repo apis/test.py (single_gpu_test_trt entry).

| # | Step | Produces |
|---|---|---|
| 1 | LoRA merge: merge_lora_completely.py (see its docstring for the full command) | ckpts/llm_clean_no_lora/ |
| 2 | extract_embed_tokens.py | ckpts/embed_tokens_orion_fp16.pt (waypoint embedding row) |
| 3 | patch_llm_checkpoint.py | ckpts/llm_clean_no_lora_patched/ (vocab_size=32001) |
| 4 | run_convert_and_build.sh --precision w8abf16 | work_dirs/llm_safetensor/x86_1gpu_w8abf16_fp32attn/rank0.engine |
| 5 | run_trt_eval.sh --llm-precision w8abf16_patched | plan_L2 report under work_dirs/trt_eval_w8abf16_patched/ |
| 6 | run_verify.sh | hidden-states cos vs the PyTorch forward |

Notes:

- Steps 1-3 run from the training-repo root (they import the mmcv fork and expect
  ckpts/Orion.pth at the documented placement).
- The vision engine is a two-step chain: [trtllm_7b/adzoo_orion/export_vision.sh](../trtllm_7b/adzoo_orion/export_vision.sh)
  exports the mixed-precision ONNX (backbone FP16, map/pts heads kept FP32;
  simplification uses the vendored [trtllm_7b/onnxsim_large_model](../trtllm_7b/onnxsim_large_model/), then
  [trtllm_7b/scripts/build_vision_engine.sh](../trtllm_7b/scripts/build_vision_engine.sh) builds it with trtexec --fp16 to
  work_dirs/vision_mixed_precision.engine, the default every driver resolves
  (ORION_VISION_ENGINE / ORION_VISION_ONNX to override). The INT8 vision tier is
  separate: [trtllm_7b/scripts/gen_vision_calib_data.py](../trtllm_7b/scripts/gen_vision_calib_data.py) + [trtllm_7b/scripts/build_vision_int8.py](../trtllm_7b/scripts/build_vision_int8.py).
- The W8A8 SmoothQuant tier (the 26 ms result) is a separate three-script recipe:
  [trtllm_7b/quant/extract_llm_weights.py](../trtllm_7b/quant/extract_llm_weights.py) (checkpoint to shards), then
  [trtllm_7b/quant/smoothquant_calib.py](../trtllm_7b/quant/smoothquant_calib.py) (calibration, writes ckpts/quant_data/smoothquant_scale.pt),
  then [trtllm_7b/quant/convert_w8a8_inject.py](../trtllm_7b/quant/convert_w8a8_inject.py) (monkey-patches the training-repo converter and
  reuses its argument parsing). All quant artifacts live under
  <ORION_PROJECT_ROOT>/ckpts/quant_data/.

## Lite chain (x86 or Orin)

```bash
export ORION_LITE_PROJECT_ROOT=/path/to/lite-training-repo
export FUSED_CKPT=/path/to/orion_lite_fused.pth   # see ckpts/README.md
bash run_pipeline.sh                              # PLATFORM=orin for the Orin leg
```

Defaults run steps 2-4 (ONNX export, engine build, real-data benchmark); the step
gates:

| Env | Default | Effect |
|---|---|---|
| RUN_STEP1 | 0 | also run distillation (teacher data collection + student training) |
| RUN_STEP2 | 1 | ONNX export of vision module + student head |
| RUN_STEP3 | 1 | TensorRT engine build (FP16; BUILD_INT8=1 for the INT8 tier) |
| RUN_STEP4 | 1 | real-data benchmark (cos gates + latency) |
| PLATFORM | x86 | x86 or orin step3/step4 implementations |
| BUILD_INT8 | 0 | build + benchmark the INT8 engine |

Step 1 (distillation) is optional and only needed to regenerate the fused checkpoint
from scratch; it needs the full teacher (Orion.pth) and the distillation data set.

Orin leg: [lite/scripts/deploy_to_orin.sh](../lite/scripts/deploy_to_orin.sh) rsyncs the release tree to the device
($ORION_HOST); the Orin side runs the same [run_pipeline.sh](../run_pipeline.sh) with PLATFORM=orin under
MAXN. Orin timing numbers must state the power mode (see [docs/power_modes.md](power_modes.md)).
