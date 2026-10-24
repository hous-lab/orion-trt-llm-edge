# orion-trt-llm-edge

TensorRT deployment toolchain for [xiaomi-mlab/Orion](https://github.com/xiaomi-mlab/Orion), a 7B VLA end-to-end driving model, on NVIDIA Jetson AGX Orin and an x86_64 dev host. 
The repository covers two deployment variants of the same pipeline:

- **7B Orion, x86**: the full ORION model, LLM part compiled with TensorRT-LLM. The upstream TRT-LLM v0.13 release cannot export hidden states, we ship a two-line source patch ([docs/hidden_states_export.md](docs/hidden_states_export.md)) that adds a `full_hidden_states`output binding, so the deployed engine exposes the exact tensor the planning head consumes. With W8A8 SmoothQuant on the LLM (26 ms vs 64.6 ms FP16, LLM part only), end-to-end is ~90 ms on an RTX 4090 at plan_L2 0.696 (official PyTorch: 0.690).
- **Orion-Lite, Orin**: a distilled 529M-parameter student replaces the 7B LLM, and the EVA02 ViT is distilled from 24 layers to 6 layers at 384x384 input. End-to-end Orin latency: **81.2 ms (MAXN, re-verified on the shipped artifacts)**, under the 100 ms real-time bar.

Every script runs from a fresh clone against the upstream training repo (not redistributed). [run_pipeline.sh](run_pipeline.sh) guards every step and fails fast with a pointer to [ckpts/README.md](ckpts/README.md) when an upstream artifact is missing. The exported artifacts are published on Hugging Face: the 6L@384 vision ONNX, the student ONNX and the distilled student checkpoint at [hous-lab/orion-trt-llm-edge](https://huggingface.co/hous-lab/orion-trt-llm-edge).

## Results

**7B Orion** - Full-pipeline end-to-end latency on the 4090 24GB GPU, measured as input frame to trajectory output through vision engine + canbus/memory update + TRT-LLM LLM + ego/planning heads + output post-processing (per-module CUDA-event profiling, 30-frame mean, plan_L2 re-verified at 0.692 on the 173-frame set):

| Configuration | Vision | LLM | Ego head | End-to-end |
|---|---|---|---|---|
| FP16-tier LLM | 59.1 ms | 57.5 ms | 5.0 ms | ~122 ms |
| W8A8 SmoothQuant LLM | 59.1 ms | 26.0 ms | 5.0 ms | ~90 ms |

LLM-only latency on two RTX 4090-class GPUs (24 GB server / 16 GB laptop), plan_L2 on the 173-frame Bench2Drive val subset (official PyTorch reference: 0.690):

| LLM tier | plan_L2 @1s | LLM latency, 4090 24 GB | LLM latency, 4090 Laptop 16 GB |
|---|---|---|---|
| PyTorch official (reference) | 0.690 | - | - |
| TRT-LLM FP16 | 0.697 | 64.6 ms | OOM (13.5 GB weights) |
| TRT-LLM INT8 (w8abf16) | 0.692 | 65.6 ms | 132.5 ms |
| TRT-LLM INT4 (w4a16) | 0.750 (degrades) | 64.9 ms | 126.0 ms |
| TRT-LLM W8A8 SmoothQuant | 0.696 | 26.0 ms | 50.4 ms |

The LLM stage reads 57.5 ms inside this pipeline vs 64.6 ms in the isolated prefill profile above - different harness and frame set, both reported as measured. The INT8/INT4 tiers were not end-to-end timed (their LLM-only latency sits within ~1 ms of FP16, so the end-to-end number is expected in the same ~122 ms band); the laptop was not end-to-end timed either. Reproduce with `trtllm_7b/scripts/01_profile_modules.py` (per-module CUDA events) and `trtllm_7b/scripts/10_e2e_latency.py` (full serial wall-clock pass), both driving the same patched training repo via `ORION_PROJECT_ROOT`.

Every row of this table is the same original upstream 7B model (Orion.pth - no distillation, no retraining anywhere in the 7B chain). Rows differ only in the LLM quantization tier; the vision engine and the planning heads are identical in every row. "LLM latency" is the LLM-only prefill (one forward pass of the 599-token prompt with max_new_tokens=1, mean over repeated post-warmup runs, f10 frame). plan_L2 is full-pipeline accuracy (vision engine + TRT-LLM LLM + planning heads) under the same eval protocol as the PyTorch reference; the INT8/INT4 tiers re-evaluated on the laptop scored 0.689/0.746, within 0.004 of the server values.

**Orion-Lite** on **Jetson AGX Orin**, MAXN power mode, plan_L2 on the same 173-frame subset:

| Variant | Orin end-to-end | plan_L2 @1s |
|---|---|---|
| 24L@640 + original student (undistilled starting point) | 678 ms | 0.814 |
| 24L@512 + original student (baseline) | 404 ms | 0.868 |
| 12L@512 (ViT distilled 24 to 12 layers) | 236 ms | 0.694 |
| 6L@448 (Route B variant) | 121 ms (154 ms at 50W) | 0.776 |
| **6L@384 + original student** | **81.2 ms** (re-verified on the shipped artifacts) | **0.704** |

Scheme notes: the 529M student is not a from-scratch small model - it is itself an LLM distillate, trained upstream to imitate the 7B LLM's planning token, and it stays frozen in every row. What this repository distills is the vision tower only: rows 1-2 are the undistilled 24L starting points, rows 3-5 replace the 24-layer EVA02 tower with 12/6-layer versions distilled against the 24L teacher at reduced input resolution. Orin end-to-end = vision tower + student + planning heads in one process at MAXN; plan_L2 = the fut-valid subset (133 frames) of the same 173-frame route. The Lite metric is not directly comparable to the 7B table above - different harness and frame count; see docs/model_variants.md.

About the 6L@384 latency numbers: a re-verification pass on the shipped artifacts (HF ONNX -> our TRT builder -> a zero-copy joint chain that aliases the vision output buffer as the student input) measures the end-to-end chain at 81.2 ms on an idle Orin at MAXN with locked clocks (per-frame wall clock, n=50). In the shipped step4 real-data harness the vision engine alone measures 84.3 ms including per-frame input copies and output copies, and the student engine 14.2 ms in an inputs-resident execute+sync loop. The earlier campaign numbers (85.2 ms = vision 69.7 + student 15.5) come from an inputs-resident execute+sync harness on an earlier engine build - the segment sums are not directly comparable to the harness numbers above, and the rebuilt engine is faster than the earlier build (88.9 ms vs 84.3 ms in the same shipped harness on the same test frame). Accuracy gates pass on the re-verification: vision-embedding cosine 0.864 (gate 0.809) and bbox cosine 0.9988 (matching the committed x86 evidence at 0.9985). Full numbers in lite/results/orin_reverify_6l384_20261024.json.

Takeaways we would want every reader to leave with:

- The hidden-states export is the enabling piece for the 7B deployment: TRT-LLM v0.13 has no API for it (upstream issue #4414); the two-line patch turns any v0.x LLaMA-family build into a hidden-states exporter (docs/hidden_states_export.md also covers the v1.x backend).
- INT8/PTQ measured no latency win on the vision workload: calibrated INT8 ran within noise of FP16 on x86 (128.8 vs 129.7 ms) and no gain on Orin (Q/DQ overhead cancelled the savings); the shipped INT8 tier exists for completeness, FP16 is the recommended tier.
- On the 7B LLM the interesting result is the opposite: W8A16/INT4 weight-only quantization did not move prefill latency (65 ms band), while full W8A8 SmoothQuant halved activation traffic and dropped it 2.5x to 26 ms. Quantize activations, not just weights.
- Resolution reduction is the training-free lever (640 to 512 alone: 678 to 404 ms e2e), 384 only worked after the backbone-only 6-layer distillation (ve cos 0.926 vs 0.866 for the 24-layer tower at 384).
- Power mode matters as much as kernel work on Orin: 50W to MAXN alone cuts the vision module 118.96 to 84.76 ms (-29%), e2e 154 to 121 ms.

## Get started

Prerequisites:

- NVIDIA GPU. The 7B engine build needs 24 GB+ (verified on RTX 4090 24 GB); the Lite pipeline runs on 8 GB.
- The upstream training repos cloned separately (we do not redistribute model definitions) - see [ckpts/README.md](ckpts/README.md) for links, checksums and placement.
- A conda env with TRT-LLM / TensorRT installed - see [docs/environment.md](docs/environment.md).

Steps:

1. Clone this repo, point the env vars at the training repos and integrate our files into them:
```bash
export ORION_PROJECT_ROOT=/path/to/orion-training-repo
export ORION_LITE_PROJECT_ROOT=/path/to/lite-training-repo
source trtllm_7b/env_template.sh
bash trtllm_7b/setup_training_repo.sh   # applies the 2 patches + copies our files
```

2. 7B chain (x86, 24 GB+ GPU): convert the checkpoint and build the engine, then run the eval and the hidden-states verify gate:
```bash
bash trtllm_7b/run_convert_and_build.sh --precision w8abf16   # default; also bfp16 | w4a16
bash trtllm_7b/run_trt_eval.sh --llm-precision w8abf16_patched
bash trtllm_7b/run_verify.sh              # single-sample hidden-states sanity check
```

3. Lite pipeline (Orin or x86): export ONNX from a fused distillation checkpoint, build the FP16 (or INT8) engine, run the real-data benchmark:
```bash
export FUSED_CKPT=/path/to/orion_lite_fused.pth   # see ckpts/README.md
bash run_pipeline.sh                              # PLATFORM=orin for the Orin leg
```

Use `BUILD_INT8=1 bash run_pipeline.sh` for the INT8 tier; sync to the Orin with `bash lite/scripts/deploy_to_orin.sh`.

## Hardware platforms

7B chain: RTX 4090 (24 GB) server GPU (128 SM, 384-bit GDDR6X, measured ~920 GB/s).
A 16 GB laptop 4090 also runs the quantized LLM tiers (dual-platform latency columns in the Results table; the FP16/BF16 engines at 13.5 GB do not fit in 16 GB).
We did not run the 7B chain on Jetson: the FP16 LLM weights alone are 13.5 GB and Orin's 204.8 GB/s LPDDR5 makes the prefill impractical.

Lite chain: Jetson AGX Orin 64GB, MAXN power mode. The Orin numbers are Jetson-specific: memory bandwidth, power caps and the DLA units are not comparable to a desktop GPU, so treat the latencies as order-of-magnitude anchors for your own hardware, not transferable constants.

## Acknowledgements and license

- [xiaomi-mlab/Orion](https://github.com/xiaomi-mlab/Orion) (Apache-2.0) - the 7B VLA model and training code; the upstream training repos are referenced, not redistributed.
- [Orion paper: ICCV 2025](https://arxiv.org/abs/2503.19755)
- TensorRT-LLM and TensorRT (NVIDIA) - the deployment backends; our `full_hidden_states` patch is a two-line source change under Apache-2.0.
- EVA-02, Qwen2-VL and OmniDrive - components referenced by the upstream model.
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) - the evaluation dataset, [data/README.md](data/README.md).

Code in this repository is Apache-2.0 (see `LICENSE`); upstream attribution is in `NOTICE`. Upstream artifacts are download-only, see [ckpts/README.md](ckpts/README.md).
