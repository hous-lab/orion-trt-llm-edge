# Environment and versions

The pipeline was developed and verified against the following stack. Newer versions
usually work for the Lite pipeline (plain TensorRT); the 7B chain pins TRT-LLM v0.13
because the patch targets the v0.x backend.

## Software

| Component | Version used | Notes |
|---|---|---|
| Python | 3.8 | orion conda env |
| TRT-LLM | 0.13.0.dev2024082700 | 7B chain only; patch targets v0.x backend |
| TensorRT | 10.4 | both chains (Lite x86 evidence ran 10.16) |
| CUDA | 11.8 / 12.x | per platform |
| torch | 2.4.1+cu118 | training repo's recipe (torchvision 0.19.1+cu118, nvidia-cudnn-from-pip cu11 chain, numpy 1.24.4) - verified end-to-end by the 173-frame reproduction run |
| mmcv/mmcv-full | upstream Orion fork | installed from the training repos |

Both training repos are cloned separately and located via `ORION_PROJECT_ROOT` /
`ORION_LITE_PROJECT_ROOT`; see [ckpts/README.md](../ckpts/README.md) for what to clone and where artifacts
are placed.

## Hardware targets

| Platform | GPU | TRT | Role |
|---|---|---|---|
| x86 server | RTX 4090 24GB x4 | 10.4 | 7B build + eval, distill training |
| x86 laptop | RTX 4090 Laptop 16GB | 10.4 | Lite x86 evidence runs, INT8 comparison |
| Jetson AGX Orin 64GB | unified memory LPDDR5 | 10.4 (JetPack) | Lite deployment target |

## Power modes (Orin)

The Orin numbers in this repo were taken at MAXN. Power-mode effects are documented in
[power_modes.md](power_modes.md).

## Known version pitfalls

- **flash_attn is not a pipeline dependency for export/build/eval.** The training
  repos' vendored mmcv forks hard-import `flash_attn` at module level, but it is a
  compiled wheel pinned to a torch ABI (a torch upgrade breaks it with an undefined
  c10 symbol at import). The ONNX export forces `flash_attn=False` on every attention
  consumer and the engine build/eval steps never import the training repo, so a
  missing or ABI-broken flash_attn does not affect the deployment pipeline: the step-2
  export script installs a raise-if-called stub when flash_attn is not importable. Only
  re-running distill training with flash attention enabled needs a real flash-attn
  built against your torch build.
- The 7B chain's [run_convert_and_build.sh](../trtllm_7b/run_convert_and_build.sh) needs the training repo's mmcv fork; the
  env must import both the training repo at `ORION_PROJECT_ROOT` and TensorRT-LLM.
  Both must import in one interpreter (sys.path glue handled by setup script).

- TRT-LLM weight conversion requires the checkpoint vocab patch (32000 to 32001) - the
  `waypoint_token` embedding row is extracted and merged by
  [trtllm_7b/scripts/extract_embed_tokens.py](../trtllm_7b/scripts/extract_embed_tokens.py) + [patch_llm_checkpoint.py](../trtllm_7b/scripts/patch_llm_checkpoint.py).
- On a 16 GB GPU the FP16/BF16 engines (13.5 GB) OOM on load; INT8/W8A8 (7.0 GB) fit.
- **A numpy install corrupted by a stacked reinstall kills the torch numpy bridge.**
  Installing numpy over an existing numpy without uninstalling first (pip's
  `--ignore-installed` cascade leaves two dist-infos interleaving one package
  tree) leaves torch unable to C-level-recognize the running numpy: every numpy
  conversion fails (`from_numpy` -> "expected np.ndarray (got numpy.ndarray)",
  `np.asarray(tensor)` -> "object __array__ method not producing an array") while
  pure-python numpy still works. Fix: `pip uninstall numpy` then one clean
  install. The Lite eval script's shims absorb torch-version drift (missing
  optional imports, a comparison shim), but a dead numpy bridge is not
  absorbable - the install must be clean.
- **mmcv config files must not `import os` at top level.** mmcv collects every
  non-dunder name in the config namespace into the config dict, so an imported
  module leaks in and Config.fromfile crashes in deepcopy ("cannot pickle
  'module' object"). The upstream configs read env vars with the inline
  `__import__("os").environ.get(...)` form instead; the vendored eval config
  follows that convention.
