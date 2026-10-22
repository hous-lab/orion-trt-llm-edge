# Weights and upstream artifacts

The pipeline does not redistribute any upstream-hosted file. Every artifact in
the table below is download-only: fetch it from the upstream location, place it
at the path listed, and verify the sha256. Rehosting upstream weights - or any
derivative that embeds them - is prohibited.

## Download table

| File | Source | sha256 | Place at |
|---|---|---|---|
| `Orion.pth` (7B teacher + vision tower, 38.5 GB) | [huggingface.co/poleyzdk/Orion](https://huggingface.co/poleyzdk/Orion/blob/main/Orion.pth) | `4977a0505f05883c7d08390c925ef6519053381cb579113c318fb6c0c19d2a06` | `<ORION_PROJECT_ROOT>/ckpts/Orion.pth` |
| `pretrain_qformer` weights (2-shard, 14.1 GB) | [huggingface.co/exiawsh/pretrain_qformer](https://huggingface.co/exiawsh/pretrain_qformer/) | `9817acae1bbbabc48e7b386c17481e58d33f67b403db8d850df0a024c54567af` (shard 1) / `061cd64a7f82110dd89cf9257988c8edca00c9cebcc2174d50158b16a40f63c0` (shard 2) | `<ORION_PROJECT_ROOT>/ckpts/hf_models/pretrain_qformer/` |
| `pytorch_model.bin.index.json` | same repo | `45213659fa64fbbba2d985c8e26b5ed657b9d17d36bc92e540881c6bb1205f6f` | `<ORION_PROJECT_ROOT>/ckpts/hf_models/pretrain_qformer/` |
| `config.json` | same repo | `0f87de89b230452a9124f8e7b823ddef3c8c83943c10e01efc00431138a85133` | `<ORION_PROJECT_ROOT>/ckpts/hf_models/pretrain_qformer/` |
| `generation_config.json` | same repo | `7d5091715eedd3e954a8ae6b7cb9bfb713ad7c706a531a9848167ebfc497e350` | `<ORION_PROJECT_ROOT>/ckpts/hf_models/pretrain_qformer/` |
| Qwen3-VL-2B-Instruct tokenizer/processor files | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | (upstream-managed) | `<release>/ckpts/hf_models/pretrain_qformer` or any dir, exposed via `ORION_QFORMER_PATH` |

Notes:

- The 7B eval (run_trt_eval.sh) consumes the QFormer directory as
  tokenizer/processor input (`ORION_QFORMER_PATH`); the 14.1 GB weights are
  only needed for the training-repo native forward path, not for the TRT
  eval.
- The `pretrain_qformer` config declares `qformer_embed_dim=256,
  qformer_query_number=256, mm_vision_tower=eva02-l-16, vision_tower_input_size=640` -
  the vision engine is built from Orion.pth (which embeds the EVA02 tower), so
  no separate EVA02 download is needed for deployment.
- Checkpoint vocabulary patch: the deployed LLM checkpoint is the upstream
  one with `vocab_size` extended to 32001 (the `waypoint_token` embedding,
  extracted by trtllm_7b/scripts/extract_embed_tokens.py and merged by
  trtllm_7b/scripts/patch_llm_checkpoint.py). See docs/hidden_states_export.md.

## Derived artifacts we ship

The following artifacts in this repository are ours (pipeline outputs), not
upstream files:

- `trtllm_7b/patches/*` - our source patches (full_hidden_states export;
  single_gpu_test_trt).
- `lite/results/*.json` - benchmark and per-layer profile evidence, produced
  by lite/step3 and lite/step4 on the listed hardware.
- The deployment ONNX exports (vision module at 640/512/384, student head)
  and the distilled student checkpoint are published on our Hugging Face:
  [hous-lab/orion-trt-llm-edge](https://huggingface.co/hous-lab/orion-trt-llm-edge).
  They are derived from the upstream checkpoint by the documented pipeline
  (ONNX graph + our own distilled student weights; the student is trained from
  scratch and contains no upstream weight tensors).
- The fused distillation checkpoints (Orion backbone + our trained student
  head) embed upstream weights and are therefore NOT redistributed; regenerate
  them with `RUN_STEP1=1` (see run_pipeline.sh) and the step1_distill scripts.
