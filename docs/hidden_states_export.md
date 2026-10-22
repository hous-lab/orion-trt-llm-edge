# Exporting full hidden states from a TRT-LLM engine

VLA models use the LLM differently from a chat model: the planning head consumes the
per-token hidden states of the full sequence, not the last-token logits. For ORION the
consumed tensor is the full 3D hidden states `[1, 599, 4096]` (529 vision tokens + system
+ instruction + waypoint token), from which the head extracts the waypoint-token position
and feeds `[1, 4096]` into the ego head.

The standard LLM deployment path does not expose this tensor:

- `gather_last_token_logits` collapses the 3D hidden states to a 2D last-token vector
  before `lm_head`, and only `logits` is marked as engine output.
- The official `additional_model_outputs=["hidden_states"]` API (v1.1+) requires the model
  forward to actively return the key, which stock models do not do (upstream issue #4414).
- `SaveHiddenStatesDecodingConfig` (v1.x) writes hidden states to disk for EAGLE3 draft
  training - offline, not a runtime tensor binding.

## The two-line patch (TRT backend, v0.7-v0.21)

In `tensorrt_llm/models/modeling_utils.py`, inside the `is_last_pp_rank()` block, insert
before the 3D-to-2D collapse:

```python
if self.config.mapping.is_last_pp_rank():
    hidden_states.mark_output("full_hidden_states", self.config.dtype)  # <- added
    hidden_states = gather_last_token_logits(...)
    lm_logits = self.lm_head(hidden_states)
```

The shipped patch is `trtllm_7b/patches/modeling_utils_v0x.patch`; `trtllm_7b/setup_training_repo.sh`
applies it to the installed TRT-LLM package (idempotent: skipped when `full_hidden_states`
is already present).

Verified on the ORION 7B build: engine output binding `full_hidden_states [1, 599, 4096]`,
hidden states cos 0.9994 vs the PyTorch forward under INT8 quantization, end-to-end
plan_L2 0.696 (official 0.690). The patch structure is identical across v0.7-v0.21
(`gather_last_token_logits` -> `lm_head` -> `mark_output('logits')`); we verified each
release's `modeling_utils.py` individually.

## Reading the tensor at inference

Through the TRT-LLM session debug buffer:

```python
full_hidden_states = session.debug_buffer["full_hidden_states"]   # [1, 599, 4096]
ego_feature = full_hidden_states[0:1, wp_idx:wp_idx + 1, :].squeeze(0)  # [1, 4096]
```

The waypoint token index is the vocab-boundary id (32000 in ORION's 32001-token vocab -
see `trtllm_7b/scripts/extract_embed_tokens.py` and `patch_llm_checkpoint.py`, which
extend the checkpoint vocab and copy the embedding row in).

## The v1.x PyTorch backend

From v1.0 the backend is PyTorch-native and the patch moves to
`DecoderModelForCausalLM.forward` (in `_torch/models/modeling_utils.py`): return the raw
3D output of `self.model(...)` as an additional output before the
`logits_processor` collapses it:

```python
def forward(self, ..., additional_model_outputs=None):
    output = self.model(...)
    results = {}
    if additional_model_outputs and "hidden_states" in additional_model_outputs:
        results["hidden_states"] = output        # full 3D tensor
    results["logits"] = self.logits_processor.forward(output, self.lm_head, ...)
    return results
```

This composes with the official `SamplingParams(additional_model_outputs=["hidden_states"])`
API instead of bypassing it. We shipped and validated the v0.x patch on a real deployment;
the v1.x variant is documented but not validated on our hardware.
