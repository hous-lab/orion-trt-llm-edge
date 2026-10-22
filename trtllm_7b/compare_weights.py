"""
Compare HF model weights with TRT-LLM checkpoint weights.

HF format: sharded safetensors (model-0000{1,2,3}-of-00003.safetensors)
TRT-LLM format: single rank0.safetensors with fused QKV

Key mappings:
  HF: model.layers.{N}.self_attn.q_proj.weight  \
  HF: model.layers.{N}.self_attn.k_proj.weight   |-> TRT: transformer.layers.{N}.attention.qkv.weight (fused along dim=0)
  HF: model.layers.{N}.self_attn.v_proj.weight  /
  HF: model.layers.{N}.self_attn.o_proj.weight   -> TRT: transformer.layers.{N}.attention.dense.weight
  HF: model.layers.{N}.mlp.gate_proj.weight      -> TRT: transformer.layers.{N}.mlp.gate.weight
  HF: model.layers.{N}.mlp.up_proj.weight        -> TRT: transformer.layers.{N}.mlp.fc.weight
  HF: model.layers.{N}.mlp.down_proj.weight      -> TRT: transformer.layers.{N}.mlp.proj.weight
  HF: model.layers.{N}.input_layernorm.weight    -> TRT: transformer.layers.{N}.input_layernorm.weight
  HF: model.layers.{N}.post_attention_layernorm.weight -> TRT: transformer.layers.{N}.post_layernorm.weight
  HF: model.embed_tokens.weight                  -> TRT: transformer.vocab_embedding.weight
  HF: model.norm.weight                          -> TRT: transformer.ln_f.weight
  lm_head.weight                                 -> lm_head.weight (same key)
"""

import os
import torch
from safetensors import safe_open

HF_DIR = os.environ.get("ORION_PROJECT_ROOT")
if not HF_DIR:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
HF_DIR = os.path.join(HF_DIR, "ckpts", "llm_clean_no_lora_patched")
TRT_DIR = os.path.join(HF_DIR, "trt")
NUM_HIDDEN_LAYERS = 32
HIDDEN_SIZE = 4096
HEAD_SIZE = 128
NUM_HEADS = 32
NUM_KV_HEADS = 32

# ── Load all HF sharded safetensors into a single dict ──────────────────────
def load_hf_weights(hf_dir):
    """Load all HF sharded safetensors and return {key: tensor} dict."""
    hf_weights = {}
    shard_files = sorted([
        f for f in os.listdir(hf_dir)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    print(f"Loading {len(shard_files)} HF shards: {shard_files}")
    for shard in shard_files:
        path = os.path.join(hf_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_weights[key] = f.get_tensor(key).to(torch.float32)
    print(f"  Loaded {len(hf_weights)} HF tensors")
    return hf_weights


def load_trt_weights(trt_dir):
    """Load TRT-LLM rank0.safetensors and return {key: tensor} dict."""
    path = os.path.join(trt_dir, "rank0.safetensors")
    trt_weights = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            trt_weights[key] = f.get_tensor(key).to(torch.float32)
    print(f"  Loaded {len(trt_weights)} TRT tensors")
    return trt_weights


def compare_tensors(name, hf_t, trt_t, rtol=1e-4, atol=1e-4):
    """Compare two tensors and report stats."""
    if hf_t.shape != trt_t.shape:
        print(f"  ⚠ SHAPE MISMATCH: HF={list(hf_t.shape)} vs TRT={list(trt_t.shape)}")
        return

    diff = (hf_t - trt_t).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # Check if any element exceeds tolerance
    n_exceed = (diff > atol).sum().item()
    status = "⚠️  EXCEEDS" if max_diff > 1e-4 else "✅"

    # Also check if they are essentially identical (accounting for dtype conversion)
    n_exceed_strict = (diff > 1e-6).sum().item()

    print(f"  {status} | {name:<55s} | max_diff={max_diff:.3e} | mean_diff={mean_diff:.3e} | "
          f"n_exceed_1e-4={n_exceed:6d} / {hf_t.numel():>8d} | "
          f"shape={list(hf_t.shape)}")
    return max_diff, mean_diff


def compare_qkv_split(hf_prefix, trt_prefix, layer_idx, hf_weights, trt_weights):
    """Compare HF split Q/K/V with TRT fused QKV."""
    hf_q = hf_weights[f"{hf_prefix}.self_attn.q_proj.weight"]
    hf_k = hf_weights[f"{hf_prefix}.self_attn.k_proj.weight"]
    hf_v = hf_weights[f"{hf_prefix}.self_attn.v_proj.weight"]
    trt_qkv = trt_weights[f"{trt_prefix}.attention.qkv.weight"]

    # TRT fuses as [q, k, v] stacked along dim 0
    d = HIDDEN_SIZE
    trt_q = trt_qkv[:d, :]
    trt_k = trt_qkv[d:2*d, :]
    trt_v = trt_qkv[2*d:, :]

    results = []
    results.append(compare_tensors(f"layer{layer_idx}.q_proj", hf_q, trt_q))
    results.append(compare_tensors(f"layer{layer_idx}.k_proj", hf_k, trt_k))
    results.append(compare_tensors(f"layer{layer_idx}.v_proj", hf_v, trt_v))
    return results


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    hf_weights = load_hf_weights(HF_DIR)
    trt_weights = load_trt_weights(TRT_DIR)

    print("\n" + "=" * 120)
    print("COMPARISON RESULTS")
    print("=" * 120)

    all_max_diffs = []

    # 1. Embed tokens
    print("\n── Embedding & LM Head ──")
    r = compare_tensors("embed_tokens",
                        hf_weights["model.embed_tokens.weight"],
                        trt_weights["transformer.vocab_embedding.weight"])
    if r: all_max_diffs.append(r[0])

    r = compare_tensors("lm_head",
                        hf_weights["lm_head.weight"],
                        trt_weights["lm_head.weight"])
    if r: all_max_diffs.append(r[0])

    # 2. Final norm
    print("\n── Final Norm ──")
    r = compare_tensors("model.norm / ln_f",
                        hf_weights["model.norm.weight"],
                        trt_weights["transformer.ln_f.weight"])
    if r: all_max_diffs.append(r[0])

    # 3. Per-layer comparisons
    for layer_idx in range(NUM_HIDDEN_LAYERS):
        hf_prefix = f"model.layers.{layer_idx}"
        trt_prefix = f"transformer.layers.{layer_idx}"

        print(f"\n── Layer {layer_idx} ──")

        # Input layernorm
        r = compare_tensors(f"layer{layer_idx}.input_layernorm",
                            hf_weights[f"{hf_prefix}.input_layernorm.weight"],
                            trt_weights[f"{trt_prefix}.input_layernorm.weight"])
        if r: all_max_diffs.append(r[0])

        # Post-attention layernorm
        r = compare_tensors(f"layer{layer_idx}.post_attention_layernorm",
                            hf_weights[f"{hf_prefix}.post_attention_layernorm.weight"],
                            trt_weights[f"{trt_prefix}.post_layernorm.weight"])
        if r: all_max_diffs.append(r[0])

        # QKV (split vs fused)
        compare_qkv_split(hf_prefix, trt_prefix, layer_idx, hf_weights, trt_weights)

        # Attention output projection
        r = compare_tensors(f"layer{layer_idx}.o_proj",
                            hf_weights[f"{hf_prefix}.self_attn.o_proj.weight"],
                            trt_weights[f"{trt_prefix}.attention.dense.weight"])
        if r: all_max_diffs.append(r[0])

        # MLP gate
        r = compare_tensors(f"layer{layer_idx}.gate_proj",
                            hf_weights[f"{hf_prefix}.mlp.gate_proj.weight"],
                            trt_weights[f"{trt_prefix}.mlp.gate.weight"])
        if r: all_max_diffs.append(r[0])

        # MLP up (fc in TRT)
        r = compare_tensors(f"layer{layer_idx}.up_proj",
                            hf_weights[f"{hf_prefix}.mlp.up_proj.weight"],
                            trt_weights[f"{trt_prefix}.mlp.fc.weight"])
        if r: all_max_diffs.append(r[0])

        # MLP down (proj in TRT)
        r = compare_tensors(f"layer{layer_idx}.down_proj",
                            hf_weights[f"{hf_prefix}.mlp.down_proj.weight"],
                            trt_weights[f"{trt_prefix}.mlp.proj.weight"])
        if r: all_max_diffs.append(r[0])

    # ─── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    if all_max_diffs:
        global_max = max(all_max_diffs)
        global_mean = sum(all_max_diffs) / len(all_max_diffs)
        print(f"  Global max_diff across all layers: {global_max:.6e}")
        print(f"  Global mean max_diff:              {global_mean:.6e}")
        if global_max > 1e-4:
            print(f"  ⚠️  WARNING: max_diff ({global_max:.3e}) > 1e-4 — potential weight loading issue!")
        else:
            print(f"  ✅ All weights match within 1e-4 tolerance.")
    else:
        print("  No comparisons performed.")

    # Also report TRT-only keys (not compared)
    trt_only = set(trt_weights.keys()) - {
        "transformer.vocab_embedding.weight",
        "transformer.ln_f.weight",
        "lm_head.weight",
        *[f"transformer.layers.{i}.input_layernorm.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"transformer.layers.{i}.post_layernorm.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"transformer.layers.{i}.attention.qkv.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"transformer.layers.{i}.attention.dense.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"transformer.layers.{i}.mlp.gate.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"transformer.layers.{i}.mlp.fc.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"transformer.layers.{i}.mlp.proj.weight" for i in range(NUM_HIDDEN_LAYERS)],
    }
    if trt_only:
        print(f"\n  TRT-only keys (buffers/metadata, not compared): {sorted(trt_only)}")

    # HF-only keys that are weight-related (skip weighted_mask)
    hf_weight_keys = set(k for k in hf_weights.keys() if "weight" in k)
    hf_only = hf_weight_keys - {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        *[f"model.layers.{i}.input_layernorm.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"model.layers.{i}.post_attention_layernorm.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"model.layers.{i}.self_attn.q_proj.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"model.layers.{i}.self_attn.k_proj.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"model.layers.{i}.self_attn.v_proj.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"model.layers.{i}.self_attn.o_proj.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"model.layers.{i}.mlp.gate_proj.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"model.layers.{i}.mlp.up_proj.weight" for i in range(NUM_HIDDEN_LAYERS)],
        *[f"model.layers.{i}.mlp.down_proj.weight" for i in range(NUM_HIDDEN_LAYERS)],
    }
    if hf_only:
        print(f"\n  HF-only weight keys (not compared): {sorted(hf_only)}")


if __name__ == "__main__":
    main()