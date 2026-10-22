#!/usr/bin/env python
"""从 patched checkpoint 加载 Orion LLM 权重（已合并 LoRA、vocab=32001），
导出干净的 Llama 层权重供量化算法复用。

patched checkpoint（ckpts/llm_clean_no_lora_patched/）已经是干净 Llama 权重：
  model.layers.X.<mod>.weight
  model.embed_tokens.weight / model.norm.weight / lm_head.weight

产出: <ORION_PROJECT_ROOT>/ckpts/quant_data/shards/*.safetensors
"""
import os, json, torch
import safetensors.torch as sf

ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not ROOT:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
CKPT_DIR = os.path.join(ROOT, "ckpts/llm_clean_no_lora_patched")
OUT = os.path.join(ROOT, "ckpts", "quant_data")
os.makedirs(OUT, exist_ok=True)

idx = json.load(open(os.path.join(CKPT_DIR, "model.safetensors.index.json")))
weight_map = idx["weight_map"]

# 按 shard 加载
shards = {}
for k, f in weight_map.items():
    if f not in shards:
        shards[f] = os.path.join(CKPT_DIR, f)
        break
print(f"shard files: {len(set(weight_map.values()))}")

result = {}
for fname in sorted(set(weight_map.values())):
    p = os.path.join(CKPT_DIR, fname)
    d = sf.load_file(p)
    for k, v in d.items():
        result[k] = v
    print(f"  loaded {fname}: {len(d)} tensors")

total = sum(v.numel() for v in result.values())
print(f"提取 {len(result)} 个张量，总参数 {total/1e9:.1f}B")
for k in list(result)[:4]:
    print(f"  {k}: {tuple(result[k].shape)} {result[k].dtype}")

# 分片保存：每层一个 safetensors（避免单次大文件写入失败）
SHARD_DIR = os.path.join(OUT, "shards")
os.makedirs(SHARD_DIR, exist_ok=True)
# 每 8 层一组（32 层 → 4 个 shard），加上 embed/norm/lm_head
groups = {}
for k in result:
    if "model.layers." in k:
        l = int(k.split("model.layers.")[1].split(".")[0])
        groups.setdefault(l // 8, {})[k] = result[k]
    else:
        groups.setdefault("misc", {})[k] = result[k]
for g, tensors in groups.items():
    p = os.path.join(SHARD_DIR, f"orion_llm_group_{g}.safetensors")
    sf.save_file({k: v.contiguous() for k, v in tensors.items()}, p)
    print(f"  保存 {os.path.basename(p)}: {len(tensors)} tensors, {os.path.getsize(p)/1e9:.1f} GB")
print(f"完成，分片目录: {SHARD_DIR}")


