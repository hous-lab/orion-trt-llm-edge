#!/usr/bin/env python
"""SmoothQuant 校准（服务器 24GB）—— 标准 Llama + Orion 权重。

原理：Orion LLM 权重已是标准 Llama 格式（ckpts/quant_data/shards/），
构造 LlamaForCausalLM 加载，用校准激活跑前向，hook 捕获每层 Linear 激活
absmax → 计算 SmoothQuant per-channel scale。

校准输入：用提取的 embed_tokens 生成真实感的视觉激活序列（近似 Orion 的
529 visual tokens），SmoothQuant 校准对输入分布鲁棒。

产出: <ORION_PROJECT_ROOT>/ckpts/quant_data/smoothquant_scale.pt
"""
import os, sys, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import safetensors.torch as sf
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not ROOT:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
DATA_DIR = os.path.join(ROOT, "ckpts", "quant_data")


def load_all_weights():
    """加载全部 Orion LLM 标准权重。"""
    weights = {}
    for p in sorted(os.listdir(os.path.join(DATA_DIR, "shards"))):
        if not p.endswith(".safetensors"):
            continue
        d = sf.load_file(os.path.join(DATA_DIR, "shards", p))
        weights.update(d)
    print(f"加载 {len(weights)} 个权重张量")
    return weights


def load_state_into_model(model, weights):
    """把我们的权重加载进标准 Llama 模型（LlamaForCausalLM 前缀是 model. 无嵌套）。"""
    sd = {}
    model_keys = set(model.state_dict().keys())
    for k, v in weights.items():
        # 我们的 key 已是 model.layers.X / model.embed_tokens 等，与 LlamaForCausalLM 一致
        if k in model_keys:
            sd[k] = v
        elif "lm_head" in k and "lm_head.weight" in k:
            sd["lm_head.weight"] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss = [m for m in missing if "layers" in m]
    print(f"load: loaded={len(sd)}, missing={len(missing)} (其中层权重{len(miss)}), unexpected={len(unexpected)}")
    for m in list(missing)[:5]:
        print(f"  MISSING: {m}")
    for u in list(unexpected)[:5]:
        print(f"  UNEXPECTED: {u}")
    return sd


def build_calib_input(n_tokens=529, hidden=4096, n_frames=8, seed=0, device="cuda"):
    """构造校准激活序列：随机但匹配 Orion 视觉 token 分布。"""
    torch.manual_seed(seed)
    # 用 embed_tokens 的统计生成真实感激活
    emb_path = os.path.join(ROOT, "ckpts", "embed_tokens_orion_fp16.pt")
    emb = torch.load(emb_path, map_location="cpu")
    emb = emb.to(device)
    # 从 embedding 表采样 32001 个 token 的权重作为激活基
    idxs = torch.randint(0, emb.shape[0], (n_frames, n_tokens))
    act = emb[idxs]  # [n_frames, 529, 4096]
    return act


@torch.no_grad()
def calibrate(model, calib_acts, seed=0):
    """hook 捕获激活，返回 per-layer x/w absmax。"""
    model.eval()
    act_scales = {}

    def stat_tensor(name, tensor):
        hd = tensor.shape[-1]
        t = tensor.view(-1, hd).abs().detach().float()
        cur = torch.max(t, dim=0)[0]
        if name not in act_scales:
            act_scales[name] = cur
        else:
            act_scales[name] = torch.max(act_scales[name], cur)

    def stat_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name + ".x", x)
        stat_tensor(name + ".y", y)
        # 权重 per-channel absmax（SmoothQuant 需要 w 侧 max）
        if m.weight is not None:
            wmax = m.weight.abs().clip(1e-8, None).max(dim=0)[0].detach().float()
            if name + ".w" not in act_scales:
                act_scales[name + ".w"] = wmax
            else:
                act_scales[name + ".w"] = torch.max(act_scales[name + ".w"], wmax)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(
                lambda mod, x, y, n=name: stat_hook(mod, x, y, n)))

    for i, act in enumerate(calib_acts):
        # [1, 529, 4096] 直接作为 input 前向（Llama 以 embeds 输入）
        out = model(inputs_embeds=act.unsqueeze(0))
        print(f"  帧 {i+1}/{len(calib_acts)} 前向 OK, out.logits={tuple(out.logits.shape)}")
        gc.collect()

    for h in hooks:
        h.remove()

    # 归一化为 {name: {"x": ..., "w": ...}}（SmoothQuant 需要的格式）
    result = {}
    for name, v in act_scales.items():
        base = name.rsplit(".", 1)[0]
        suf = name.rsplit(".", 1)[1]
        if suf == "x":
            result.setdefault(base, {})["x"] = v
        elif suf == "w":
            result.setdefault(base, {})["w"] = v
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "smoothquant_scale.pt"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    print("加载 Orion LLM 标准权重...")
    weights = load_all_weights()

    print("构造 Llama 模型（7B，~13.5GB）...")
    cfg = LlamaConfig(vocab_size=32001, hidden_size=4096, intermediate_size=11008,
                      num_hidden_layers=32, num_attention_heads=32,
                      num_key_value_heads=32, torch_dtype=torch.float16)
    model = LlamaForCausalLM(cfg)
    model = model.half().to(dev)
    load_state_into_model(model, weights)
    del weights
    gc.collect()

    print("构建校准激活...")
    calib = build_calib_input(n_frames=args.n_frames, device=dev)

    print("校准...")
    act_scales = calibrate(model, calib)
    torch.save(act_scales, args.out)
    print(f"保存 scale: {args.out} ({sum(1 for v in act_scales.values() if isinstance(v, dict))} 层)")


if __name__ == "__main__":
    main()