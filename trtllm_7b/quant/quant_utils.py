#!/usr/bin/env python
"""量化公共工具：RTN 量化、误差指标、权重分片加载、outlier 分析。

供 GPTQ / AWQ / SmoothQuant / 敏感度分析 复用。
"""
import os, glob
import numpy as np
import torch
import safetensors.torch as sf

ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not ROOT:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
DATA_DIR = os.path.join(ROOT, "ckpts", "quant_data", "shards")


def load_layer_shards(dtype=torch.float16):
    """加载全部分片 LLM 权重为 dict。"""
    layers = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.safetensors"))):
        d = sf.load_file(p)
        layers.update(d)
    if dtype is not None:
        layers = {k: v.to(dtype) for k, v in layers.items()}
    return layers


def sym_quant_group(W, bits=4, group_size=128):
    """对称 per-group 量化：返回 (q, scale)。
    W: [out, in]
    返回 q: [out, in] int tensor, scale: [out, n_group]
    """
    Wf = W.float()
    out_feat, in_feat = Wf.shape
    max_q = (1 << (bits - 1)) - 1  # 对称无零点：7 or 127
    n_groups = (in_feat + group_size - 1) // group_size if group_size > 0 else 1
    if group_size <= 0:
        group_size = in_feat
        n_groups = 1
    Wg = Wf.view(out_feat, n_groups, group_size)
    scale = Wg.abs().amax(dim=2).clamp(min=1e-8)  # [out, n_group]
    q = torch.round(Wg / scale.unsqueeze(-1) * max_q).clamp(-max_q, max_q)
    return q.view(out_feat, in_feat).to(torch.int8), scale


def dequant_group(q, scale, bits=4, group_size=128):
    """反量化。"""
    max_q = (1 << (bits - 1)) - 1
    out_feat, in_feat = q.shape
    n_groups = (in_feat + group_size - 1) // group_size if group_size > 0 else 1
    if group_size <= 0:
        group_size = in_feat
        n_groups = 1
    qf = q.float().view(out_feat, n_groups, group_size)
    s = scale.unsqueeze(-1)
    return (qf * s / max_q).view(out_feat, in_feat)


def quant_rtn_int4(W, group_size=128):
    q, s = sym_quant_group(W, bits=4, group_size=group_size)
    return dequant_group(q, s, bits=4, group_size=group_size)


def quant_rtn_int8(W, group_size=128):
    q, s = sym_quant_group(W, bits=8, group_size=group_size)
    return dequant_group(q, s, bits=8, group_size=group_size)


def quant_error(W, Wq):
    """量化误差：MSE + 相对 MSE + 余弦相似度。"""
    Wf = W.float()
    Wqf = Wq.float()
    mse = (Wf - Wqf).pow(2).mean().item()
    rel_mse = mse / Wf.pow(2).mean().clamp(min=1e-12).item()
    cos = torch.nn.functional.cosine_similarity(
        Wf.view(-1), Wqf.view(-1), dim=0).item()
    return {"mse": mse, "rel_mse": rel_mse, "cos": cos}


def per_channel_absmax(W):
    """per-channel absmax（用于 SmoothQuant 的 W scale）。"""
    return W.float().abs().amax(dim=0).clamp(min=1e-8)


def outlier_analysis(W, thresh_ratio=3.0):
    """权重异常值分析：统计 |W| > thresh_ratio * median 的元素占比。"""
    Wf = W.float()
    flat = Wf.view(-1)
    med = flat.abs().median().item()
    thresh = med * thresh_ratio
    n_out = (flat.abs() > thresh).sum().item()
    return {
        "median_abs": med, "thresh": thresh,
        "num_outliers": n_out, "outlier_ratio": n_out / flat.numel(),
        "max_abs": flat.abs().max().item(),
        "mean_abs": flat.abs().mean().item(),
    }


def load_single_layer(layer_idx=0, mod="self_attn.q_proj"):
    """加载单层权重。"""
    g = layer_idx // 8
    p = os.path.join(DATA_DIR, f"orion_llm_group_{g}.safetensors")
    d = sf.load_file(p)
    key = f"model.layers.{layer_idx}.{mod}.weight"
    return d[key]
