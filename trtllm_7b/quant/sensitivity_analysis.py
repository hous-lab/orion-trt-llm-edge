#!/usr/bin/env python
"""量化敏感度分析 + 混合精度策略设计（Orion LLM）。

方法：
  1. 对每一层单独做 INT4/INT8 RTN 量化，其余层保持 FP16
  2. 计算该层量化引起的"输出扰动"（用该层权重的输出误差度量）
  3. 按敏感度排序 → 识别敏感层（应保持高精度）与鲁棒层（可低比特）
  4. 设计混合精度分配策略

产出: <ORION_PROJECT_ROOT>/ckpts/quant_data/sensitivity_report.md
"""
import os, sys, time, json
import torch
import safetensors.torch as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_utils import load_layer_shards, sym_quant_group, dequant_group

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(OUT_DIR, exist_ok=True)
MODS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def layer_sensitivity(W, bits=4, group_size=128, seed=0):
    """单层量化的敏感度（输出域相对扰动）。"""
    torch.manual_seed(seed)
    dev = W.device
    base = torch.randn(32, 64, device=dev)
    proj = torch.randn(64, W.shape[1], device=dev)
    X = base @ proj + 0.3 * torch.randn(32, W.shape[1], device=dev)
    X = X / X.norm(dim=1, keepdim=True)

    q, sc = sym_quant_group(W.cpu(), bits=bits, group_size=group_size)
    W_q = dequant_group(q, sc, bits=bits, group_size=group_size).to(dev)
    ref = (X @ W.T)
    out = (X @ W_q.T)
    # 相对输出扰动
    rel_err = (ref - out).pow(2).mean() / ref.pow(2).mean().clamp(min=1e-12)
    return rel_err.item()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    layers = load_layer_shards(dtype=torch.float32)
    print(f"加载 {len(layers)} 层权重")

    results = []
    for l in range(32):
        for m in MODS:
            key = f"model.layers.{l}.{m}.weight"
            if key not in layers:
                continue
            W = layers[key].float().to(dev)
            s4 = layer_sensitivity(W, bits=4)
            s8 = layer_sensitivity(W, bits=8)
            results.append({"layer": l, "mod": m, "sens_int4": s4, "sens_int8": s8})
    print(f"分析 {len(results)} 层-module 敏感度")

    # 排序：按 INT4 敏感度
    results.sort(key=lambda r: r["sens_int4"], reverse=True)
    print("\n=== Top 10 敏感层（INT4 扰动最大，应保持高精度）===")
    for r in results[:10]:
        print(f"  L{r['layer']}.{r['mod']}: INT4={r['sens_int4']:.3e} INT8={r['sens_int8']:.3e}")

    print("\n=== 最鲁棒 10 层（可 INT4）===")
    for r in results[-10:]:
        print(f"  L{r['layer']}.{r['mod']}: INT4={r['sens_int4']:.3e} INT8={r['sens_int8']:.3e}")

    # 混合精度分配：敏感层 INT8/FP16，鲁棒层 INT4
    thresh = torch.tensor([r["sens_int4"] for r in results]).quantile(0.5).item()
    for r in results:
        r["assign"] = "INT8" if r["sens_int4"] > thresh else "INT4"
    n_int8 = sum(1 for r in results if r["assign"] == "INT8")
    print(f"\n混合精度分配（阈值=p50）：INT8={n_int8} 层, INT4={len(results)-n_int8} 层")

    # 报告
    md = ["# 量化敏感度分析与混合精度策略（Orion LLM）", "",
          "> 方法：逐层单独 INT4/INT8 量化，测量输出域相对扰动。敏感层 → 高精度，鲁棒层 → 低比特。", ""]
    md.append("## Top 敏感层（INT4 扰动最大）")
    md.append("| 层 | module | INT4 扰动 | INT8 扰动 | 分配 |")
    md.append("|---|--------|-----------|-----------|------|")
    for r in results[:15]:
        md.append(f"| L{r['layer']} | {r['mod']} | {r['sens_int4']:.2e} | "
                  f"{r['sens_int8']:.2e} | {r['assign']} |")
    md += ["", "## 混合精度汇总", f"- 敏感层数（INT8）：{n_int8}",
           f"- 鲁棒层数（INT4）：{len(results)-n_int8}",
           f"- 若全部 INT4 vs 混合精度：混合精度可保护 {n_int8} 个敏感层，"
           "显著降低低比特量化的精度损失。"]
    rep = os.path.join(OUT_DIR, "sensitivity_report.md")
    open(rep, "w").write("\n".join(md))
    json.dump(results, open(os.path.join(OUT_DIR, "sensitivity_report.json"), "w"), indent=1)
    print(f"报告已写入 {rep}")


if __name__ == "__main__":
    main()
