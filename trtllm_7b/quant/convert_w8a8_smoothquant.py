#!/usr/bin/env python
"""Orion W8A8 SmoothQuant 转换脚本（服务器 24GB 执行）。

绕过 TRT-LLM 的 llava 加载限制：
  1. 用标准 Llama + 我们的 Orion 权重（已验证 291 全加载）
  2. 加载 smoothquant_calib.py 校准的 scale 作为 act_range
  3. 调用 TRT-LLM 的 smooth_llama_model + load_weights_from_hf_model
  4. 保存 TRT-LLM checkpoint → trtllm-build

用法:
  python convert_w8a8_smoothquant.py \
    --scale data/smoothquant_scale.pt \
    --out ckpts/llm_clean_no_lora_patched/trt_sq
"""
import os, sys, json, copy, argparse, importlib
import numpy as np
import torch
import safetensors.torch as sf
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not ROOT:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATA_DIR = os.path.join(ROOT, "ckpts", "quant_data")


def load_weights():
    weights = {}
    for p in sorted(os.listdir(os.path.join(DATA_DIR, "shards"))):
        if not p.endswith(".safetensors"):
            continue
        weights.update(sf.load_file(os.path.join(DATA_DIR, "shards", p)))
    return weights


def load_into_model(model, weights):
    model_keys = set(model.state_dict().keys())
    sd = {k: v for k, v in weights.items() if k in model_keys}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_miss_layer = sum(1 for m in missing if "layers" in m)
    print(f"load: loaded={len(sd)}, missing={len(missing)} (layer={n_miss_layer}), unexpected={len(unexpected)}")
    assert n_miss_layer == 0, "关键层权重未加载！"


def build_act_range_complete(scale, model, n_cal=8):
    """把我们的 x/w scale 补全为 TRT-LLM 需要的 {x, y, w} 格式。

    act_range[name] = {"x": [in_feat], "y": [out_feat], "w": [out_feat]}
    用模型前向捕获 y（输出侧 max）。
    """
    act_range = {}
    hooks = []

    def stat_hook(name):
        def hook(m, x, y):
            if isinstance(x, tuple):
                x = x[0]
            if isinstance(y, tuple):
                y = y[0]
            hd_y = y.shape[-1]
            ymax = y.view(-1, hd_y).abs().detach().float()
            cur_y = torch.max(ymax, dim=0)[0]
            if act_range[name].get("y") is None:
                act_range[name]["y"] = cur_y
            else:
                act_range[name]["y"] = torch.max(act_range[name]["y"], cur_y)
        return hook

    # 初始化 act_range 结构（从校准 scale）
    for k, v in scale.items():
        act_range[k] = {"x": v["x"], "y": None, "w": v["w"]}

    # 注册 hook + 前向捕获 y
    for name, m in model.named_modules():
        if name in act_range:
            hooks.append(m.register_forward_hook(stat_hook(name)))

    torch.manual_seed(0)
    emb = torch.load(os.path.join(ROOT, "ckpts", "embed_tokens_orion_fp16.pt"), map_location="cpu").to("cuda")
    idxs = torch.randint(0, emb.shape[0], (n_cal, 529))
    with torch.no_grad():
        for i in range(n_cal):
            act = emb[idxs[i]].unsqueeze(0).half()
            model(inputs_embeds=act)
            print(f"  y-capture 帧 {i+1}/{n_cal}")
    for h in hooks:
        h.remove()
    # 清理未捕获 y（lm_head 等非 Linear hook，用 x 近似）
    for k in act_range:
        if act_range[k]["y"] is None:
            act_range[k]["y"] = act_range[k]["x"].clone()
    return act_range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default=os.path.join(DATA_DIR, "smoothquant_scale.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "ckpts/llm_clean_no_lora_patched/trt_sq"))
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import tensorrt_llm
    from tensorrt_llm.models.llava_llama.convert import (
        smooth_llama_model, load_weights_from_hf_model, get_prefix_and_param_name_map)
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo
    from tensorrt_llm.mapping import Mapping

    cfg = LlamaConfig(vocab_size=32001, hidden_size=4096, intermediate_size=11008,
                      num_hidden_layers=32, num_attention_heads=32,
                      num_key_value_heads=32, torch_dtype=torch.float16)
    model = LlamaForCausalLM(cfg).half().cuda()
    load_into_model(model, load_weights())
    model.eval()

    print(f"加载校准 scale: {args.scale}")
    scale = torch.load(args.scale, map_location="cpu")
    print(f"  scale 层数: {len(scale)}")

    print("补全 act_range（捕获 y）...")
    act_range = build_act_range_complete(scale, model)

    # 构造 LlavaLlamaConfig（TRT-LLM 需要）
    from tensorrt_llm.models.llava_llama.config import LlavaLlamaConfig
    trt_cfg = LlavaLlamaConfig(
        model_type="llava_llama",
        architecture="LlavaLlamaForCausalLM",
        vocab_size=32001, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=4096, dtype="float16",
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        quantization=QuantConfig(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
                                 smoothquant_val=args.alpha))
    print("手动构造 act_range（跳过 TRT-LLM 内部校准）...")

    qkv_para, smoother = {}, {}
    # smooth_llama_model 需要 LlamaDecoderLayer 模块——我们的标准 Llama 满足
    smooth_llama_model(model, act_range, args.alpha, qkv_para, smoother)
    print(f"smooth 完成: qkv_para={len(qkv_para)}, smoother={len(smoother)}")

    # 生成量化权重
    weights = load_weights_from_hf_model(
        model, config=trt_cfg, act_range=act_range, qkv_para=qkv_para, smoother=smoother)
    print(f"量化权重: {len(weights)} 个 tensor")

    # 保存 safetensors
    # 按 rank 保存（手动处理 config）
    weights_cpu = {k: v.cpu().contiguous() for k, v in weights.items()}
    sf.save_file(weights_cpu, os.path.join(args.out, "rank0.safetensors"))
    print(f"已保存量化权重到 {args.out}/rank0.safetensors "
          f"({os.path.getsize(os.path.join(args.out, 'rank0.safetensors'))/1e9:.1f} GB)")

    # 保存 config.json（TRT-LLM checkpoint config）
    import json
    json.dump(trt_cfg.to_dict(), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print("已保存 config.json")


if __name__ == "__main__":
    main()