#!/usr/bin/env python
"""Orion W8A8 SmoothQuant 转换（monkey-patch 方式，服务器 24GB）。

策略：TRT-LLM 的 quantize() 流程完整（smooth_llama_model + load_weights），
唯一障碍是 AutoModelForCausalLM.from_pretrained 无法加载 llava_llama。
我们 monkey-patch 该函数：对 llava_llama 目录 → 加载标准 Llama + 我们的权重。
同时 patch capture_activation_range 用我们的校准 scale（跳过文本校准）。

用法:
  python convert_w8a8_patch.py \
    --scale data/smoothquant_scale.pt \
    --out ckpts/llm_clean_no_lora_patched/trt_sq
"""
import os, sys, json, copy, argparse, importlib
import numpy as np
import torch
import safetensors.torch as sf
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

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


def _from_pretrained_hijack(model_dir, **kwargs):
    """替代 AutoModelForCausalLM.from_pretrained：标准 Llama + 我们的权重。"""
    print(f"[hijack] 加载 {model_dir} → 标准 Llama + Orion 权重")
    cfg = LlamaConfig(vocab_size=32001, hidden_size=4096, intermediate_size=11008,
                      num_hidden_layers=32, num_attention_heads=32,
                      num_key_value_heads=32, torch_dtype=torch.float16)
    model = LlamaForCausalLM(cfg).half()
    if kwargs.get("device_map") == "auto" or "cuda" in str(kwargs.get("device", "")):
        model = model.cuda()
    sd = {}
    for k, v in load_weights().items():
        if k in model.state_dict():
            sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_miss = sum(1 for m in missing if "layers" in m)
    print(f"[hijack] loaded={len(sd)}, missing={len(missing)} (layer={n_miss}), unexpected={len(unexpected)}")
    assert n_miss == 0
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default=os.path.join(DATA_DIR, "smoothquant_scale.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "ckpts/llm_clean_no_lora_patched/trt_sq"))
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--calib-dataset", default="ccdv/cnn_dailymail")  # 占位
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- monkey-patch ----
    import tensorrt_llm.models.llava_llama.convert as cvt
    cvt.AutoModelForCausalLM.from_pretrained = staticmethod(_from_pretrained_hijack)

    # 校准数据：加载我们的 scale 后，capture_activation_range 仍需被执行（生成 act_range 结构）
    # 我们直接 patch 它的输入：让 tokenizer + dataset 走我们已校准的 scale
    scale = torch.load(args.scale, map_location="cpu")

    def _capture_act_range(model, tokenizer, dataset, *a, **kw):
        """替代 capture_activation_range：直接用校准好的 scale（补 y 用前向）。"""
        print("[hijack] 使用预校准 act_range，补全 y...")
        act_range = {k: {"x": v["x"], "y": None, "w": v["w"]} for k, v in scale.items()}
        hooks = []

        def stat_hook(name):
            def hook(m, x, y):
                if isinstance(y, tuple):
                    y = y[0]
                hd = y.shape[-1]
                ym = y.view(-1, hd).abs().detach().float()
                cur_y = torch.max(ym, dim=0)[0]
                if act_range[name]["y"] is None:
                    act_range[name]["y"] = cur_y
                else:
                    act_range[name]["y"] = torch.max(act_range[name]["y"], cur_y)
            return hook

        for name, m in model.named_modules():
            if name in act_range:
                hooks.append(m.register_forward_hook(stat_hook(name)))
        # 用 embed 前向补 y
        emb = torch.load(os.path.join(ROOT, "ckpts/embed_tokens_orion_fp16.pt"), map_location="cpu").half().cuda()
        torch.manual_seed(0)
        idxs = torch.randint(0, emb.shape[0], (4, 529))
        with torch.no_grad():
            for i in range(4):
                model(inputs_embeds=emb[idxs[i]].unsqueeze(0))
        for h in hooks:
            h.remove()
        for k in act_range:
            if act_range[k]["y"] is None:
                act_range[k]["y"] = act_range[k]["x"].clone()
        return act_range

    cvt.capture_activation_range = _capture_act_range
    print("monkey-patch 完成")

    # ---- 走 TRT-LLM 原生 quantize() ----
    from tensorrt_llm.models.llava_llama.config import LlavaLlamaConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo
    from tensorrt_llm.mapping import Mapping

    # 复用项目 convert 脚本的参数构造（简化）
    from tensorrt_llm.models.llava_llama.model import LlavaLlamaForCausalLM
    trt_cfg = LlavaLlamaConfig(
        model_type="llava_llama", architecture="LlavaLlamaForCausalLM",
        vocab_size=32001, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=4096, dtype="float16",
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        quantization=QuantConfig(
            quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
            smoothquant_val=args.alpha))
    trt_cfg.set_rank(0)

    # quantize() 内部：from_pretrained(被 hijack) → smooth_llama_model → load_weights
    LlavaLlamaForCausalLM.quantize(
        argsheet := "",  # placeholder 不存在
        args.out,
        dtype="float16", mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        quant_config=trt_cfg.quantization, device="cuda",
        calib_dataset=args.calib_dataset,
        **trt_cfg.to_dict())
    print(f"量化权重保存到 {args.out}")


if __name__ == "__main__":
    main()