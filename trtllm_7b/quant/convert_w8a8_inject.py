#!/usr/bin/env python
"""Orion W8A8 SmoothQuant 转换（monkey-patch 注入器，服务器 24GB）。

在运行项目 convert_llm_checkpoint.py 前，注入：
  1. AutoModelForCausalLM.from_pretrained → 标准 Llama + 我们的 Orion 权重
  2. capture_activation_range → 用我们校准的 scale（补 y 用前向）
然后调用其 main()，走 TRT-LLM 原生 quantize() 流程。

用法:
  python convert_w8a8_inject.py --scale data/smoothquant_scale.pt --alpha 0.5
"""
import os, sys, argparse
import numpy as np
import torch
import safetensors.torch as sf
from transformers import LlamaConfig, LlamaForCausalLM
import importlib.util

ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not ROOT:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
DATA_DIR = os.path.join(ROOT, "ckpts", "quant_data")


def load_weights():
    weights = {}
    for p in sorted(os.listdir(os.path.join(DATA_DIR, "shards"))):
        if not p.endswith(".safetensors"):
            continue
        weights.update(sf.load_file(os.path.join(DATA_DIR, "shards", p)))
    return weights


def make_from_pretrained_hijack(weights, device_map="cuda"):
    def _from_pretrained(model_dir, **kwargs):
        print(f"[hijack] {model_dir} → 标准 Llama + Orion 权重")
        cfg = LlamaConfig(vocab_size=32001, hidden_size=4096, intermediate_size=11008,
                          num_hidden_layers=32, num_attention_heads=32,
                          num_key_value_heads=32, torch_dtype=torch.float16)
        model = LlamaForCausalLM(cfg).half()
        if kwargs.get("device_map") == "auto" or kwargs.get("device") != "cpu":
            model = model.cuda()
        sd = {}
        for k, v in weights.items():
            if k in model.state_dict():
                sd[k] = v
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_miss = sum(1 for m in missing if "layers" in m)
        print(f"[hijack] loaded={len(sd)}, missing={len(missing)} (layer={n_miss}), unexpected={len(unexpected)}")
        assert n_miss == 0, f"关键层未加载: {missing[:5]}"
        model.eval()
        return model
    return _from_pretrained


def make_capture_hijack(scale, emb_path):
    def _capture_act_range(model, tokenizer, dataset, *a, **kw):
        print(f"[hijack] 用预校准 scale ({len(scale)} 层)，补全 y...")
        act_range = {k: {"x": v["x"], "y": None, "w": v["w"]} for k, v in scale.items()}
        # 补充 TRT-LLM smooth_llama_model 期望的 qkv_proj / mlp.fc 等合并 key
        for name, m in model.named_modules():
            if "self_attn.q_proj" in name:
                base = name.replace(".self_attn.q_proj", "")
                qkv = base + ".self_attn.qkv_proj"
                if qkv not in act_range:
                    xq = act_range.get(name, {}).get("x")
                    act_range[qkv] = {"x": xq, "y": None, "w": None}
            if "mlp.gate_proj" in name:
                base = name.replace(".mlp.gate_proj", "")
                act_range.setdefault(base + ".mlp.fc", act_range.get(name))
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
            if name in act_range and isinstance(m, torch.nn.Linear):
                hooks.append(m.register_forward_hook(stat_hook(name)))
        emb = torch.load(emb_path, map_location="cpu").half().cuda()
        torch.manual_seed(0)
        idxs = torch.randint(0, emb.shape[0], (4, 529))
        with torch.no_grad():
            for i in range(4):
                model(inputs_embeds=emb[idxs[i]].unsqueeze(0))
        for h in hooks:
            h.remove()
        for k in act_range:
            if act_range[k] is not None and act_range[k]["y"] is None:
                act_range[k]["y"] = act_range[k]["x"].clone()
        return act_range
    return _capture_act_range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default=os.path.join(DATA_DIR, "smoothquant_scale.pt"))
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    # ---- 先修复 peft 坏依赖（awq 未安装）----
    import peft.import_utils as _piu
    _piu.is_auto_awq_available = lambda: False
    import peft.tuners.lora.awq as _la
    if _la.is_auto_awq_available():
        pass
    # 确保 lora.awq 模块加载时 is_auto_awq_available 已返回 False
    import importlib
    importlib.reload(_la)
    print("peft awq 依赖已屏蔽")

    scale = torch.load(args.scale, map_location="cpu")
    weights = load_weights()
    emb_path = os.path.join(ROOT, "ckpts/embed_tokens_orion_fp16.pt")

    # 直接 import tensorrt_llm（peft 已修复）
    sys.path.insert(0, ROOT)  # mmcv 需要项目根
    os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    import tensorrt_llm
    print(f"tensorrt_llm {tensorrt_llm.__version__}")

    import tensorrt_llm.models.llava_llama.convert as cvt
    cvt.AutoModelForCausalLM.from_pretrained = staticmethod(
        make_from_pretrained_hijack(weights))
    cvt.capture_activation_range = make_capture_hijack(scale, emb_path)
    # 修复 param_name_map：llava 分支应返回 model_prefix="model"（标准 Llama 前缀）
    orig_gpnpm = cvt.get_prefix_and_param_name_map

    def _gpnpm(architecture, use_safetensors=False):
        mp, lp, pnm = orig_gpnpm(architecture, use_safetensors)
        if any(a in architecture.lower() for a in ["llava"]):
            mp = "model"
        return mp, lp, pnm

    cvt.get_prefix_and_param_name_map = _gpnpm
    # smooth_llama_model 在 GPU 上计算 smoother，校准 scale 需同步到 GPU
    scale_gpu = {k: {"x": v["x"].cuda(), "y": v["y"].cuda() if v.get("y") is not None else v["x"].cuda(),
                     "w": v["w"].cuda()}
                 for k, v in scale.items()}
    torch.save(scale_gpu, "/tmp/scale_gpu.pt")
    cvt.capture_activation_range = make_capture_hijack(scale_gpu, emb_path)
    print("monkey-patch 注入完成 (scale → GPU, prefix → model)")

    # 调用项目 convert 脚本的 main（复用其参数解析）
    sys.path.insert(0, ROOT)
    sys.argv = [os.path.join(ROOT, "adzoo/orion/convert_llm_checkpoint.py"),
                "--model_dir", os.path.join(ROOT, "ckpts/llm_clean_no_lora_patched"),
                "--output_dir", os.path.join(ROOT, "ckpts/llm_clean_no_lora_patched/trt_sq"),
                "--dtype", "float16",
                "--smoothquant", str(args.alpha),
                "--per_channel",
                "--per_token"]
    from adzoo.orion.convert_llm_checkpoint import main as conv_main
    conv_main()


if __name__ == "__main__":
    main()