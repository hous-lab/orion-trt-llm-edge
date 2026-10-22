"""
End-to-end plan_L2 Accuracy Validation for Quantized Orion Models
==================================================================
Tests FP16/W8A16/W4A16 quantization on the full Orion model pipeline
using the tiny validation dataset.

Usage:
  python adzoo/orion/test_quant_plan_l2.py --mode fp16
  python adzoo/orion/test_quant_plan_l2.py --mode w8a16
  python adzoo/orion/test_quant_plan_l2.py --mode w4a16
  python adzoo/orion/test_quant_plan_l2.py --mode all
"""

import argparse
import os
import sys
import time
import gc
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmcv.utils import Config, load_checkpoint
from mmcv.models import build_model
from mmcv.datasets import build_dataset, build_dataloader
from adzoo.orion.apis.test import single_gpu_test


class W8A16Linear(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        weight_fp16 = original_linear.weight.data
        scale = weight_fp16.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp(min=1e-8)
        self.register_buffer("weight_int8", torch.round(weight_fp16 / scale).to(torch.int8))
        self.register_buffer("scale", scale.half())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.half())
        else:
            self.bias = None

    def forward(self, x):
        weight_fp16 = self.weight_int8.to(x.dtype) * self.scale
        return torch.nn.functional.linear(x, weight_fp16, self.bias)

    @property
    def weight(self):
        return self.weight_int8.half() * self.scale


class W4A16Linear(nn.Module):
    def __init__(self, original_linear, group_size=128):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.group_size = group_size
        weight_fp16 = original_linear.weight.data
        out_features, in_features = weight_fp16.shape
        assert in_features % group_size == 0
        weight_reshaped = weight_fp16.reshape(out_features, in_features // group_size, group_size)
        scale = weight_reshaped.abs().amax(dim=2, keepdim=True) / 7.0
        scale = scale.clamp(min=1e-8)
        weight_quant = torch.round(weight_reshaped / scale).clamp(-8, 7).to(torch.int8)
        weight_packed = self._pack_4bit(weight_quant)
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scale", scale.reshape(out_features, in_features // group_size).half())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.half())
        else:
            self.bias = None

    @staticmethod
    def _pack_4bit(weight_int8):
        weight_unsigned = (weight_int8 + 8).to(torch.uint8)
        even = weight_unsigned[..., 0::2]
        odd = weight_unsigned[..., 1::2]
        packed = (even | (odd << 4)).to(torch.uint8)
        return packed

    @staticmethod
    def _unpack_4bit(packed, num_elements):
        even_u = (packed & 0x0F).to(torch.int8) - 8
        odd_u = ((packed >> 4) & 0x0F).to(torch.int8) - 8
        result = torch.stack([even_u, odd_u], dim=-1).reshape(*packed.shape[:-1], num_elements)
        return result

    def forward(self, x):
        out_features = self.out_features
        in_features = self.in_features
        group_size = self.group_size
        num_groups = in_features // group_size
        weight_int8 = self._unpack_4bit(self.weight_packed, group_size)
        scale = self.scale.reshape(out_features, num_groups, 1)
        weight_fp16 = (weight_int8.reshape(out_features, num_groups, group_size).half() * scale).reshape(out_features, in_features)
        return torch.nn.functional.linear(x, weight_fp16, self.bias)

    @property
    def weight(self):
        group_size = self.group_size
        num_groups = self.in_features // group_size
        weight_int8 = self._unpack_4bit(self.weight_packed, group_size)
        scale = self.scale.reshape(self.out_features, num_groups, 1)
        return (weight_int8.reshape(self.out_features, num_groups, group_size).half() * scale).reshape(self.out_features, self.in_features)


def replace_linear_with_w8a16(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features > 256:
            setattr(module, name, W8A16Linear(child))
        else:
            replace_linear_with_w8a16(child)


def replace_linear_with_w4a16(module, group_size=128):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features > 256 and child.in_features % group_size == 0:
            setattr(module, name, W4A16Linear(child, group_size=group_size))
        else:
            replace_linear_with_w4a16(child, group_size=group_size)


def get_model_size_mb(model):
    total = 0
    for name, param in model.named_parameters():
        total += param.numel() * param.element_size()
    for name, buf in model.named_buffers():
        total += buf.numel() * buf.element_size()
    return total / (1024 ** 2)


custom_fp16 = dict(map_head=False, pts_bbox_head=False)


def custom_wrap_fp16_model(model):
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for module_name, v in custom_fp16.items():
        model._modules[module_name].fp16_enabled = v


def test_quantized_mode(config_path, checkpoint_path, quant_mode):
    """Run test with specified quantization mode."""
    from mmcv.datasets import replace_ImageToTensor
    
    print(f"\n{'='*70}")
    print(f"[INFO] === Testing: {quant_mode.upper()} ===")
    print(f"{'='*70}")
    
    cfg = Config.fromfile(config_path)
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )
    
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        custom_wrap_fp16_model(model)
    
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # Convert LLM to FP16 for quantization (LoRA weights need to be FP16)
    if hasattr(model, 'lm_head'):
        model.lm_head = model.lm_head.half()
    
    if quant_mode == "w8a16":
        print("[INFO] Applying W8A16 quantization to LLM...")
        if hasattr(model, "lm_head"):
            replace_linear_with_w8a16(model.lm_head)
    elif quant_mode == "w4a16":
        print("[INFO] Applying W4A16 quantization to LLM...")
        if hasattr(model, "lm_head"):
            replace_linear_with_w4a16(model.lm_head)
    
    model = DataParallel(model.cuda(), device_ids=[0])
    model.eval()
    
    model_size_mb = get_model_size_mb(model)
    print(f"[INFO] Model size ({quant_mode}): {model_size_mb:.0f} MB")
    
    print("[INFO] Running inference...")
    t0 = time.time()
    outputs = single_gpu_test(model, data_loader)
    elapsed = time.time() - t0
    print(f"[INFO] Inference time: {elapsed:.1f}s for {len(dataset)} samples")
    
    plan_L2_1s_list = []
    plan_L2_2s_list = []
    plan_L2_3s_list = []
    n_valid = 0
    
    # outputs is a dict with 'bbox_results' key
    bbox_results = outputs.get('bbox_results', outputs) if isinstance(outputs, dict) else outputs
    
    for out in bbox_results:
        if isinstance(out, dict):
            metric = out.get("metric_results", {})
            if metric.get("fut_valid_flag", False):
                plan_L2_1s_list.append(metric.get("plan_L2_1s", 0))
                plan_L2_2s_list.append(metric.get("plan_L2_2s", 0))
                plan_L2_3s_list.append(metric.get("plan_L2_3s", 0))
                n_valid += 1
    
    result = {
        "plan_L2_1s": np.mean(plan_L2_1s_list) if plan_L2_1s_list else float("nan"),
        "plan_L2_2s": np.mean(plan_L2_2s_list) if plan_L2_2s_list else float("nan"),
        "plan_L2_3s": np.mean(plan_L2_3s_list) if plan_L2_3s_list else float("nan"),
        "n_valid": n_valid,
        "n_total": len(outputs),
        "model_size_mb": model_size_mb,
        "inference_time_s": elapsed,
    }
    
    print(f"\n[INFO] {quant_mode.upper()} Results:")
    print(f"  plan_L2_1s: {result['plan_L2_1s']:.4f}")
    print(f"  plan_L2_2s: {result['plan_L2_2s']:.4f}")
    print(f"  plan_L2_3s: {result['plan_L2_3s']:.4f}")
    print(f"  Valid/Total: {n_valid}/{len(outputs)}")
    print(f"  Model size: {model_size_mb:.0f} MB")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="End-to-end plan_L2 Validation for Quantized Orion")
    parser.add_argument("--config", type=str, default="adzoo/orion/configs/orion_stage3_fp16.py")
    parser.add_argument("--checkpoint", type=str, default="ckpts/Orion.pth")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "fp16", "w8a16", "w4a16"])
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore")
    
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Mode: {args.mode}")
    
    modes = ["fp16", "w8a16", "w4a16"] if args.mode == "all" else [args.mode]
    all_results = {}
    
    for mode in modes:
        all_results[mode] = test_quantized_mode(args.config, args.checkpoint, mode)
    
    print(f"\n{'='*80}")
    print(f"[INFO] === plan_L2 Validation Summary ===")
    print(f"{'='*80}")
    print(f"  Original PyTorch FP16 baseline (from exploration notes):")
    print(f"    plan_L2_1s=0.6905, plan_L2_2s=1.4808, plan_L2_3s=2.3768")
    print()
    print(f"{'Mode':<10} {'L2_1s':<10} {'L2_2s':<10} {'L2_3s':<10} {'Size(MB)':<12} {'Valid':<8}")
    print(f"{'-'*60}")
    print(f"{'Baseline':<10} {'0.6905':<10} {'1.4808':<10} {'2.3768':<10} {'-':<12} {'-':<8}")
    for mode in all_results:
        r = all_results[mode]
        print(f"{mode:<10} {r['plan_L2_1s']:<10.4f} {r['plan_L2_2s']:<10.4f} {r['plan_L2_3s']:<10.4f} "
              f"{r['model_size_mb']:<12.0f} {r['n_valid']:<8}")
    
    print()
    if "fp16" in all_results:
        fp16_l2 = all_results["fp16"]["plan_L2_1s"]
        print(f"  FP16 vs Baseline plan_L2_1s: {fp16_l2:.4f} vs 0.6905 (delta={fp16_l2 - 0.6905:+.4f})")
    if "w8a16" in all_results and "fp16" in all_results:
        delta = all_results["w8a16"]["plan_L2_1s"] - all_results["fp16"]["plan_L2_1s"]
        print(f"  W8A16 vs FP16 plan_L2_1s delta: {delta:+.4f}")
        if abs(delta) < 0.1:
            print(f"  -> W8A16: Accuracy acceptable (delta < 0.1)")
        else:
            print(f"  -> W8A16: Significant accuracy change (delta >= 0.1)")
    if "w4a16" in all_results and "fp16" in all_results:
        delta = all_results["w4a16"]["plan_L2_1s"] - all_results["fp16"]["plan_L2_1s"]
        print(f"  W4A16 vs FP16 plan_L2_1s delta: {delta:+.4f}")
        if abs(delta) < 0.1:
            print(f"  -> W4A16: Accuracy acceptable (delta < 0.1)")
        else:
            print(f"  -> W4A16: Accuracy NOT acceptable (delta >= 0.1)")


if __name__ == "__main__":
    main()
