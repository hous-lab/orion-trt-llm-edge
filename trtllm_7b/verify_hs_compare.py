#!/usr/bin/env python
"""
verify_hs_compare.py — 对比 TRT engine 与 PyTorch 原版的 hidden_states

设计目标：
  1. 在同一帧输入上，分别运行 PyTorch LLM 和 TRT-LLM engine
  2. 对比 full_hidden_states 在关键 token 位置（waypoint、文本、视觉）的数值
  3. 输出 max_diff, mean_diff, cos_sim 结构化结果

运行模式（自动选择）：
  - full:  完整对比，加载 PyTorch LLM + TRT Vision + TRT LLM（需 ~20GB GPU）
  - embed: 仅对比 embed_tokens 权重（需 ~8GB GPU，无需 Vision engine）

用法：
  完整模式:
    cd $ORION_PROJECT_ROOT
    $CONDA_PREFIX/bin/python \\
        verify_hs_compare.py

  仅权重对比:
    $CONDA_PREFIX/bin/python \\
        verify_hs_compare.py --mode embed

  embed_tokens 对比，只需指定 embed_tokens 文件:
    $CONDA_PREFIX/bin/python \\
        verify_hs_compare.py --mode embed \\
        --embed-tokens ckpts/embed_tokens_orion_fp16.pt

输出:
  === 维度 2: embed_tokens 权重对比 ===
  === 维度 1: embedding 注入对比 ===
  === 维度 6: full_hidden_states 对比 ===

依赖:
  - conda env with Python 3.8 + TRT-LLM v0.13 dev (see docs/environment.md)
  - TRT-LLM 0.13.0 (已修改 modeling_utils.py 插入 mark_output 'full_hidden_states')
  - TensorRT 10.4.0.26
"""

import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# 默认路径配置（与 run_trt_eval.sh 一致）
# ============================================================
DEFAULT_PATHS = {
    'llm_engine_dir': 'work_dirs/llm_safetensor/x86_1gpu_w8abf16_fp32attn',
    'vision_engine_path': 'work_dirs/vision_mixed_precision.engine',
    'tokenizer_path': 'ckpts/hf_models/pretrain_qformer/',
    'hf_model_path': 'ckpts/llm_clean_no_lora_patched',
    'embed_tokens_path': 'ckpts/embed_tokens_orion_fp16.pt',
    'orion_pth_path': 'ckpts/Orion.pth',
    'config_path': 'adzoo/orion/configs/orion_stage3.py',
    'checkpoint_path': 'ckpts/Orion.pth',
    'data_root': 'data/bench2drive',
    'ann_file': 'data/infos-tiny/b2d_infos_val.pkl',
    'map_file': 'data/infos-tiny/b2d_map_infos.pkl',
    'map_root': 'data/bench2drive/maps',
}


def resolve_path(rel_path):
    """将相对路径（相对于 PROJECT_ROOT）转为绝对路径."""
    if rel_path.startswith('/'):
        return rel_path
    return os.path.join(PROJECT_ROOT, rel_path)


def get_gpu_free_memory(device=0):
    """获取 GPU 可用显存（GB）."""
    if not torch.cuda.is_available():
        return 0
    with torch.cuda.device(device):
        total = torch.cuda.get_device_properties(device).total_memory
        free = total - torch.cuda.memory_allocated(device)
    return free / (1024 ** 3)


# ============================================================
# 工具函数
# ============================================================

def compare_tensors(name, a, b, prefix=''):
    """对比两个 tensor，输出 max_diff, mean_diff, cos_sim."""
    if a is None or b is None:
        print(f"  {prefix}[SKIP] {name}: one of the tensors is None")
        return None
    if a.shape != b.shape:
        print(f"  {prefix}[SKIP] {name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return None

    a_f = a.float().cpu()
    b_f = b.float().cpu()

    max_diff = (a_f - b_f).abs().max().item()
    mean_diff = (a_f - b_f).abs().mean().item()

    # cos_sim
    a_flat = a_f.flatten()
    b_flat = b_f.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

    print(f"  {prefix}{name}:")
    print(f"    shape      = {tuple(a.shape)}")
    print(f"    max_diff   = {max_diff:.6e}")
    print(f"    mean_diff  = {mean_diff:.6e}")
    print(f"    cos_sim    = {cos_sim:.6f}")

    # 摘要判定
    if max_diff < 1e-4:
        verdict = "✅ PASS (max_diff < 1e-4)"
    elif max_diff < 1e-2:
        verdict = "⚠️  WARN (max_diff < 1e-2)"
    else:
        verdict = "❌ FAIL (max_diff >= 1e-2)"
    print(f"    verdict    = {verdict}")

    return {'max_diff': max_diff, 'mean_diff': mean_diff, 'cos_sim': cos_sim}


def print_section_header(title):
    """打印分段标题."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ============================================================
# 维度 2: embed_tokens 权重对比
# ============================================================

def compare_embed_tokens(embed_tokens_path, llm_engine_dir):
    """
    对比两条路径提取的 embed_tokens：
      - Path A: extract_embed_tokens.py 从 Orion.pth 提取的 embed_tokens_orion_fp16.pt
      - Path B: TRT engine 内部编译的 vocab_embedding（通过 forward 提取）
    """
    print_section_header("维度 2: embed_tokens 权重对比")

    # ---- Path A: 从文件加载 ----
    pt_path = resolve_path(embed_tokens_path)
    if not os.path.exists(pt_path):
        print(f"  [SKIP] embed_tokens file not found: {pt_path}")
        return None

    embed_from_pt = torch.load(pt_path, map_location='cpu')  # (32001, 4096) fp16
    print(f"  [Path A] embed_tokens_orion_fp16.pt: {tuple(embed_from_pt.shape)}, {embed_from_pt.dtype}")

    # ---- Path B: 从 TRT engine 提取 ----
    # 通过 forward input_ids=arange(0, vocab_size) 来提取 engine 内部的 vocab_embedding
    # 但这需要加载 LLM engine 并运行 forward，比较重
    # 简化：直接对比 patched checkpoint 中的 embed_tokens 与 Orion.pth 提取的
    # 因为 patched checkpoint 的 embed_tokens 就是用 patch_llm_checkpoint.py 从 Orion.pth 写入的
    hf_model_path = resolve_path(DEFAULT_PATHS['hf_model_path'])
    hf_config_path = os.path.join(hf_model_path, 'config.json')

    if not os.path.exists(hf_config_path):
        print(f"  [SKIP] HF checkpoint not found: {hf_model_path}")
        return None

    with open(hf_config_path, 'r') as f:
        hf_config = json.load(f)

    # 从 HuggingFace safetensors 加载 embed_tokens 权重
    from safetensors.torch import load_file as safe_load

    # 加载 model.safetensors.index.json 获取分片信息
    index_path = os.path.join(hf_model_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        weight_map = index_data.get('weight_map', {})
        # 找到 embed_tokens 所在的文件
        embed_key = 'model.embed_tokens.weight'
        if embed_key in weight_map:
            shard_file = weight_map[embed_key]
            shard_path = os.path.join(hf_model_path, shard_file)
            if os.path.exists(shard_path):
                shard_sd = safe_load(shard_path)
                embed_from_hf = shard_sd[embed_key]
                print(f"  [Path B] HF checkpoint embed_tokens: {tuple(embed_from_hf.shape)}, {embed_from_hf.dtype}")

                # 对比
                compare_tensors('embed_tokens (Path A vs Path B)', embed_from_pt, embed_from_hf)
                return embed_from_pt, embed_from_hf
            else:
                print(f"  [SKIP] Shard file not found: {shard_path}")
        else:
            print(f"  [SKIP] embed_tokens key not found in weight map")
    else:
        print(f"  [SKIP] model.safetensors.index.json not found")

    return None


# ============================================================
# 维度 1: embedding 注入对比
# ============================================================

def compare_embedding_injection(pytorch_model, trt_engine, input_ids, vision_embeded):
    """
    对比 embedding 注入阶段：
      - PyTorch: prepare_inputs_labels_for_multimodal → inputs_embeds
      - TRT: prepare_inputs_labels_for_multimodal_ → prompt_table + input_ids
    """
    print_section_header("维度 1: embedding 注入对比")

    # 检测 PyTorch model 所在设备
    pt_device = next(pytorch_model.parameters()).device
    print(f"  PyTorch model device: {pt_device}")
    print(f"  vision_embeded device: {vision_embeded.device}")
    print(f"  input_ids device: {input_ids.device}")

    # ---- PyTorch 路径 ----
    # 将输入移到 PyTorch model 所在设备（CPU 模式时 vision_embeded 和 input_ids 需要从 GPU 移到 CPU）
    pt_input_ids = input_ids.to(pt_device)
    pt_vision_embeded = vision_embeded.to(pt_device)

    # 使用模型的 prepare_inputs_labels_for_multimodal 方法
    # 返回: None, position_ids, attention_mask, past_key_values, inputs_embeds, labels, new_input_ids
    pt_result = pytorch_model.prepare_inputs_labels_for_multimodal(
        input_ids=pt_input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        image_features=pt_vision_embeded,
        image_sizes=None,
    )
    _, _, _, _, pt_inputs_embeds, _, pt_new_input_ids = pt_result
    # pt_inputs_embeds: (1, seq_len, 4096), pt_new_input_ids: (1, seq_len)

    # ---- TRT 路径 ----
    trt_inputs_embeds, trt_new_input_ids = trt_engine.prepare_inputs_labels_for_multimodal_(
        input_ids, vision_embeded
    )
    # trt_inputs_embeds: (1, seq_len, 4096), trt_new_input_ids: (1, seq_len)

    if pt_inputs_embeds is None:
        print("  [SKIP] PyTorch returned None inputs_embeds (no image tokens in input)")
        return

    # 1. 对比 inputs_embeds
    compare_tensors('inputs_embeds (full seq)', pt_inputs_embeds, trt_inputs_embeds, prefix='[A] ')

    # 2. 分别对比文本段和视觉段
    # 找出视觉 token 位置
    IMAGE_TOKEN_INDEX = -200
    vis_mask_pt = (pt_new_input_ids[0] == IMAGE_TOKEN_INDEX)
    text_mask_pt = ~vis_mask_pt

    # TRT 侧的视觉 token ID 是 >= 32001
    vocab_size = 32001
    vis_mask_trt = (trt_new_input_ids[0] >= vocab_size)
    text_mask_trt = ~vis_mask_trt

    # 文本段对比
    if text_mask_pt.sum() > 0 and text_mask_trt.sum() > 0:
        pt_text = pt_inputs_embeds[0, text_mask_pt]
        trt_text = trt_inputs_embeds[0, text_mask_trt]
        if pt_text.shape == trt_text.shape:
            compare_tensors('inputs_embeds (text tokens)', pt_text, trt_text, prefix='[B] ')

    # 视觉段对比
    if vis_mask_pt.sum() > 0 and vis_mask_trt.sum() > 0:
        pt_vis = pt_inputs_embeds[0, vis_mask_pt]
        trt_vis = trt_inputs_embeds[0, vis_mask_trt]
        if pt_vis.shape == trt_vis.shape:
            compare_tensors('inputs_embeds (vision tokens)', pt_vis, trt_vis, prefix='[C] ')

    # 3. 对比 new_input_ids（文本 token 部分）
    if text_mask_pt.sum() > 0 and text_mask_trt.sum() > 0:
        pt_text_ids = pt_new_input_ids[0, text_mask_pt]
        trt_text_ids = trt_new_input_ids[0, text_mask_trt]
        if pt_text_ids.shape == trt_text_ids.shape:
            # 确保在同一设备上比较
            id_match = (pt_text_ids.cpu() == trt_text_ids.cpu()).all().item()
            print(f"  [D] text token IDs match: {id_match}")
            if not id_match:
                diff_pos = (pt_text_ids != trt_text_ids).nonzero(as_tuple=True)[0]
                print(f"      diff at {len(diff_pos)} positions: first few: {diff_pos[:5].tolist()}")

    return pt_inputs_embeds, pt_new_input_ids, trt_inputs_embeds, trt_new_input_ids


# ============================================================
# 维度 6: full_hidden_states 对比
# ============================================================

def compare_full_hidden_states(pytorch_model, trt_engine,
                                pt_inputs_embeds, pt_new_input_ids,
                                trt_inputs_embeds, trt_new_input_ids):
    """
    对比 LLM forward 输出的 full_hidden_states：
      - PyTorch: model(inputs_embeds=inputs_embeds) → outputs[0]
      - TRT: generate_test(inputs_embeds, new_input_ids) → full_hidden_states

    注意：PyTorch 和 TRT 使用不同的 input_ids 格式：
      - PyTorch: 视觉 token = -200 (IMAGE_TOKEN_INDEX)
      - TRT: 视觉 token >= 32001 (vocab_size)

    对比位置：
      1. waypoint_token(32000) 位置
      2. 文本 token 位置（所有文本 token 的均值）
      3. 视觉 token 位置（所有视觉 token 的均值）
    """
    print_section_header("维度 6: full_hidden_states 对比")

    # 检测 PyTorch model 所在设备，确保所有 tensor 在同一设备上
    pt_device = next(pytorch_model.parameters()).device
    gpu_device = torch.device('cuda:0')
    print(f"  PyTorch model device: {pt_device}")

    # ---- PyTorch forward ----
    print("  [PyTorch] Running forward...")
    if str(pt_device) == 'cpu':
        torch.cuda.empty_cache()
    t0 = time.time()

    # 构造 attention_mask 和 position_ids（与 inputs_embeds 同设备）
    seq_len = pt_inputs_embeds.shape[1]
    pt_inputs = pt_inputs_embeds.to(pt_device)
    pt_new_ids = pt_new_input_ids.to(pt_device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=pt_device)
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=pt_device).unsqueeze(0)

    with torch.no_grad():
        pt_outputs = pytorch_model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=pt_inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    pt_hidden_states = pt_outputs[0]  # (1, seq_len, 4096)
    t_pt = time.time() - t0
    print(f"    done in {t_pt:.2f}s, shape={tuple(pt_hidden_states.shape)}")

    # ---- TRT forward ----
    print("  [TRT] Running forward...")
    torch.cuda.empty_cache()
    t0 = time.time()
    # TRT engine 需要输入在 GPU 上，使用 TRT 格式的 inputs_embeds 和 new_input_ids
    trt_inputs = trt_inputs_embeds.to(gpu_device)
    trt_new_ids = trt_new_input_ids.to(gpu_device)
    # generate_test 内部调用 model.generate + 提取 full_hidden_states
    # 注意：generate_test 返回的是 single-position hidden states，
    # 但 debug_buffer 中有完整序列，我们在调用后直接读取
    trt_output_ids, _, trt_per_layer_hidden = trt_engine.generate_test(trt_inputs, trt_new_ids)
    # 直接从 debug_buffer 读取完整序列的 hidden states
    trt_full_hidden = trt_engine.model.session.debug_buffer["full_hidden_states"]
    # trt_full_hidden shape: (seq_len, 4096) — 2D，需要 reshape 为 (1, seq_len, 4096)
    if trt_full_hidden.dim() == 2:
        trt_full_hidden = trt_full_hidden.unsqueeze(0)  # (1, seq_len, 4096)
    t_trt = time.time() - t0
    print(f"    done in {t_trt:.2f}s, shape={tuple(trt_full_hidden.shape)}")

    # 对齐形状
    if trt_full_hidden.dim() == 2:
        # 2D → 3D
        B, D = pt_hidden_states.shape[0], pt_hidden_states.shape[-1]
        S = trt_full_hidden.shape[0] // B
        trt_full_hidden = trt_full_hidden.view(B, S, D)

    # 对齐逐层 hidden states 形状
    trt_layer_hs = {}
    for key, hs in trt_per_layer_hidden.items():
        if hs.dim() == 2:
            B, D = pt_hidden_states.shape[0], pt_hidden_states.shape[-1]
            S = hs.shape[0] // B
            hs = hs.view(B, S, D)
        trt_layer_hs[key] = hs

    # ---- 使用 PyTorch hooks 提取逐层 hidden states ----
    print("  [PyTorch] Extracting per-layer hidden states...")
    pt_layer_hs = {}  # layer_idx -> hidden_states

    def get_hook(layer_idx):
        def hook_fn(module, input, output):
            # output[0] is the hidden states (after layer computation)
            pt_layer_hs[layer_idx] = output[0].detach().cpu()
        return hook_fn

    hooks = []
    for i, layer in enumerate(pytorch_model.model.layers):
        hook = layer.register_forward_hook(get_hook(i))
        hooks.append(hook)

    # 重新运行 PyTorch forward 以捕获逐层输出
    pt_inputs = pt_inputs_embeds.to(pt_device)
    pt_new_ids = pt_new_input_ids.to(pt_device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=pt_device)
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=pt_device).unsqueeze(0)

    with torch.no_grad():
        pt_outputs = pytorch_model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=pt_inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    # Remove hooks
    for hook in hooks:
        hook.remove()
    print(f"    captured {len(pt_layer_hs)} layers")

    # ---- 对比 ----
    results = {}
    seq_len_pt = pt_hidden_states.shape[1]
    seq_len_trt = trt_full_hidden.shape[1]
    common_len = min(seq_len_pt, seq_len_trt)

    # 只对比公共长度
    pt_hs = pt_hidden_states[:, :common_len, :]
    trt_hs = trt_full_hidden[:, :common_len, :]

    # 全局对比
    r = compare_tensors('full_hidden_states (global)', pt_hs, trt_hs, prefix='[E] ')
    results['global'] = r

    # 找出 waypoint_token 位置（使用 PyTorch 格式的 input_ids 做 mask）
    waypoint_token_idx = 32000
    IMAGE_TOKEN_INDEX = -200
    new_ids = pt_new_input_ids[0]  # (seq_len,)
    wp_positions = (new_ids == waypoint_token_idx).nonzero(as_tuple=True)[0]

    if len(wp_positions) > 0:
        wp_pos = wp_positions[0].item()
        if wp_pos < common_len:
            pt_wp = pt_hs[:, wp_pos:wp_pos+1, :]
            trt_wp = trt_hs[:, wp_pos:wp_pos+1, :]
            r = compare_tensors(f'hidden_states @ waypoint_token(pos={wp_pos})',
                                pt_wp, trt_wp, prefix='[F] ')
            results['waypoint'] = r
        else:
            print(f"  [SKIP] waypoint position {wp_pos} exceeds common length {common_len}")
    else:
        print(f"  [SKIP] waypoint_token(32000) not found in input_ids")

    # 文本 token 位置对比（取均值）
    # PyTorch 格式：文本 token 是 != -200 且 != 32000 的正常 token
    new_ids_clamped = new_ids[:common_len].clone()
    text_mask = (new_ids_clamped != IMAGE_TOKEN_INDEX) & (new_ids_clamped != waypoint_token_idx) & (new_ids_clamped >= 0)
    text_mask = text_mask[:common_len]
    if text_mask.sum() > 0:
        pt_text = pt_hs[0, text_mask]
        trt_text = trt_hs[0, text_mask]
        r = compare_tensors('hidden_states @ text tokens (mean)',
                            pt_text.mean(dim=0, keepdim=True),
                            trt_text.mean(dim=0, keepdim=True),
                            prefix='[G] ')
        results['text_mean'] = r

    # 视觉 token 位置对比
    vis_mask = (new_ids_clamped == IMAGE_TOKEN_INDEX)
    vis_mask = vis_mask[:common_len]
    if vis_mask.sum() > 0:
        pt_vis = pt_hs[0, vis_mask]
        trt_vis = trt_hs[0, vis_mask]
        r = compare_tensors('hidden_states @ vision tokens (mean)',
                            pt_vis.mean(dim=0, keepdim=True),
                            trt_vis.mean(dim=0, keepdim=True),
                            prefix='[H] ')
        results['vision_mean'] = r

    # ================================================================
    # 新增: 逐位置误差分布分析
    # ================================================================
    print(f"\n  {'=' * 55}")
    print(f"  [POSITION-LEVEL] 逐位置误差分析")
    print(f"  {'=' * 55}")

    # 计算每个位置的 mean abs diff（确保在同一设备上）
    pt_hs_flat = pt_hs[0].float().cpu()
    trt_hs_flat = trt_hs[0].float().cpu()
    pos_diffs = (pt_hs_flat - trt_hs_flat).abs().mean(dim=1)  # (common_len,)
    pos_diffs_np = pos_diffs.cpu().numpy()

    # 全局统计
    print(f"  pos_diff: mean={pos_diffs_np.mean():.6e}, max={pos_diffs_np.max():.6e}, "
          f"min={pos_diffs_np.min():.6e}, std={pos_diffs_np.std():.6e}")

    # 按 token 类型统计误差
    text_mask_np = text_mask[:common_len].cpu().numpy() if text_mask.sum() > 0 else np.array([], dtype=bool)
    vis_mask_np = vis_mask[:common_len].cpu().numpy() if vis_mask.sum() > 0 else np.array([], dtype=bool)
    wp_mask_np = np.zeros(common_len, dtype=bool)
    if len(wp_positions) > 0:
        wp_pos = wp_positions[0].item()
        if wp_pos < common_len:
            wp_mask_np[wp_pos] = True

    if text_mask_np.sum() > 0:
        text_diffs = pos_diffs_np[text_mask_np]
        print(f"  [text tokens]   ({text_mask_np.sum():4d} tokens): "
              f"mean={text_diffs.mean():.6e}, max={text_diffs.max():.6e}")
    if vis_mask_np.sum() > 0:
        vis_diffs = pos_diffs_np[vis_mask_np]
        print(f"  [vision tokens] ({vis_mask_np.sum():4d} tokens): "
              f"mean={vis_diffs.mean():.6e}, max={vis_diffs.max():.6e}")
    if wp_mask_np.sum() > 0:
        wp_diff = pos_diffs_np[wp_mask_np][0]
        print(f"  [waypoint]      (1 token):         diff={wp_diff:.6e}")

    # 按段分析: 文本前缀 → 视觉 → 文本后缀(含waypoint)
    # 找到视觉段的起始和结束位置
    if vis_mask_np.sum() > 0:
        vis_start = np.where(vis_mask_np)[0][0]
        vis_end = np.where(vis_mask_np)[0][-1]

        # 文本前缀 (0 ~ vis_start-1)
        if vis_start > 0:
            prefix_diffs = pos_diffs_np[:vis_start]
            print(f"  [text prefix]   (pos 0~{vis_start-1:3d}, {len(prefix_diffs):3d} tokens): "
                  f"mean={prefix_diffs.mean():.6e}, max={prefix_diffs.max():.6e}")

        # 视觉段 (vis_start ~ vis_end)
        vis_region_diffs = pos_diffs_np[vis_start:vis_end+1]
        print(f"  [vision region] (pos {vis_start:3d}~{vis_end:3d}, {len(vis_region_diffs):3d} tokens): "
              f"mean={vis_region_diffs.mean():.6e}, max={vis_region_diffs.max():.6e}")

        # 文本后缀 (vis_end+1 ~ end)
        if vis_end + 1 < common_len:
            suffix_diffs = pos_diffs_np[vis_end+1:]
            print(f"  [text suffix]   (pos {vis_end+1:3d}~{common_len-1:3d}, {len(suffix_diffs):3d} tokens): "
                  f"mean={suffix_diffs.mean():.6e}, max={suffix_diffs.max():.6e}")

    # 找出 top-10 最差位置
    worst_positions = np.argsort(pos_diffs_np)[-10:][::-1]
    print(f"\n  Top-10 worst positions:")
    for rank, pos in enumerate(worst_positions):
        pos_id = new_ids[pos].item() if pos < len(new_ids) else -1
        # 判断 token 类型
        if pos_id == waypoint_token_idx:
            tok_type = "waypoint"
        elif pos_id == IMAGE_TOKEN_INDEX:
            tok_type = "vision"
        elif pos_id >= 0:
            tok_type = f"text(id={pos_id})"
        else:
            tok_type = "special"
        print(f"    #{rank+1}: pos={pos:4d}, diff={pos_diffs_np[pos]:.6e}, token_type={tok_type}")

    # 保存误差数据到文件（用于后续画图）
    save_dir = os.path.join(os.path.dirname(__file__), 'debug')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'pos_diffs.npy'), pos_diffs_np)
    np.save(os.path.join(save_dir, 'pt_hs.npy'), pt_hs[0].float().cpu().numpy())
    np.save(os.path.join(save_dir, 'trt_hs.npy'), trt_hs[0].float().cpu().numpy())
    np.save(os.path.join(save_dir, 'new_ids.npy'), new_ids[:common_len].cpu().numpy())
    print(f"  [SAVED] Error data to {save_dir}/")

    # ================================================================
    # 逐层 hidden states 对比
    # ================================================================
    print(f"\n  {'=' * 55}")
    print(f"  [PER-LAYER] 逐层 hidden states 对比")
    print(f"  {'=' * 55}")

    num_layers = len(pt_layer_hs)
    for layer_idx in range(num_layers):
        if layer_idx not in pt_layer_hs:
            continue
        pt_ly = pt_layer_hs[layer_idx]  # (1, seq_len, 4096)
        ly_key = f"hidden_states_layer_{layer_idx}"
        if ly_key not in trt_layer_hs:
            continue
        trt_ly = trt_layer_hs[ly_key]  # (1, seq_len, 4096)
        # 对齐长度
        common = min(pt_ly.shape[1], trt_ly.shape[1])
        pt_ly = pt_ly[:, :common, :]
        trt_ly = trt_ly[:, :common, :]

        # 计算误差
        pt_ly_f = pt_ly.float().cpu()
        trt_ly_f = trt_ly.float().cpu()
        max_diff = (pt_ly_f - trt_ly_f).abs().max().item()
        mean_diff = (pt_ly_f - trt_ly_f).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            pt_ly_f.flatten().unsqueeze(0),
            trt_ly_f.flatten().unsqueeze(0)
        ).item()

        # 标记首次显著误差
        if layer_idx == 0:
            first_bad = max_diff > 1e-2
        marker = " <<< FIRST BAD" if (layer_idx == 0 and first_bad) else ""
        print(f"  Layer {layer_idx:2d}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, cos_sim={cos_sim:.6f}{marker}")

    return results


# ============================================================
# PyTorch 模型加载
# ============================================================

def load_pytorch_model(hf_model_path, device='cuda:0'):
    """
    加载 PyTorch LlavaLlamaForCausalLM，只加载 LLM 部分（不含 vision tower）。

    策略：
      1. 注册自定义 LlavaConfig / LlavaLlamaForCausalLM
      2. 创建模型骨架（不加载权重）
      3. 从 HuggingFace safetensors 加载 LLM 权重（跳过 vision_tower 相关 key）
      4. 删除 vision_tower 相关模块，释放内存
      5. 将 LLM (model + lm_head) 移到 GPU
    """
    print_section_header("加载 PyTorch reference model")

    # 注册自定义模型类型
    from mmcv.utils.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register("llava_llama", LlavaConfig)
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

    hf_path = resolve_path(hf_model_path)
    if not os.path.exists(os.path.join(hf_path, 'config.json')):
        print(f"  [ERROR] HF checkpoint not found: {hf_path}")
        return None

    print(f"  Loading config from: {hf_path}")
    config = LlavaConfig.from_pretrained(hf_path)

    # 修改 config 以减小内存
    config.use_cache = False

    print(f"  Creating model skeleton (vocab_size={config.vocab_size})...")
    model = LlavaLlamaForCausalLM(config)
    model.eval()

    # 从 safetensors 加载权重
    from safetensors.torch import load_file as safe_load

    # 获取分片信息
    index_path = os.path.join(hf_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        weight_map = index_data.get('weight_map', {})

        # 收集所有需要加载的文件
        shard_files = set()
        llm_keys = []
        skipped_keys = []
        for key, shard_rel in weight_map.items():
            if 'vision_tower' in key or 'mm_projector' in key:
                skipped_keys.append(key)
            else:
                shard_files.add(shard_rel)
                llm_keys.append(key)

        print(f"  LLM keys: {len(llm_keys)}, skipped vision keys: {len(skipped_keys)}")
        print(f"  Loading {len(shard_files)} shard files...")

        # 逐文件加载
        llm_state_dict = {}
        for shard_rel in sorted(shard_files):
            shard_path = os.path.join(hf_path, shard_rel)
            if not os.path.exists(shard_path):
                print(f"    [WARN] Shard not found: {shard_path}")
                continue
            shard_sd = safe_load(shard_path)
            # 只保留 LLM 相关 key
            for k, v in shard_sd.items():
                if 'vision_tower' not in k and 'mm_projector' not in k:
                    llm_state_dict[k] = v
            del shard_sd
            print(f"    Loaded {shard_rel}: {len([k for k in llm_state_dict.keys() if k.startswith(shard_rel.split('.')[0])])} keys so far")

        # 加载到模型
        missing, unexpected = model.load_state_dict(llm_state_dict, strict=False)
        print(f"  Missing keys: {len(missing)} (expected: vision_tower, mm_projector)")
        print(f"  Unexpected keys: {len(unexpected)}")
        del llm_state_dict
    else:
        # 单文件 safetensors
        model_path = os.path.join(hf_path, 'model.safetensors')
        if os.path.exists(model_path):
            print(f"  Loading single safetensors file...")
            sd = safe_load(model_path)
            llm_sd = {k: v for k, v in sd.items()
                      if 'vision_tower' not in k and 'mm_projector' not in k}
            missing, unexpected = model.load_state_dict(llm_sd, strict=False)
            print(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
            del sd, llm_sd
        else:
            print(f"  [ERROR] No safetensors found in {hf_path}")
            return None

    # 删除 vision_tower 相关模块（如果存在），释放内存
    if hasattr(model.model, 'vision_tower'):
        del model.model.vision_tower
        print("  Deleted vision_tower module")
    if hasattr(model.model, 'mm_projector'):
        del model.model.mm_projector
        print("  Deleted mm_projector module")

    # 移到指定设备（fp16）
    # 注意：如果 device 是 'cpu'，半精度在 CPU 上可能不支持某些操作，
    # 但 Llama 7B 在 CPU 上做单帧推理是可行的（慢但可用）。
    print(f"  Moving LLM to {device} (fp16)...")
    if device == 'cpu':
        # CPU 推理：用 fp32 避免 CPU 不支持 fp16 的问题
        model = model.float()
        model = model.to(device)
        print(f"  [CPU mode] Using fp32 on CPU (single frame inference may take ~30s)")
    else:
        model = model.half().to(device)
    model.eval()

    # 确保 embed_tokens 也在正确设备上
    model.model.embed_tokens = model.model.embed_tokens.to(device)

    # 打印参数量
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {param_count / 1e9:.2f}B params on {device}")

    return model


# ============================================================
# TRT engine 加载
# ============================================================

def load_trt_engines(llm_engine_dir, vision_engine_path, tokenizer_path):
    """
    加载 TRT Vision engine 和 LLM engine。

    返回 (vision_engine, llm_engine)。
    vision_engine 是 InferTrt 实例（未初始化 binding），
    llm_engine 是 InferTrtLLM 实例。
    """
    print_section_header("加载 TRT engines")

    # 导入
    import tensorrt as trt
    import pycuda.driver as cuda
    from adzoo.orion.test_trt import InferTrtLLM, InferTrt

    # 解析路径
    llm_dir = resolve_path(llm_engine_dir)
    vis_path = resolve_path(vision_engine_path)
    tok_path = resolve_path(tokenizer_path)

    # 检查文件
    missing = []
    if not os.path.isdir(llm_dir) or not os.path.exists(os.path.join(llm_dir, 'rank0.engine')):
        missing.append(f"LLM engine: {llm_dir}")
    if not os.path.exists(vis_path):
        missing.append(f"Vision engine: {vis_path}")
    if not os.path.exists(os.path.join(tok_path, 'config.json')):
        missing.append(f"Tokenizer: {tok_path}")

    if missing:
        print(f"  [ERROR] Missing files: {missing}")
        return None, None

    # ---- 加载 LLM engine ----
    print(f"  Loading LLM engine from: {llm_dir}")
    llm_engine = InferTrtLLM(llm_engine_pth=llm_dir, tokenizer_pth=tok_path)
    print(f"    LLM engine loaded. vocab_size={llm_engine.model_config.vocab_size}")

    # ---- 加载 Vision engine ----
    print(f"  Loading Vision engine from: {vis_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    # 不初始化全部 binding，只加载 engine 文件
    # 注意：InferTrt.__init__ 会调用 load_ego_engine，传入 None 即可
    vision_engine = InferTrt(logger, qa_save_path=None, LLM_engine=llm_engine,
                             torch_ref_model=None, ego_engine_pth=None)
    vision_engine.read(vis_path)
    vision_engine.eval()  # 初始化 binding
    print(f"    Vision engine loaded. num_bindings={len(vision_engine.bindings)}")

    return vision_engine, llm_engine


# ============================================================
# 数据加载：从 dataloader 获取单帧样本
# ============================================================

def load_single_sample(config_path, data_root, ann_file, map_root, map_file):
    """
    从 validation dataloader 加载一个样本。

    返回 (img_metas, input_ids, img, lidar2img, cam_intrinsic, timestamp,
           ego_pose, ego_pose_inv, command, can_bus, ...)
    """
    from mmcv import Config
    from mmcv.datasets import build_dataset, build_dataloader

    print_section_header("加载样本")

    cfg_path = resolve_path(config_path)
    cfg = Config.fromfile(cfg_path)

    # 覆盖数据路径
    cfg.data.test.data_root = resolve_path(data_root)
    cfg.data.test.ann_file = resolve_path(ann_file)
    cfg.data.test.map_root = resolve_path(map_root)
    cfg.data.test.map_file = resolve_path(map_file)
    cfg.data.test.test_mode = True

    # 设置 workers_per_gpu=0 避免 fork 问题
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    # 取第一个样本
    for batch in data_loader:
        break

    input_ids_raw = batch['input_ids'][0]
    if isinstance(input_ids_raw, list):
        print(f"  Sample loaded. input_ids is a list, len={len(input_ids_raw)}")
        # 递归检查深层结构
        for depth in range(3):
            if isinstance(input_ids_raw, list) and len(input_ids_raw) > 0:
                input_ids_raw = input_ids_raw[0]
        if hasattr(input_ids_raw, 'shape'):
            print(f"    deepest tensor shape: {tuple(input_ids_raw.shape)}")
    else:
        print(f"  Sample loaded. input_ids shape: {tuple(input_ids_raw.shape)}")
    return batch


# ============================================================
# Vision engine forward：获取 vision_embeded
# ============================================================

def run_vision_engine(vision_engine, batch):
    """
    运行 Vision engine 获取 vision_embeded。

    注意：Vision engine 的 forward 会执行完整的 3D 检测 + 地图 + LLM 推理。
    我们只需要 vision_embeded 输出。
    这里通过直接调用 engine forward 并提取 bindings 中的 vision_embeded。
    """
    print_section_header("Vision engine forward")

    # 整理数据（与 InferTrt.forward 逻辑一致）
    img_metas = batch['img_metas'][0]
    # input_ids 在 batch 中是 [1, 1, 1, seq_len] 格式
    input_ids = batch['input_ids'][0][0][0].unsqueeze(0)

    img = batch['img'][0].to(device=f"cuda:0", dtype=torch.float32).contiguous()
    lidar2img = batch['lidar2img'][0].to(device=f"cuda:0", dtype=torch.float32).inverse()
    cam_intrinsic = batch['cam_intrinsic'][0].to(device=f"cuda:0", dtype=torch.float32)
    timestamp = batch['timestamp'][0]
    ego_pose = batch['ego_pose'][0].to(device=f"cuda:0", dtype=torch.float32).contiguous()
    ego_pose_inv = batch['ego_pose_inv'][0].to(device=f"cuda:0", dtype=torch.float32).contiguous()
    command = batch['command'][0].to(device=f"cuda:0", dtype=torch.float32).contiguous()
    can_bus = batch['can_bus'][0].to(device=f"cuda:0", dtype=torch.float32).contiguous()
    ego_fut_cmd = batch['ego_fut_cmd'][0].to(device=f"cuda:0", dtype=torch.float32).contiguous()
    gt_attr_labels = batch['gt_attr_labels'][0][0]
    gt_bboxes_3d = batch['gt_bboxes_3d'][0][0]
    gt_labels_3d = batch['gt_labels_3d'][0][0]
    ego_fut_trajs = batch['ego_fut_trajs'][0].to(device=f"cuda:0", dtype=torch.float32).contiguous()
    fut_valid_flag = bool(batch['fut_valid_flag'][0])

    # 处理 timestamp
    # 简化：用当前时间戳
    timestamp = timestamp.type(torch.float32).to(device=f"cuda:0", dtype=torch.float32).contiguous()

    # 处理 is_first_frame
    is_first_frame = torch.ones([1]).to(device=f"cuda:0", dtype=torch.float32).contiguous()

    # ---- 复制输入到 binding ----
    vision_engine.bindings["img"].copy_(img)
    vision_engine.bindings["img2lidars"].copy_(lidar2img.to(device=f"cuda:0", dtype=torch.float32).contiguous())
    vision_engine.bindings["command"].copy_(command)
    vision_engine.bindings["can_bus"].copy_(can_bus)
    vision_engine.bindings["is_first_frame"].copy_(is_first_frame)
    vision_engine.bindings["ego_pose"].copy_(ego_pose)
    vision_engine.bindings["ego_pose_inv"].copy_(ego_pose_inv)
    vision_engine.bindings["timestamp"].copy_(timestamp)

    # 复制 memory 循环 buffer
    for key in ['memory_embedding_bbox', 'memory_reference_point_bbox',
                'memory_timestamp_bbox', 'memory_egopose_bbox', 'memory_canbus_bbox',
                'sample_time_bbox', 'memory_scene_query_bbox', 'scene_memory_timestamp_bbox',
                'memory_timestamp_map', 'sample_time_map', 'memory_egopose_map',
                'memory_embedding_map', 'memory_reference_point_map']:
        in_key = f"{key}_in"
        out_key = f"{key}_out"
        if in_key in vision_engine.bindings and out_key in vision_engine.bindings:
            vision_engine.bindings[in_key].copy_(vision_engine.bindings[out_key])

    # ---- 执行 inference ----
    vision_engine.cuda_ctx.push()
    for i in range(len(vision_engine.names)):
        vision_engine.context.set_tensor_address(
            vision_engine.names[i],
            vision_engine.bindings[str(vision_engine.names[i])].data_ptr()
        )
    vision_engine.context.execute_async_v3(stream_handle=vision_engine.stream.handle)
    vision_engine.stream.synchronize()
    vision_engine.cuda_ctx.pop()

    # ---- 提取 vision_embeded ----
    vision_embeded = vision_engine.bindings["vision_embeded"].clone()
    print(f"  vision_embeded: {tuple(vision_embeded.shape)}, {vision_embeded.dtype}")

    return input_ids, vision_embeded


# ============================================================
# 主流程
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='对比 TRT engine 与 PyTorch 原版的 hidden_states',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode', choices=['auto', 'full', 'embed'], default='auto',
                        help='运行模式: auto=自动选择, full=完整对比, embed=仅权重对比')
    parser.add_argument('--llm-engine', default=DEFAULT_PATHS['llm_engine_dir'],
                        help='LLM engine 目录 (相对 PROJECT_ROOT)')
    parser.add_argument('--vision-engine', default=DEFAULT_PATHS['vision_engine_path'],
                        help='Vision engine 路径 (相对 PROJECT_ROOT)')
    parser.add_argument('--tokenizer', default=DEFAULT_PATHS['tokenizer_path'],
                        help='Tokenizer 路径 (相对 PROJECT_ROOT)')
    parser.add_argument('--hf-model', default=DEFAULT_PATHS['hf_model_path'],
                        help='HuggingFace 格式模型路径 (相对 PROJECT_ROOT)')
    parser.add_argument('--embed-tokens', default=DEFAULT_PATHS['embed_tokens_path'],
                        help='embed_tokens 文件路径 (相对 PROJECT_ROOT)')
    parser.add_argument('--config', default=DEFAULT_PATHS['config_path'],
                        help='MMCV config 路径 (相对 PROJECT_ROOT)')
    parser.add_argument('--checkpoint', default=DEFAULT_PATHS['checkpoint_path'],
                        help='Orion.pth 路径 (相对 PROJECT_ROOT)')
    parser.add_argument('--data-root', default=DEFAULT_PATHS['data_root'])
    parser.add_argument('--ann-file', default=DEFAULT_PATHS['ann_file'])
    parser.add_argument('--map-file', default=DEFAULT_PATHS['map_file'])
    parser.add_argument('--map-root', default=DEFAULT_PATHS['map_root'])
    parser.add_argument('--device', default='cuda:0', help='GPU 设备')
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查 GPU
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required but not available.")
        sys.exit(1)

    gpu_free = get_gpu_free_memory(0)
    print(f"GPU free memory: {gpu_free:.1f} GB")

    # 自动选择模式
    mode = args.mode
    if mode == 'auto':
        # full 模式：PyTorch LLM 7B 在 GPU 上需要 ~14GB + TRT engine 占用，
        # 总面积 ~20GB+。GPU 只有 16GB，所以 full 模式将 PyTorch 模型放在 CPU 上。
        # 需要额外 ~30GB CPU RAM 加载模型 + 单帧推理约 30s。
        if gpu_free >= 8:
            mode = 'full_cpu_pt'
            print(f"  → auto-selected: full_cpu_pt mode (PyTorch on CPU, TRT on GPU)")
        else:
            print(f"[ERROR] Insufficient GPU memory: {gpu_free:.1f}GB (< 8GB).")
            print(f"  Free up GPU memory or use a different device.")
            sys.exit(1)

    # ============================================================
    # 维度 2: embed_tokens 权重对比（始终执行）
    # ============================================================
    print("\n" + "=" * 70)
    print("  开始对比")
    print("=" * 70)

    embed_results = compare_embed_tokens(args.embed_tokens, args.llm_engine)

    # ============================================================
    # 完整模式：加载 TRT engines + 样本 + 对比 hidden_states
    # ============================================================
    if mode == 'full' or mode == 'full_cpu_pt':
        # ---- 加载 TRT engines ----
        vision_engine, llm_engine = load_trt_engines(
            args.llm_engine, args.vision_engine, args.tokenizer
        )
        if vision_engine is None or llm_engine is None:
            print("[ERROR] Failed to load TRT engines. Falling back to embed mode.")
            mode = 'embed'
        else:
            # ---- 加载 PyTorch model（full_cpu_pt 模式用 CPU）----
            pt_device = 'cpu' if mode == 'full_cpu_pt' else args.device
            pytorch_model = load_pytorch_model(args.hf_model, pt_device)
            if pytorch_model is None:
                print("[ERROR] Failed to load PyTorch model. Falling back to embed mode.")
                mode = 'embed'
            else:
                # ---- 加载样本 ----
                batch = load_single_sample(
                    args.config, args.data_root, args.ann_file,
                    args.map_root, args.map_file
                )

                # ---- 运行 Vision engine 获取 vision_embeded ----
                input_ids, vision_embeded = run_vision_engine(vision_engine, batch)

                # ---- 维度 1: embedding 注入对比 ----
                embed_result = compare_embedding_injection(
                    pytorch_model, llm_engine, input_ids, vision_embeded
                )

                if embed_result is not None:
                    pt_inputs_embeds, pt_new_ids, trt_inputs_embeds, trt_new_ids = embed_result

                    # ---- 维度 6: full_hidden_states 对比 ----
                    # PyTorch 路径用 pt_inputs_embeds/pt_new_ids（-200 格式）
                    # TRT 路径用 trt_inputs_embeds/trt_new_ids（>= 32001 格式）
                    hs_results = compare_full_hidden_states(
                        pytorch_model, llm_engine,
                        pt_inputs_embeds, pt_new_ids,
                        trt_inputs_embeds, trt_new_ids,
                    )

                # ---- 清理 ----
                del vision_engine, llm_engine, pytorch_model, batch
                torch.cuda.empty_cache()

    elif mode == 'embed':
        # ---- 仅维度 2: embed_tokens 对比 ----
        # 额外对比：从 Orion.pth 提取 embed_tokens 并与 patched safetensors 对比
        orion_path = resolve_path(args.checkpoint)
        if os.path.exists(orion_path):
            print_section_header("维度 2 (补充): Orion.pth embed_tokens 验证")
            EXPECTED_KEY = 'lm_head.base_model.model.model.embed_tokens.weight'
            print(f"  Loading Orion.pth (may take a few minutes)...")
            sd = torch.load(orion_path, map_location='cpu', weights_only=False)
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            if EXPECTED_KEY in sd:
                emb_orion = sd[EXPECTED_KEY]
                print(f"  {EXPECTED_KEY}: {tuple(emb_orion.shape)}, {emb_orion.dtype}")
                print(f"    WP(32000)[:5] = {emb_orion[32000, :5].float().tolist()}")
                print(f"    mean={emb_orion.float().mean().item():.6f}, std={emb_orion.float().std().item():.6f}")

                # 对比 embed_tokens_orion_fp16.pt
                embed_pt_path = resolve_path(args.embed_tokens)
                if os.path.exists(embed_pt_path):
                    emb_pt = torch.load(embed_pt_path, map_location='cpu')
                    compare_tensors('Orion.pth vs extract_embed_tokens',
                                    emb_orion, emb_pt)
            else:
                print(f"  [WARN] Key '{EXPECTED_KEY}' not found in Orion.pth")
                available = [k for k in sd.keys() if 'embed_tokens' in k][:5]
                print(f"    Available embed_tokens keys: {available}")
            del sd

    print(f"\n{'=' * 70}")
    print(f"  对比完成 (mode={mode})")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()