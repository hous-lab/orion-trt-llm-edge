"""
Export Orion-Lite Vision Module to ONNX for TensorRT deployment.

This script exports the vision pipeline (img_backbone + position_encoder +
pts_bbox_head + map_head) to ONNX, WITHOUT the LLM/student_model parts.

The output `vision_embeded` tensor (B, 513, 4096) is the direct input to the
student model TensorRT engine.

Mixed-precision strategy (Plan A):
  - img_backbone Linear/Conv layers  -> FP16 (safe, ~80% compute)
  - img_backbone LayerNorm/Softmax   -> FP32 (eps=1e-6, precision critical)
  - pts_bbox_head / map_head         -> FP32 (inverse_sigmoid, LayerNorm)
  - position_encoder                 -> FP32 (inverse_sigmoid)

Usage:
    cd $ORION_LITE_PROJECT_ROOT
    python scripts/export_vision_onnx.py \
        adzoo/orion/configs/orion_distill_new_agent_with_mimic.py \
        eval/fused_ckpts/orion_lite_subset8.pth \
        eval/fused_ckpts/vision_onnx/
"""

import argparse
import numpy as np
import os
import sys
import types
import warnings

import torch
from torch import nn
import torch.nn.functional as F

_training_repo = os.environ.get("ORION_LITE_PROJECT_ROOT")
if not _training_repo:
    raise SystemExit("error: ORION_LITE_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (Lite variant, see ckpts/README.md)")
sys.path.insert(0, _training_repo)

_numba = types.ModuleType('numba')
_numba.jit = lambda *a, **kw: (a[0] if a and callable(a[0]) else lambda f: f)
_numba.njit = _numba.jit
_numba.prange = range
_numba.float32 = 'float32'
_numba.float64 = 'float64'
_numba.int32 = 'int32'
_numba.int64 = 'int64'
_numba_np = types.ModuleType('numba.np')
_numba_np_ufunc = types.ModuleType('numba.np.ufunc')
_numba_np_ufunc_internal = types.ModuleType('numba.np.ufunc._internal')
_numba_np_ufunc.vectorize = lambda *a, **kw: lambda f: f
_numba_np_ufunc.guvectorize = lambda *a, **kw: lambda f: f
sys.modules['numba'] = _numba
sys.modules['numba.np'] = _numba_np
sys.modules['numba.np.ufunc'] = _numba_np_ufunc
sys.modules['numba.np.ufunc._internal'] = _numba_np_ufunc_internal

# flash_attn is hard-imported at module level by the training repo's vendored mmcv
# fork, but it is a compiled wheel pinned to a torch ABI (a torch upgrade routinely
# breaks it). The export below forces flash_attn=False on every attention consumer
# (the cfg.model.*["flash_attn"] = False overrides), so the CUDA functions are never
# called during the trace. If flash_attn is not importable, install a raise-if-called
# stub instead of failing; same pattern as the numba stub above.
try:
    import flash_attn  # noqa: F401
    from flash_attn.bert_padding import index_first_axis  # noqa: F401
except Exception:
    def _flash_stub(*_a, **_kw):
        raise RuntimeError(
            'flash_attn was not importable, so this script stubbed it; the export '
            'forces flash_attn=False and should never call it. This call means the '
            'flash_attn=False overrides below do not cover your model config.')
    import importlib.util

    def _mk(name):
        spec = importlib.util.spec_from_loader(name, None)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        return mod

    _fa = _mk('flash_attn')
    _fi = _mk('flash_attn.flash_attn_interface')
    _fi.flash_attn_varlen_kvpacked_func = _flash_stub
    _fi.flash_attn_unpadded_kvpacked_func = _flash_stub
    _bp = _mk('flash_attn.bert_padding')
    _bp.unpad_input = _flash_stub
    _bp.pad_input = _flash_stub
    _bp.index_first_axis = _flash_stub
    print('note: flash_attn not importable - stubbed (export runs flash_attn=False)')

import mmcv
from mmcv.utils import Config, DictAction, load_checkpoint, set_random_seed
from mmcv.models import build_model, fuse_conv_bn
from mmcv.models.utils.transformer import inverse_sigmoid
from mmcv.models.utils.positional_encoding import pos2posemb1d, nerf_positional_encoding
from mmcv.utils.misc import MLN, topk_gather, transform_reference_points_lane, transform_reference_points, memory_refresh

from torch.onnx import register_custom_op_symbolic


def _patch_mha_for_onnx():
    """
    Monkey-patch nn.MultiheadAttention.forward to avoid SDPA path
    which is not ONNX-exportable in PyTorch 2.0.
    Forces the traditional Q/K/V matmul path.
    """
    _orig_forward = nn.MultiheadAttention.forward

    def _onnx_friendly_forward(self, query, key, value, key_padding_mask=None,
                                need_weights=True, attn_mask=None,
                                average_attn_weights=True, is_causal=False):
        if self.batch_first and query.dim() == 3:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        head_dim = embed_dim // self.num_heads

        if self._qkv_same_embed_dim:
            q_proj = F.linear(query, self.in_proj_weight[:embed_dim], self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
            k_proj = F.linear(key, self.in_proj_weight[embed_dim:2*embed_dim], self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None)
            v_proj = F.linear(value, self.in_proj_weight[2*embed_dim:], self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None)
        else:
            q_proj = F.linear(query, self.q_proj_weight, self.q_proj_bias)
            k_proj = F.linear(key, self.k_proj_weight, self.k_proj_bias)
            v_proj = F.linear(value, self.v_proj_weight, self.v_proj_bias)

        q_proj = q_proj.reshape(tgt_len, bsz, self.num_heads, head_dim).permute(1, 2, 0, 3)
        k_proj = k_proj.reshape(src_len, bsz, self.num_heads, head_dim).permute(1, 2, 0, 3)
        v_proj = v_proj.reshape(src_len, bsz, self.num_heads, head_dim).permute(1, 2, 0, 3)

        scale = head_dim ** -0.5
        q_proj = q_proj * scale
        attn_weights = torch.matmul(q_proj, k_proj.transpose(-2, -1))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.dropout(attn_weights, self.dropout, train=False)
        attn_output = torch.matmul(attn_weights, v_proj)

        attn_output = attn_output.permute(1, 2, 0, 3).reshape(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        return attn_output, None

    nn.MultiheadAttention.forward = _onnx_friendly_forward
    print("Patched nn.MultiheadAttention.forward for ONNX export (SDPA disabled)")


_patch_mha_for_onnx()


def unflatten_symbolic(g, self, dim, sizes):
    return g.op('Reshape', self, sizes)


register_custom_op_symbolic("aten::unflatten", unflatten_symbolic, 17)


def parse_args():
    parser = argparse.ArgumentParser(description='Orion-Lite Vision Module ONNX Export')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (square). Default 640. Use 512 for reduced resolution.')
    parser.add_argument('onnx_save_path', help='directory to save ONNX files')
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--save-test-data', action='store_true', default=True,
                        help='Save numpy test data for accuracy verification')
    args = parser.parse_args()
    return args


class OrionLiteVisionProxy(nn.Module):
    """
    Proxy module that wraps OrionDistilledNew and exposes only the vision
    pipeline for ONNX export. No LLM, no student_model.

    Forward flow:
        img -> img_backbone -> position_encoder -> pts_bbox_head -> map_head
                                                                    |
        Output: vision_embeded (B, 513, 4096)  +  detection/map results
                + memory states for temporal tracking
    """

    def __init__(self, mod, img_size=640):
        super().__init__()
        self.mod = mod
        self.img_size = img_size
        self.grid_size = img_size // 16  # patch_size=16

    def locations(self, stride, pad_h, pad_w, device=None):
        h, w = self.grid_size, self.grid_size
        if device is None:
            device = "cpu"
        shifts_x = (torch.arange(
            0, stride * w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        locations = locations.reshape(h, w, 2)
        return locations

    def pos_embedding(self, cam_intrinsic, img2lidars):
        pad_h, pad_w, _ = (self.img_size, self.img_size, 3)
        memory_centers = self.locations(16, pad_h, pad_w, device=cam_intrinsic.device)[None].repeat(1 * 6, 1, 1, 1)

        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = cam_intrinsic.size(0)

        cam_intrinsic = torch.stack([cam_intrinsic[..., 0, 0], cam_intrinsic[..., 1, 1]], dim=-1)
        cam_intrinsic = torch.abs(cam_intrinsic) / 1e3
        cam_intrinsic = cam_intrinsic.repeat(1, H * W, 1).view(B, -1, 2)
        LEN = cam_intrinsic.size(1)

        num_sample_tokens = LEN
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.mod.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.mod.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1, 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)

        coords = coords.unsqueeze(-1)

        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H * W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.mod.position_range[0:3]) / (self.mod.position_range[3:6] - self.mod.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D * 3)

        pos_embed = inverse_sigmoid(coords3d)
        return pos_embed

    def bboxOrionHead_forward(self, img_feats, pos_embed, command, can_bus, is_first_frame,
                               memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_canbus,
                               sample_time,
                               memory_scene_query, scene_memory_timestamp,
                               ego_pose, timestamp, ego_pose_inv):
        head = self.mod.pts_bbox_head

        B, N, C, H, W = img_feats.shape
        memory_timestamp += timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time += timestamp
        sample_time_x = (torch.abs(sample_time) < 2.0).to(img_feats.dtype)
        sample_time_x *= (1. - is_first_frame)

        scene_memory_timestamp += timestamp.unsqueeze(-1).unsqueeze(-1)
        scene_memory_timestamp = memory_refresh(scene_memory_timestamp[:, :head.scence_memory_len], sample_time_x)
        memory_scene_query = memory_refresh(memory_scene_query[:, :head.scence_memory_len], sample_time_x)

        memory_egopose = ego_pose_inv.unsqueeze(1) @ memory_egopose
        memory_reference_point = transform_reference_points(memory_reference_point, ego_pose_inv, reverse=False)
        memory_timestamp = memory_refresh(memory_timestamp[:, :head.memory_len], sample_time_x)
        memory_timestamp *= (1. - is_first_frame)
        memory_reference_point = memory_refresh(memory_reference_point[:, :head.memory_len], sample_time_x)
        memory_reference_point *= (1. - is_first_frame)
        memory_embedding = memory_refresh(memory_embedding[:, :head.memory_len], sample_time_x)
        memory_embedding *= (1. - is_first_frame)
        memory_egopose = memory_refresh(memory_egopose[:, :head.memory_len], sample_time_x)
        memory_egopose *= (1. - is_first_frame)
        memory_canbus = memory_refresh(memory_canbus[:, :head.can_bus_len], sample_time_x)
        memory_canbus *= (1. - is_first_frame)
        sample_time = timestamp.new_zeros(B)
        sample_time *= (1. - is_first_frame)

        pseudo_reference_points = head.pseudo_reference_points.weight * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3]
        memory_reference_point[:, :head.num_propagated] = memory_reference_point[:, :head.num_propagated] + (1 - sample_time_x).view(B, 1, 1) * pseudo_reference_points
        memory_egopose[:, :head.num_propagated] = memory_egopose[:, :head.num_propagated] + (1 - sample_time_x).view(B, 1, 1, 1) * torch.eye(4, device=sample_time_x.device)

        num_tokens = N * H * W
        memory = img_feats.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = head.input_projection(memory)

        reference_points = head.reference_points.weight
        reference_points = torch.cat([torch.zeros_like(reference_points[:head.num_extra]), reference_points], dim=0)
        attn_mask = None
        tgt_size = head.num_query + head.memory_len + head.num_extra
        query_size = head.num_query + head.num_propagated + head.num_extra
        attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
        attn_mask[head.num_extra:, :head.num_extra] = True

        reference_points = reference_points.unsqueeze(0).repeat(B, 1, 1)

        query_pos = head.query_pos(nerf_positional_encoding(reference_points.repeat(1, 1, head.n_control)))
        tgt = torch.zeros_like(query_pos)

        temp_reference_point = (memory_reference_point - head.pc_range[:3]) / (head.pc_range[3:6] - head.pc_range[0:3])
        temp_pos = head.query_pos(nerf_positional_encoding(temp_reference_point.repeat(1, 1, head.n_control)))
        temp_memory = memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)

        rec_ego_motion = torch.cat([torch.zeros_like(reference_points[..., :1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
        rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
        memory_ego_motion = torch.cat([memory_timestamp, memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
        memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
        temp_pos = head.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += head.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[..., :1])))
        temp_pos += head.time_embedding(pos2posemb1d(memory_timestamp).float())

        tgt = torch.cat([tgt, temp_memory[:, :head.num_propagated]], dim=1)
        query_pos = torch.cat([query_pos, temp_pos[:, :head.num_propagated]], dim=1)
        reference_points = torch.cat([reference_points, temp_reference_point[:, :head.num_propagated]], dim=1)
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1] + head.num_propagated, 1, 1)
        temp_memory = temp_memory[:, head.num_propagated:]
        temp_pos = temp_pos[:, head.num_propagated:]

        tgt[:, :head.num_extra, :] = head.query_embedding.weight.unsqueeze(0)
        query_pos[:, :head.num_extra, :] = query_pos[:, :head.num_extra, :] * 0

        outs_dec = head.transformer(tgt, memory, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)
        reference_points = reference_points[:, head.num_extra:, :]
        outs_dec = torch.nan_to_num(outs_dec)
        vlm_memory = outs_dec[-1, :, :head.num_extra, :]
        outs_dec = outs_dec[:, :, head.num_extra:, :]

        current_query = head.memory_query.weight.unsqueeze(0).repeat(B, 1, 1)
        time_embeding = head.scene_time_embedding(pos2posemb1d(scene_memory_timestamp).float())
        cur_time_embeding = torch.zeros_like(current_query)
        temp_query_embedding = head.memory_decoder_mq(
            query=current_query,
            key=memory_scene_query,
            query_pos=cur_time_embeding,
            key_pos=time_embeding,
        )
        history_query = head.memory_decoder_cq(
            query=temp_query_embedding,
            key=vlm_memory,
            query_pos=None,
            key_pos=None,
        )

        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []
        outputs_trajs_classes = []
        outputs_traffic_states = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            outputs_class = head.cls_branches[lvl](outs_dec[lvl])
            outputs_traffic_state = head.tl_branches[lvl](outs_dec[lvl])
            outputs_traffic_states.append(outputs_traffic_state)
            tmp = head.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_traffic_states = torch.stack(outputs_traffic_states)
        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3])

        rec_can_bus = can_bus.clone()
        rec_can_bus[:, :3] = 0
        rec_can_bus[:, -1] = 0
        rec_can_bus = torch.cat([command.unsqueeze(-1), rec_can_bus], dim=-1)
        memory_ego_pose = memory_egopose.reshape(B, -1, head.topk_proposals, 4, 4).flatten(-2)

        vlm_memory = torch.cat((vlm_memory, history_query), dim=1)
        vlm_memory = head.output_projection(vlm_memory)
        can_bus_input = torch.cat([rec_can_bus, memory_canbus.flatten(-2), memory_ego_pose.mean(-2).flatten(-2)], dim=-1)
        can_bus_embed = head.can_bus_embed(can_bus_input)
        vlm_memory = torch.cat([vlm_memory, can_bus_embed.unsqueeze(-2)], dim=-2)

        motion_query = outs_dec[-1].permute(1, 0, 2)
        mode_query = head.motion_mode_query.weight
        motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
        motion_query = motion_query.permute(1, 0, 2)
        motion_hs = head.motion_decoder(query=motion_query, key=motion_query,
                                        query_pos=None, key_pos=None, attn_mask=None)
        ca_motion_query = motion_hs.flatten(0, 1).unsqueeze(0)
        num_agent = outs_dec[-1].shape[1]
        motion_hs = motion_hs.reshape(B, num_agent, head.fut_mode, -1)
        ca_motion_query = ca_motion_query.squeeze(0).reshape(
            B, num_agent, head.fut_mode, -1)
        motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)

        outputs_traj = head.traj_branches[0](motion_hs)
        outputs_trajs.append(outputs_traj)
        outputs_traj_class = head.traj_cls_branches[0](motion_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
        outputs_trajs = torch.stack(outputs_trajs)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        all_traj_preds = outputs_trajs.repeat(all_bbox_preds.shape[0], 1, 1, 1, 1)
        all_traj_cls_scores = outputs_trajs_classes.repeat(all_bbox_preds.shape[0], 1, 1, 1)

        rec_reference_points = all_bbox_preds[..., :3][-1]
        rec_velo = all_bbox_preds[..., -2:][-1]
        out_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        _, topk_indexes = torch.topk(rec_score, head.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        memory_embedding = torch.cat([rec_memory, memory_embedding], dim=1)
        memory_timestamp = torch.cat([rec_timestamp, memory_timestamp], dim=1)
        memory_egopose = torch.cat([rec_ego_pose, memory_egopose], dim=1)
        memory_reference_point = torch.cat([rec_reference_points, memory_reference_point], dim=1)
        memory_canbus = torch.cat([rec_can_bus.unsqueeze(-2), memory_canbus], dim=1)
        memory_reference_point = transform_reference_points(memory_reference_point, ego_pose, reverse=False)
        memory_timestamp -= timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time -= timestamp
        memory_egopose = ego_pose.unsqueeze(1) @ memory_egopose

        scene_memory_timestamp = torch.cat([torch.zeros_like(scene_memory_timestamp[:, :head.num_memory, :]), scene_memory_timestamp], dim=1)
        scene_memory_timestamp -= timestamp.unsqueeze(-1).unsqueeze(-1)
        memory_scene_query = torch.cat([history_query.detach(), memory_scene_query], dim=1)

        return all_cls_scores, all_bbox_preds, \
            all_traffic_states, \
            all_traj_preds, all_traj_cls_scores, \
            vlm_memory, \
            memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_canbus, sample_time, \
            memory_scene_query, scene_memory_timestamp

    def mapOrionHeadM_forward(self, img_feats, pos_embed, timestamp, ego_pose,
                               ego_pose_inv, is_first_frame, sample_time,
                               memory_timestamp, memory_egopose,
                               memory_embedding, memory_reference_point):
        head = self.mod.map_head

        B = img_feats.size(0)
        memory_timestamp += timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time += timestamp
        sample_time_x = (torch.abs(sample_time) < 2.0).to(img_feats.dtype)
        sample_time_x *= (1. - is_first_frame)
        memory_egopose = ego_pose_inv.unsqueeze(1) @ memory_egopose
        memory_reference_point = transform_reference_points_lane(memory_reference_point, ego_pose_inv, reverse=False)
        memory_timestamp = memory_refresh(memory_timestamp[:, :head.memory_len], sample_time_x)
        memory_timestamp *= (1. - is_first_frame)
        memory_reference_point = memory_refresh(memory_reference_point[:, :head.memory_len], sample_time_x)
        memory_reference_point *= (1. - is_first_frame)
        memory_embedding = memory_refresh(memory_embedding[:, :head.memory_len], sample_time_x)
        memory_embedding *= (1. - is_first_frame)
        memory_egopose = memory_refresh(memory_egopose[:, :head.memory_len], sample_time_x)
        memory_egopose *= (1. - is_first_frame)
        sample_time = timestamp.new_zeros(B)

        x = img_feats
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)

        memory = head.input_projection(memory)
        lane_embedding = head.instance_embedding_lane.weight.unsqueeze(-2) + head.points_embedding_lane.weight.unsqueeze(0)
        reference_points_lane = head.reference_points_lane(lane_embedding).sigmoid().flatten(-2).unsqueeze(0).repeat(B, 1, 1)
        query_pos = head.query_pos(nerf_positional_encoding(reference_points_lane))
        tgt = head.instance_embedding_lane.weight.unsqueeze(0).repeat(B, 1, 1)
        query_embedding = head.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)

        self_attn_mask = (
            torch.zeros([head.num_lane + head.num_extra, head.num_lane + head.num_extra]).bool().to(x.device)
        )
        self_attn_mask[head.num_lanes_one2one + head.num_extra:, 0:head.num_lanes_one2one + head.num_extra] = True
        self_attn_mask[0:head.num_lanes_one2one + head.num_extra, head.num_lanes_one2one + head.num_extra:] = True
        temporal_attn_mask = (
            torch.zeros([head.num_lane + head.num_extra, head.num_lane + head.num_extra + head.memory_len]).bool().to(x.device)
        )
        temporal_attn_mask[:self_attn_mask.size(0), :self_attn_mask.size(1)] = self_attn_mask
        if head.with_mask:
            temporal_attn_mask[head.num_extra:, :head.num_extra] = True

        B_query_pos = query_pos.size(0)
        temp_reference_point = (memory_reference_point - head.pc_range[:3]) / (head.pc_range[3:6] - head.pc_range[0:3])
        temp_pos = head.query_pos(nerf_positional_encoding(temp_reference_point.flatten(-2)))
        temp_memory = memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B_query_pos, query_pos.size(1), 1, 1)

        if head.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(tgt[..., :1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([memory_timestamp, memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = head.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += head.time_embedding(pos2posemb1d(torch.zeros_like(tgt[..., :1])))
        temp_pos += head.time_embedding(pos2posemb1d(memory_timestamp).float())

        tgt = torch.cat([query_embedding, tgt], dim=1)
        query_pos = torch.cat([torch.zeros_like(query_embedding), query_pos], dim=1)

        outs_dec = head.transformer(tgt, memory, query_pos, pos_embed, temporal_attn_mask, temp_memory, temp_pos)

        vlm_memory = outs_dec[-1, :, :head.num_extra, :]
        outs_dec = outs_dec[:, :, head.num_extra:, :]

        outs_dec = torch.nan_to_num(outs_dec)

        lane_queries = outs_dec
        outputs_lane_preds = []
        outputs_lane_clses = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points_lane.clone())
            reference = reference.view(B, head.num_lane, head.n_control * 3)
            tmp = head.reg_branches[lvl](lane_queries[lvl])
            outputs_lanecls = head.cls_branches[lvl](lane_queries[lvl])

            tmp = tmp.reshape(B, head.num_lane, head.n_control * 3)
            tmp += reference
            tmp = tmp.sigmoid()

            outputs_coord = tmp
            outputs_coord = outputs_coord.reshape(B, head.num_lane, head.n_control, 3)
            outputs_lane_preds.append(outputs_coord)
            outputs_lane_clses.append(outputs_lanecls)

        all_lane_preds = torch.stack(outputs_lane_preds)
        all_lane_clses = torch.stack(outputs_lane_clses)

        all_lane_preds[..., 0:3] = (all_lane_preds[..., 0:3] * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3])
        all_lane_preds = all_lane_preds.flatten(-2)

        all_lane_cls_one2one = all_lane_clses[:, :, 0:head.num_lanes_one2one, :]
        all_lane_preds_one2one = all_lane_preds[:, :, 0:head.num_lanes_one2one, :]
        all_lane_cls_one2many = all_lane_clses[:, :, head.num_lanes_one2one:, :]
        all_lane_preds_one2many = all_lane_preds[:, :, head.num_lanes_one2one:, :]
        outs_dec_one2one = outs_dec[:, :, 0:head.num_lanes_one2one, :]
        outs_dec_one2many = outs_dec[:, :, head.num_lanes_one2one:, :]

        rec_reference_points = all_lane_preds_one2one[-1].reshape(outs_dec_one2one.shape[1], -1, head.n_control, 3)
        out_memory = outs_dec_one2one[-1]
        rec_score = all_lane_cls_one2one[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        _, topk_indexes = torch.topk(rec_score, head.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)

        memory_embedding = torch.cat([rec_memory, memory_embedding], dim=1)
        memory_timestamp = torch.cat([rec_timestamp, memory_timestamp], dim=1)
        memory_egopose = torch.cat([rec_ego_pose, memory_egopose], dim=1)
        memory_reference_point = torch.cat([rec_reference_points, memory_reference_point], dim=1)
        memory_reference_point = transform_reference_points_lane(memory_reference_point, ego_pose, reverse=False)
        memory_timestamp -= timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time -= timestamp
        memory_egopose = ego_pose.unsqueeze(1) @ memory_egopose

        vlm_memory = head.output_projection(vlm_memory)

        return all_lane_cls_one2one, all_lane_preds_one2one, all_lane_cls_one2many, \
            all_lane_preds_one2many, outs_dec_one2one, outs_dec_one2many, vlm_memory, \
            memory_embedding, memory_timestamp, memory_egopose, memory_reference_point, \
            sample_time

    def forward(self, img, cam_intrinsic, img2lidars, command, can_bus, is_first_frame,
                ego_pose, timestamp, ego_pose_inv,
                memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox,
                memory_egopose_bbox, memory_canbus_bbox, sample_time_bbox,
                memory_scene_query_bbox, scene_memory_timestamp_bbox,
                memory_timestamp_map, sample_time_map, memory_egopose_map,
                memory_embedding_map, memory_reference_point_map):
        B, N, _, _, _ = img.size()
        img = img.view(N, *img.shape[2:])
        img = self.mod.grid_mask(img)
        img_feats = self.mod.img_backbone(img)
        _, C, H, W = img_feats[0].size()
        img_feats = img_feats[0].view(B, N, C, H, W)

        pos_embed_input = self.pos_embedding(cam_intrinsic=cam_intrinsic, img2lidars=img2lidars)
        pos_embed = self.mod.position_encoder(pos_embed_input)

        all_cls_scores, all_bbox_preds, all_traffic_states, \
            all_traj_preds, all_traj_cls_scores, \
            vlm_memory_bbox, \
            memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox, memory_egopose_bbox, memory_canbus_bbox, sample_time_bbox, memory_scene_query_bbox, scene_memory_timestamp_bbox = \
            self.bboxOrionHead_forward(img_feats, pos_embed,
                                       command, can_bus, is_first_frame,
                                       memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox,
                                       memory_egopose_bbox, memory_canbus_bbox,
                                       sample_time_bbox,
                                       memory_scene_query_bbox, scene_memory_timestamp_bbox,
                                       ego_pose, timestamp, ego_pose_inv)

        all_lane_cls_one2one, all_lane_preds_one2one, \
            all_lane_cls_one2many, all_lane_preds_one2many, \
            outs_dec_one2one, outs_dec_one2many, vlm_memory_map, \
            memory_embedding_map, memory_timestamp_map, memory_egopose_map, memory_reference_point_map, sample_time_map = \
            self.mapOrionHeadM_forward(img_feats, pos_embed, timestamp, ego_pose,
                                       ego_pose_inv, is_first_frame,
                                       sample_time_map,
                                       memory_timestamp_map, memory_egopose_map,
                                       memory_embedding_map, memory_reference_point_map)

        vision_embeded = torch.cat([vlm_memory_bbox, vlm_memory_map], dim=1)

        return vision_embeded, \
            all_cls_scores, all_bbox_preds, all_traffic_states, \
            all_traj_preds, all_traj_cls_scores, \
            all_lane_cls_one2one, all_lane_preds_one2one, all_lane_cls_one2many, all_lane_preds_one2many, \
            outs_dec_one2one, outs_dec_one2many, \
            memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox, \
            memory_egopose_bbox, memory_canbus_bbox, sample_time_bbox, \
            memory_scene_query_bbox, scene_memory_timestamp_bbox, \
            memory_embedding_map, memory_timestamp_map, memory_egopose_map, memory_reference_point_map, sample_time_map


def large_onnx_simplify_mixed_precision(onnx_base_dir, onnx_model_load_path,
                                         out_onnx_name_suffix="_sim_mixed.onnx",
                                         fp16_nodes=None):
    """
    Simplify ONNX model and mark mixed-precision nodes.

    Plan A mixed-precision strategy:
    - Nodes with names containing fp16_nodes keys -> keep FP16
    - Nodes with 'inverse_sigmoid', 'LayerNorm', 'Softmax', 'sigmoid' -> FORCEFP32
    - All other nodes -> FORCEFP32 (conservative default)

    The FORCEFP32 suffix is used by TensorRT's --layerPrecisions=*_FORCEFP32:fp32
    """
    if fp16_nodes is None:
        fp16_nodes = ["img_backbone"]

    import onnx
    import onnx_graphsurgeon as gs
    try:
        from onnxsim import simplify
    except ImportError:
        print("WARNING: onnxsim not available, skipping simplification")
        return onnx_model_load_path

    output_dir = os.path.dirname(onnx_base_dir) if os.path.isfile(onnx_base_dir) else onnx_base_dir
    out_model_path = os.path.join(output_dir, os.path.basename(onnx_model_load_path).replace(".onnx", out_onnx_name_suffix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    onnx_model = onnx.load(onnx_model_load_path)
    graph = gs.import_onnx(onnx_model)

    fp32_keywords = [
        'inverse_sigmoid', 'LayerNorm', 'layer_norm',
        'Softmax', 'softmax',
        'Log', 'log',
        'Sigmoid', 'sigmoid',
        'position_encoder', 'query_pos', 'time_embedding',
        'ego_pose_pe', 'nerf_positional_encoding', 'pos2posemb1d',
        'input_projection', 'output_projection',
        'cls_branches', 'reg_branches', 'tl_branches',
        'traj_branches', 'traj_cls_branches',
        'reference_points', 'instance_embedding', 'points_embedding',
        'memory_decoder', 'motion_decoder', 'motion_mode_query',
        'can_bus_embed',
    ]

    if len(fp16_nodes) > 0:
        for idx in range(len(graph.nodes)):
            node_name = graph.nodes[idx].name
            is_fp32_required = any(kw in node_name for kw in fp32_keywords)
            node_op = graph.nodes[idx].op if hasattr(graph.nodes[idx], 'op') else ''

            if is_fp32_required or node_op in ['Softmax', 'Log', 'Sigmoid', 'LayerNormalization']:
                graph.nodes[idx].name = node_name + "_FORCEFP32"

    graph.toposort().cleanup()
    onnx_model = gs.export_onnx(graph)

    try:
        from onnxsim_large_model.onnx_utils import set_onnx_input_shape, get_onnx_size_mb
        from onnxsim_large_model.compress_model import compress_onnx_model, uncompress_onnx_model
        onnx_model, removed_inits = compress_onnx_model(onnx_model)
        print("compress model success")
        onnx_model = set_onnx_input_shape(onnx_model, "")
        skipped_optimizer = "".split(";")
        onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizer)
        if not check:
            raise ValueError(f"simplify compressed model {onnx_model_load_path} failed")
        print("simplify model success")
        onnx_model = uncompress_onnx_model(onnx_model, removed_inits)
        print("uncompress model success")
        save_extern = (get_onnx_size_mb(onnx_model) > 1.8 * 1024)
        onnx.save(onnx_model, out_model_path, save_as_external_data=save_extern)
    except Exception as e:
        print(f"Large model simplification failed ({e}), trying standard simplification...")
        try:
            onnx_model, check = simplify(onnx_model)
            if not check:
                raise ValueError("Standard simplification also failed")
            onnx.save(onnx_model, out_model_path)
        except Exception as e2:
            print(f"Standard simplification also failed ({e2}), saving unsimplified model")
            onnx.save(onnx_model, out_model_path)

    print(f"Simplified ONNX saved to: {out_model_path}")
    return out_model_path


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if not os.path.exists(args.onnx_save_path):
        os.makedirs(args.onnx_save_path, exist_ok=True)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    cfg.model.img_backbone["flash_attn"] = False
    cfg.model.img_backbone["with_cp"] = False
    cfg.model.pts_bbox_head["transformer"]["flash_attn"] = False
    cfg.model.pts_bbox_head["transformer"]["with_cp"] = False
    cfg.model.pts_bbox_head["motion_transformer_decoder"]["flash_attn"] = False
    cfg.model.pts_bbox_head["motion_transformer_decoder"]["with_cp"] = False
    cfg.model.pts_bbox_head["memory_decoder_transformer"]["flash_attn"] = False
    cfg.model.pts_bbox_head["memory_decoder_transformer"]["with_cp"] = False
    cfg.model.map_head["transformer"]["flash_attn"] = False
    cfg.model.map_head["transformer"]["with_cp"] = False

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        for m in model.modules():
            if hasattr(m, 'fp16_enabled'):
                m.fp16_enabled = True

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model.float()
    model.cpu()
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and not module.elementwise_affine:
            module.elementwise_affine = True
            module.weight = nn.Parameter(torch.ones(module.normalized_shape))
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape))

    img = torch.randn(1, 6, 3, args.img_size, args.img_size, dtype=torch.float32)
    cam_intrinsic = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1)
    img2lidars = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1)
    command = torch.zeros(1, dtype=torch.float32)
    can_bus = torch.zeros(1, 18, dtype=torch.float32)
    is_first_frame = torch.ones([1])
    ego_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    timestamp = torch.zeros(1, dtype=torch.float32)
    ego_pose_inv = torch.eye(4, dtype=torch.float32).unsqueeze(0)

    memory_embedding_bbox_in = np.ones([1, 900, 256])
    memory_reference_point_bbox_in = np.ones([1, 900, 3])
    memory_timestamp_bbox_in = np.ones([1, 900, 1])
    memory_egopose_bbox_in = np.ones([1, 900, 4, 4])
    memory_canbus_bbox_in = np.ones([1, 3, 19])
    sample_time_bbox_in = np.ones([1])
    memory_scene_query_bbox_in = np.ones([1, 272, 256])
    scene_memory_timestamp_bbox_in = np.ones([1, 272, 1])

    memory_timestamp_map_in = np.ones([1, 900, 1])
    sample_time_map_in = np.ones([1])
    memory_egopose_map_in = np.ones([1, 900, 4, 4])
    memory_embedding_map_in = np.ones([1, 900, 256])
    memory_reference_point_map_in = np.ones([1, 900, 11, 3])

    onnx_device = 'cpu'
    input_precision = np.float32

    inputs = [
        img,
        cam_intrinsic,
        img2lidars,
        command,
        can_bus.to(torch.float32),
        is_first_frame,
        ego_pose,
        timestamp,
        ego_pose_inv,
        torch.from_numpy(memory_embedding_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_reference_point_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_timestamp_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_egopose_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_canbus_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(sample_time_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_scene_query_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(scene_memory_timestamp_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_timestamp_map_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(sample_time_map_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_egopose_map_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_embedding_map_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_reference_point_map_in.astype(input_precision)).to(onnx_device),
    ]

    input_names = [
        "img",
        "cam_intrinsic",
        "img2lidars",
        "command",
        "can_bus",
        "is_first_frame",
        "ego_pose",
        "timestamp",
        "ego_pose_inv",
        "memory_embedding_bbox_in",
        "memory_reference_point_bbox_in",
        "memory_timestamp_bbox_in",
        "memory_egopose_bbox_in",
        "memory_canbus_bbox_in",
        "sample_time_bbox_in",
        "memory_scene_query_bbox_in",
        "scene_memory_timestamp_bbox_in",
        "memory_timestamp_map_in",
        "sample_time_map_in",
        "memory_egopose_map_in",
        "memory_embedding_map_in",
        "memory_reference_point_map_in",
    ]

    output_names = [
        "vision_embeded",
        "all_cls_scores",
        "all_bbox_preds",
        "all_traffic_states",
        "all_traj_preds",
        "all_traj_cls_scores",
        "all_lane_cls_one2one",
        "all_lane_preds_one2one",
        "all_lane_cls_one2many",
        "all_lane_preds_one2many",
        "outs_dec_one2one",
        "outs_dec_one2many",
        "memory_embedding_bbox_out",
        "memory_reference_point_bbox_out",
        "memory_timestamp_bbox_out",
        "memory_egopose_bbox_out",
        "memory_canbus_bbox_out",
        "sample_time_bbox_out",
        "memory_scene_query_bbox_out",
        "scene_memory_timestamp_bbox_out",
        "memory_embedding_map_out",
        "memory_timestamp_map_out",
        "memory_egopose_map_out",
        "memory_reference_point_map_out",
        "sample_time_map_out",
    ]

    model.eval()
    proxy = OrionLiteVisionProxy(model, img_size=args.img_size)

    output_onnx_pth = os.path.join(args.onnx_save_path, "orion_lite_vision.onnx")

    print("=" * 60)
    print("Exporting Orion-Lite Vision Module to ONNX...")
    print(f"  Output: {output_onnx_pth}")
    print("=" * 60)

    with torch.no_grad():
        torch.onnx.export(
            proxy,
            tuple(inputs),
            output_onnx_pth,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            # The torch>=2.9 default (torch.export/dynamo based) fails on this graph
            # ("No Adapter To Version $17 for Resize"); the trace matches the one the
            # committed benchmark evidence was produced with.
            dynamo=False,
        )

    print(f"\nRaw ONNX exported to: {output_onnx_pth}")

    onnx_size_mb = os.path.getsize(output_onnx_pth) / (1024 * 1024)
    print(f"ONNX file size: {onnx_size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("Running ONNX simplification with mixed-precision marking...")
    print("=" * 60)

    sim_onnx_path = large_onnx_simplify_mixed_precision(
        onnx_base_dir=args.onnx_save_path,
        onnx_model_load_path=output_onnx_pth,
        out_onnx_name_suffix="_sim_mixed.onnx",
        fp16_nodes=["img_backbone"]
    )

    if args.save_test_data:
        test_data_dir = os.path.join(args.onnx_save_path, "test_data")
        os.makedirs(test_data_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("Saving test data for accuracy verification...")
        print("=" * 60)

        with torch.no_grad():
            ref_outputs = proxy(*inputs)

        for i, inp in enumerate(inputs):
            name = input_names[i]
            arr = inp.detach().cpu().numpy() if isinstance(inp, torch.Tensor) else inp
            np.save(os.path.join(test_data_dir, f"{name}.npy"), arr)
            print(f"  Saved input: {name} shape={arr.shape}")

        for i, out in enumerate(ref_outputs):
            name = output_names[i]
            if isinstance(out, torch.Tensor):
                arr = out.detach().cpu().float().numpy()
            else:
                arr = np.array(out, dtype=np.float32)
            np.save(os.path.join(test_data_dir, f"{name}_ref.npy"), arr)
            print(f"  Saved output: {name} shape={arr.shape} mean={arr.mean():.6f} std={arr.std():.6f}")

        print(f"\nTest data saved to: {test_data_dir}")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Transfer ONNX to Orin:")
    print(f"   rsync -avP {sim_onnx_path} orin:{sim_onnx_path}")
    print(f"2. Compile TensorRT engine (mixed precision):")
    print(f"   trtexec --onnx={os.path.basename(sim_onnx_path)} \\")
    print(f"     --saveEngine=orion_lite_vision_mixed.engine \\")
    print(f"     --fp16 --precisionConstraints=obey \\")
    print(f"     --layerPrecisions=*_FORCEFP32:fp32 \\")
    print(f"     --useCudaGraph --verbose")
    print(f"3. Compile TensorRT engine (full FP16 for comparison):")
    print(f"   trtexec --onnx={os.path.basename(sim_onnx_path)} \\")
    print(f"     --saveEngine=orion_lite_vision_fp16.engine \\")
    print(f"     --fp16 --useCudaGraph --verbose")


if __name__ == '__main__':
    main()
