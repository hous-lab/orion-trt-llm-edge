import argparse
import numpy as np
import os
import warnings
import io

import torch
from torch import nn
from torch.nn import DataParallel
import torch.nn.functional as F

import mmcv
from mmcv.utils import get_dist_info, init_dist, wrap_fp16_model, set_random_seed, Config, DictAction, load_checkpoint
from mmcv.models import build_model, fuse_conv_bn
from mmcv.datasets import build_dataset, build_dataloader, replace_ImageToTensor
from mmcv.models.utils.transformer import inverse_sigmoid
from mmcv.models.utils.positional_encoding import pos2posemb1d, nerf_positional_encoding
from mmcv.utils.misc import MLN, topk_gather, transform_reference_points_lane, transform_reference_points, memory_refresh

def parse_args():
    parser = argparse.ArgumentParser(description='Orion ONNX export(Vision part).')
    parser.add_argument('config',help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file', default=None)
    parser.add_argument('onnx_save_path',help='onnx_save path')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

class OrionVisionTrtProxy(nn.Module):
    def __init__(self, mod, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod
        
    def locations(self, stride, pad_h, pad_w):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = 40, 40
        device = "cpu"
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
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
        pad_h, pad_w, _ = (640, 640, 3)
        memory_centers = self.locations(16, pad_h, pad_w)[None].repeat(1*6, 1, 1, 1)

        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = cam_intrinsic.size(0)

        cam_intrinsic = torch.stack([cam_intrinsic[..., 0, 0], cam_intrinsic[..., 1, 1]], dim=-1)
        cam_intrinsic = torch.abs(cam_intrinsic) / 1e3
        cam_intrinsic = cam_intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = cam_intrinsic.size(1)

        num_sample_tokens = LEN
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.mod.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.mod.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.mod.position_range[0:3]) / (self.mod.position_range[3:6] - self.mod.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        return pos_embed

    def mapOrionHeadM_forward(self, img_feats, pos_embed, timestamp, ego_pose, 
                                ego_pose_inv, is_first_frame, sample_time, 
                                memory_timestamp,  memory_egopose,
                                memory_embedding, memory_reference_point
                            ):
        head = self.mod.map_head

        B = img_feats.size(0)
        memory_timestamp += timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time += timestamp
        sample_time_x = (torch.abs(sample_time) < 2.0).to(img_feats.dtype)
        sample_time_x *= (1.-is_first_frame)
        memory_egopose = ego_pose_inv.unsqueeze(1) @ memory_egopose
        memory_reference_point = transform_reference_points_lane(memory_reference_point, ego_pose_inv, reverse=False)
        memory_timestamp = memory_refresh(memory_timestamp[:, :head.memory_len], sample_time_x)
        memory_timestamp *= (1.-is_first_frame)
        memory_reference_point = memory_refresh(memory_reference_point[:, :head.memory_len], sample_time_x)
        memory_reference_point *= (1.-is_first_frame)
        memory_embedding = memory_refresh(memory_embedding[:, :head.memory_len], sample_time_x)
        memory_embedding *= (1.-is_first_frame)
        memory_egopose = memory_refresh(memory_egopose[:, :head.memory_len], sample_time_x)
        memory_egopose *= (1.-is_first_frame)
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
            torch.zeros([head.num_lane+head.num_extra, head.num_lane+head.num_extra]).bool().to(x.device)
        )
        self_attn_mask[head.num_lanes_one2one+head.num_extra:, 0: head.num_lanes_one2one+head.num_extra] = True
        self_attn_mask[0: head.num_lanes_one2one+head.num_extra, head.num_lanes_one2one+head.num_extra:] = True
        temporal_attn_mask = (
            torch.zeros([head.num_lane+head.num_extra, head.num_lane+head.num_extra+head.memory_len]).bool().to(x.device)
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
            rec_ego_motion = torch.cat([torch.zeros_like(tgt[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([memory_timestamp, memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = head.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += head.time_embedding(pos2posemb1d(torch.zeros_like(tgt[...,:1])))
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
            reference = reference.view(B, head.num_lane, head.n_control*3)
            tmp = head.reg_branches[lvl](lane_queries[lvl])
            outputs_lanecls = head.cls_branches[lvl](lane_queries[lvl])

            tmp = tmp.reshape(B, head.num_lane, head.n_control*3)
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

        all_lane_cls_one2one = all_lane_clses[:, :, 0: head.num_lanes_one2one, :]
        all_lane_preds_one2one = all_lane_preds[:, :, 0: head.num_lanes_one2one, :]
        all_lane_cls_one2many = all_lane_clses[:, :, head.num_lanes_one2one:, :]
        all_lane_preds_one2many = all_lane_preds[:, :, head.num_lanes_one2one:, :]
        outs_dec_one2one = outs_dec[:, :, 0: head.num_lanes_one2one, :]
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
        memory_egopose= torch.cat([rec_ego_pose, memory_egopose], dim=1)
        memory_reference_point = torch.cat([rec_reference_points, memory_reference_point], dim=1)
        memory_reference_point = transform_reference_points_lane(memory_reference_point, ego_pose, reverse=False)
        memory_timestamp -= timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time -= timestamp
        memory_egopose = ego_pose.unsqueeze(1) @ memory_egopose

        vlm_memory = head.output_projection(vlm_memory)

        return all_lane_cls_one2one, all_lane_preds_one2one, all_lane_cls_one2many,  \
                all_lane_preds_one2many, outs_dec_one2one, outs_dec_one2many, vlm_memory, \
                memory_embedding, memory_timestamp, memory_egopose, memory_reference_point, \
                sample_time

    def bboxOrionHead_forward(self, img_feats, pos_embed, command, can_bus, is_first_frame,
                               memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_canbus,
                               sample_time, 
                               memory_scene_query, scene_memory_timestamp,
                               ego_pose, timestamp, ego_pose_inv):
        head = self.mod.pts_bbox_head
        # pts_bbox_head forward

        # def pre_update_memory start
        B, N, C, H, W = img_feats.shape
        memory_timestamp += timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time += timestamp
        sample_time_x = (torch.abs(sample_time) < 2.0).to(img_feats.dtype)
        sample_time_x *= (1.-is_first_frame)

        scene_memory_timestamp += timestamp.unsqueeze(-1).unsqueeze(-1)
        scene_memory_timestamp = memory_refresh(scene_memory_timestamp[:, :head.scence_memory_len], sample_time_x)
        memory_scene_query = memory_refresh(memory_scene_query[:, :head.scence_memory_len], sample_time_x) 

        memory_egopose = ego_pose_inv.unsqueeze(1) @ memory_egopose
        memory_reference_point = transform_reference_points(memory_reference_point, ego_pose_inv, reverse=False)
        memory_timestamp = memory_refresh(memory_timestamp[:, :head.memory_len], sample_time_x)
        memory_timestamp *= (1.-is_first_frame)
        memory_reference_point = memory_refresh(memory_reference_point[:, :head.memory_len], sample_time_x)
        memory_reference_point *= (1.-is_first_frame)
        memory_embedding = memory_refresh(memory_embedding[:, :head.memory_len], sample_time_x)
        memory_embedding *= (1.-is_first_frame)
        memory_egopose = memory_refresh(memory_egopose[:, :head.memory_len], sample_time_x)
        memory_egopose *= (1.-is_first_frame)
        memory_canbus = memory_refresh(memory_canbus[:, :head.can_bus_len], sample_time_x)
        memory_canbus *= (1.-is_first_frame)
        sample_time = timestamp.new_zeros(B)
        sample_time *= (1.-is_first_frame)

        pseudo_reference_points = head.pseudo_reference_points.weight * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3]
        memory_reference_point[:, :head.num_propagated]  = memory_reference_point[:, :head.num_propagated] + (1-sample_time_x).view(B, 1, 1) * pseudo_reference_points
        memory_egopose[:, :head.num_propagated]  = memory_egopose[:, :head.num_propagated] + (1-sample_time_x).view(B, 1, 1, 1) * torch.eye(4, device=sample_time_x.device)

        num_tokens = N * H * W
        memory = img_feats.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = head.input_projection(memory)

        reference_points = head.reference_points.weight
        reference_points = torch.cat([torch.zeros_like(reference_points[:head.num_extra]), reference_points], dim=0)
        # self.prepare_for_dn
        attn_mask = None
        # if head.with_mask:
        tgt_size = head.num_query + head.memory_len + head.num_extra
        query_size = head.num_query + head.num_propagated + head.num_extra
        attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
        attn_mask[head.num_extra:, :head.num_extra] = True
        
        reference_points = reference_points.unsqueeze(0).repeat(B, 1, 1)
    
        query_pos = head.query_pos(nerf_positional_encoding(reference_points.repeat(1, 1, head.n_control)))
        tgt = torch.zeros_like(query_pos)

        # def temporal_alignment start
        temp_reference_point = (memory_reference_point - head.pc_range[:3]) / (head.pc_range[3:6] - head.pc_range[0:3])
        temp_pos = head.query_pos(nerf_positional_encoding(temp_reference_point.repeat(1, 1, head.n_control))) 
        temp_memory = memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        # if head.with_ego_pos:
        rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
        rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
        memory_ego_motion = torch.cat([memory_timestamp, memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
        memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
        temp_pos = head.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += head.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += head.time_embedding(pos2posemb1d(memory_timestamp).float())

        # if head.num_propagated > 0:
        tgt = torch.cat([tgt, temp_memory[:, :head.num_propagated]], dim=1)
        query_pos = torch.cat([query_pos, temp_pos[:, :head.num_propagated]], dim=1)
        reference_points = torch.cat([reference_points, temp_reference_point[:, :head.num_propagated]], dim=1)
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+head.num_propagated, 1, 1)
        temp_memory = temp_memory[:, head.num_propagated:]
        temp_pos = temp_pos[:, head.num_propagated:]
        # def temporal_alignment end

        tgt[:, :head.num_extra, :] = head.query_embedding.weight.unsqueeze(0)
        query_pos[:, :head.num_extra, :] = query_pos[:, :head.num_extra, :] * 0

        outs_dec = head.transformer(tgt, memory, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)
        reference_points = reference_points[:, head.num_extra:, :]

        outs_dec = torch.nan_to_num(outs_dec)
        vlm_memory = outs_dec[-1, :, :head.num_extra, :]
        outs_dec = outs_dec[:, :, head.num_extra:, :]

        current_query = head.memory_query.weight.unsqueeze(0).repeat(B,1,1) # (4, 16, 256)
        time_embeding = head.scene_time_embedding(pos2posemb1d(scene_memory_timestamp).float()) # (4, 256, 1)  -> (4, 256, 256) 
        cur_time_embeding = torch.zeros_like(current_query) # (4, 16, 256) 
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
            assert reference.shape[-1] == 3
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

        # calculate outputs_trajs and outputs_trajs_classes
        motion_query = outs_dec[-1].permute(1, 0, 2)  # [A, B, D]
        mode_query = head.motion_mode_query.weight  # [fut_mode, D]
        motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
        motion_query = motion_query.permute(1, 0, 2) # batch first
        motion_hs = head.motion_decoder(query=motion_query, key=motion_query,
                                        query_pos=None, key_pos=None, attn_mask=None)
        ca_motion_query = motion_hs.flatten(0, 1).unsqueeze(0)
        num_agent = outs_dec[-1].shape[1]
        motion_hs = motion_hs.unflatten(dim=1, sizes=(num_agent, head.fut_mode))
        ca_motion_query = ca_motion_query.squeeze(0).unflatten(
            dim=0, sizes=(B, num_agent, head.fut_mode))
        motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        # 
        outputs_traj = head.traj_branches[0](motion_hs)
        outputs_trajs.append(outputs_traj)
        outputs_traj_class = head.traj_cls_branches[0](motion_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
        outputs_trajs = torch.stack(outputs_trajs)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        all_traj_preds = outputs_trajs.repeat(all_bbox_preds.shape[0], 1, 1, 1, 1)
        all_traj_cls_scores = outputs_trajs_classes.repeat(all_bbox_preds.shape[0], 1, 1, 1)

        # def post_update_memory(
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
        memory_egopose= torch.cat([rec_ego_pose, memory_egopose], dim=1)
        memory_reference_point = torch.cat([rec_reference_points, memory_reference_point], dim=1)
        memory_canbus = torch.cat([rec_can_bus.unsqueeze(-2), memory_canbus], dim=1)
        memory_reference_point = transform_reference_points(memory_reference_point, ego_pose, reverse=False)
        memory_timestamp -= timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time -= timestamp
        memory_egopose = ego_pose.unsqueeze(1) @ memory_egopose

        scene_memory_timestamp = torch.cat([torch.zeros_like(scene_memory_timestamp[:, :head.num_memory,:]), scene_memory_timestamp], dim=1)
        scene_memory_timestamp -= timestamp.unsqueeze(-1).unsqueeze(-1)
        memory_scene_query = torch.cat([history_query.detach(), memory_scene_query], dim=1)

        return  all_cls_scores, all_bbox_preds, \
                all_traffic_states, \
                all_traj_preds, all_traj_cls_scores, \
                vlm_memory, \
                memory_embedding, memory_reference_point, memory_timestamp, \
                memory_egopose, memory_canbus, sample_time, \
                memory_scene_query, scene_memory_timestamp 

    def forward(self, img, cam_intrinsic, img2lidars, command, can_bus, is_first_frame, ego_pose, timestamp, ego_pose_inv, input_ids, 
                # ego_fut_cmd,
                memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox,
                memory_egopose_bbox, memory_canbus_bbox, sample_time_bbox, 
                memory_scene_query_bbox, scene_memory_timestamp_bbox,
                memory_timestamp_map, sample_time_map, memory_egopose_map, 
                memory_embedding_map, memory_reference_point_map):
        # img.size() torch.Size([1, 6, 3, 640, 640])
        # img_feats = self.mod.extract_img_feat(img) # [6, 1, 1024, 40, 40] ❌，应该是[1, 6, 1024, 40, 40]
        B, N,_,_,_ = img.size()
        img.squeeze_()
        img = self.mod.grid_mask(img)
        img_feats = self.mod.img_backbone(img)
        _, C, H, W = img_feats[0].size()
        img_feats = img_feats[0].view(B, N, C, H, W)

        pos_embed_input = self.pos_embedding(cam_intrinsic=cam_intrinsic, img2lidars=img2lidars)
        pos_embed = self.mod.position_encoder(pos_embed_input)

        # bboxOrionHead
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

        # mapOrionHead
        all_lane_cls_one2one, all_lane_preds_one2one,  \
            all_lane_cls_one2many, all_lane_preds_one2many, \
            outs_dec_one2one, outs_dec_one2many, vlm_memory_map, \
            memory_embedding_map, memory_timestamp_map, memory_egopose_map, memory_reference_point_map, sample_time_map = \
        self.mapOrionHeadM_forward(img_feats, pos_embed, timestamp, ego_pose, 
                                    ego_pose_inv, is_first_frame,
                                    sample_time_map, 
                                    memory_timestamp_map, memory_egopose_map, 
                                    memory_embedding_map, memory_reference_point_map)
        
        vision_embeded = torch.cat([vlm_memory_bbox, vlm_memory_map], dim=1)
        # 确保device一致
        input_ids = input_ids.to(vision_embeded.device)
        # ego_feature = self.mod.lm_head.inference_ego(
        ego_feature, hidden_states_out, inputs_embeds_out = self.inference_ego(
            inputs=input_ids, # torch.Size([1, 72]) 'torch.cuda.LongTensor'
            images=vision_embeded, # torch.Size([1, 529, 4096]) 'torch.cuda.FloatTensor'
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=320,
            use_cache=True,
            return_ego_feature=True
        )

        # ego_feature torch.Size([1, 4096])
        ego_feature = ego_feature.to(torch.float32)
        current_states = ego_feature.unsqueeze(1)

        sample, _ = self.mod.distribution_forward(current_states)
        hidden_states = ego_feature.unsqueeze(1)
        states_hs, _ = \
            self.mod.future_states_predict(1, sample, hidden_states, current_states)
        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
        ego_fut_trajs_list = []
        for i in range(self.mod.fut_ts):
            outputs_ego_trajs = self.mod.ego_fut_decoder(ego_query_hs[i]).reshape(1, self.mod.ego_fut_mode, 2) # [1, 6, 2]
            ego_fut_trajs_list.append(outputs_ego_trajs)
        ego_fut_pred = torch.stack(ego_fut_trajs_list, dim=2) # [1, 6, 6, 2]

        return  ego_fut_pred, \
                vision_embeded, hidden_states_out, inputs_embeds_out, \
                all_cls_scores, all_bbox_preds, all_traffic_states, \
                all_traj_preds, all_traj_cls_scores, \
                all_lane_cls_one2one, all_lane_preds_one2one,\
                all_lane_cls_one2many, all_lane_preds_one2many, outs_dec_one2one, outs_dec_one2many, \
                memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox, memory_egopose_bbox, memory_canbus_bbox, sample_time_bbox, \
                memory_scene_query_bbox, scene_memory_timestamp_bbox, \
                memory_embedding_map, memory_timestamp_map, memory_egopose_map, memory_reference_point_map, sample_time_map

    def inference_ego(
        self,
        inputs,
        images,
        image_sizes = None,
        return_ego_feature = False,
        **kwargs,
    ):
        lm_head = self.mod.lm_head

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        (
            inputs,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
            new_input_ids
        ) = lm_head.base_model.model.prepare_inputs_labels_for_multimodal(
            inputs,
            position_ids,
            attention_mask,
            None,
            None,
            images,
            image_sizes=image_sizes
        )

        output_attentions = lm_head.config.output_attentions
        output_hidden_states = (lm_head.config.output_hidden_states)
        return_dict = lm_head.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = lm_head.base_model.model.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values= None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # find 2d position  self.model.to(torch.float32)
        hidden_states = outputs[0]
        # loc_positions = ( (new_input_ids == self.mod.lm_head.config.waypoint_token_idx)) # (1, 600)[BOOL]
        # selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]

        # # 使用等值比较创建掩码
        # mask = (new_input_ids == self.mod.lm_head.config.waypoint_token_idx).float()
        # # 找到每个batch中第一个匹配的位置
        # _, indices = torch.max(mask, dim=1, keepdim=True)
        # 使用gather获取对应位置的 hidden states
        # selected_hidden_states = hidden_states[0, -2].squeeze(1)
        selected_hidden_states = hidden_states[0, -2].unsqueeze(0)

        return selected_hidden_states, hidden_states, inputs_embeds



import onnx
import onnxsim
from onnxsim import simplify
from onnxsim_large_model.onnx_utils import set_onnx_input_shape, get_onnx_size_mb
from onnxsim_large_model.compress_model import compress_onnx_model, uncompress_onnx_model
import onnx_graphsurgeon as gs

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if not os.path.exists(args.onnx_save_path):
        os.makedirs(args.onnx_save_path, exist_ok=True)
    output_onnx_pth = args.onnx_save_path+args.config.split("/")[-1].split(".")[0]+".onnx"

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
    assert distributed == False, "onnx export only support non distributed launch."

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # cfg.model.lm_head = None
    # cfg.model.tokenizer = None
    # cfg.model.use_lora = False
    # cfg.model.use_gen_token = False
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
        # wrap_fp16_model(model)
        custom_wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    for i, idata in enumerate(data_loader):
        if i == 0:
            data = idata
            break
    img = data['img'][0]
    cam_intrinsic = data['cam_intrinsic'][0]
    img2lidars = data['lidar2img'][0].inverse()
    command =data['command'][0]
    can_bus =data['can_bus'][0]
    is_first_frame = torch.ones([1])
    ego_pose = data['ego_pose'][0]
    timestamp =data['timestamp'][0]
    ego_pose_inv = data['ego_pose_inv'][0]
    input_ids = data['input_ids'][0][0][0].unsqueeze(0) # (1,71), data['input_ids'][0][0][0]:(71)

    memory_embedding_bbox_in = np.ones([1,900,256])
    memory_reference_point_bbox_in = np.ones([1,900,3])
    memory_timestamp_bbox_in = np.ones([1,900,1])
    memory_egopose_bbox_in = np.ones([1,900,4,4])
    memory_canbus_bbox_in = np.ones([1,3,19]) # b2d 19
    sample_time_bbox_in = np.ones([1])
    memory_scene_query_bbox_in = np.ones([1,272,256]) # 256 + 16(num_memory)
    scene_memory_timestamp_bbox_in = np.ones([1,272,1])

    memory_timestamp_map_in = np.ones([1,900,1])
    sample_time_map_in = np.ones([1])
    memory_egopose_map_in = np.ones([1,900,4,4])
    memory_embedding_map_in = np.ones([1,900,256])
    memory_reference_point_map_in = np.ones([1,900,11,3])

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
        input_ids.to(torch.int32),

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
        "input_ids",

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
        "ego_fut_pred",
        "vision_embeded",
        "hidden_states_out",
        "inputs_embeds_out",
        "all_cls_scores", 
        "all_bbox_preds",
        "all_traffic_states",
        "all_traj_preds",
        "all_traj_cls_scores",
        "all_lane_cls_one2one",
        "all_lane_preds_one2one",
        "all_lane_cls_one2many",
        'all_lane_preds_one2many',
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
    proxy = OrionVisionTrtProxy(model, cfg)
    
    torch.onnx.export(
        proxy,
        tuple(inputs),
        output_onnx_pth,
        input_names=input_names, 
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        verbose=True
    )

    large_onnx_simplify(onnx_base_dir=args.onnx_save_path, 
                        onnx_model_load_path=output_onnx_pth,
                        out_onnx_name_suffix="_sim_img_f16_bbox_map_llm_f32.onnx")
    
def large_onnx_simplify(onnx_base_dir, onnx_model_load_path, out_onnx_name_suffix="_sim.onnx"):
    output_dir = os.path.dirname(onnx_base_dir) +"_onnxsim/"
    out_model_path = os.path.join(output_dir, os.path.basename(onnx_model_load_path).replace(".onnx", out_onnx_name_suffix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    onnx_model = onnx.load(onnx_model_load_path)

    graph = gs.import_onnx(onnx_model)

    # 定义不支持 INT8 的操作类型
    fp32_ops = {"Softmax", "LayerNorm", "GELU", "Reshape", "Transpose"}
    # 修改后的节点标记逻辑
    for idx in range(len(graph.nodes)):
        node = graph.nodes[idx]
        
        # # 如果是不支持 INT8 的操作，强制使用 FP32
        # if node.op in fp32_ops:
        #     node.name = f"{node.name}_FORCEFP32"
        #     continue
            
        # # 对于 lm_head 相关的计算密集型操作使用 INT8
        # if "/model/" in node.name and node.op in {"MatMul", "Conv", "Gemm"}:
        #     node.name = f"{node.name}_FORCEINT8"
        #     continue
            
        # backbone 相关层保持 FP16
        if "img_backbone" in node.name or len(node.name) == 0:
            continue
            
        # 其他层使用 FP32
        node.name = f"{node.name}_FORCEFP32"
    
    # 添加验证步骤
    int8_count = sum(1 for node in graph.nodes if "_FORCEINT8" in node.name)
    fp32_count = sum(1 for node in graph.nodes if "_FORCEFP32" in node.name)
    fp16_count = len(graph.nodes) - int8_count - fp32_count
    
    print("\n=== 精度分布统计 ===")
    print(f"INT8 节点数量: {int8_count}")
    print(f"FP32 节点数量: {fp32_count}")
    print(f"FP16 节点数量: {fp16_count}")
    print(f"总节点数量: {len(graph.nodes)}")

    graph.toposort().cleanup()
    onnx_model = gs.export_onnx(graph)

    onnx_model, removed_inits = compress_onnx_model(onnx_model)
    print(f"compress model success")

    onnx_model = set_onnx_input_shape(onnx_model, "")
    skipped_optimizer = "".split(";")

    onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizer)
    if not check:
        raise ValueError(f"simplify compressed model {onnx_model_load_path} failed")
    print(f"simplify model success")

    onnx_model = uncompress_onnx_model(onnx_model, removed_inits)
    print(f"uncompress model success")

    save_extern = True or (get_onnx_size_mb(onnx_model) > 1.8 * 1024)
    onnx.save(onnx_model, out_model_path, save_as_external_data=save_extern)

custom_fp16 = dict(
                    map_head=False,
                    pts_bbox_head=False)
def custom_wrap_fp16_model(model):
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for module_name, v in custom_fp16.items():
        model._modules[module_name].fp16_enabled = v

if __name__ == '__main__':
    main()
