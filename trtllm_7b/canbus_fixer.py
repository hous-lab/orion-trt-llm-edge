"""完整 can_bus_embed 修复模块（复现 orion_head.py pts_bbox_head 的跨帧 memory 逻辑）。

memory_canbus 语义（orion_head.py）：
  - self.memory_canbus: (B, can_bus_len, 19)，存储历史帧的 can_bus 相对当前帧的位移
  - pre_update_memory: 每帧 forward 前，把已有历史帧 relative 到当前帧（-= 当前 can_bus[:3]/[-1]）
  - can_bus_input 用 pre_update 后的 memory_canbus
  - post_update_memory: forward 后，cat 当前帧 can_bus 到头部，再加回绝对（+= 当前 can_bus）

memory_egopose 语义（orion_head.py）：
  - pre_update_memory: ego_pose_inv @ memory_egopose（相对到当前帧）+ memory_refresh 截断
    + num_propagated 行 padding eye(4)（新场景第一帧）
  - post_update_memory: 注入 topk identity rec_ego_pose + ego_pose @ memory_egopose
"""
import os
import torch
import torch.nn as nn


def memory_refresh(memory, prev_exist):
    """与 mmcv/utils/misc.py 一致。"""
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:])
    return memory * prev_exist


class CanBusFixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.can_bus_len = 2
        self.memory_len = 600
        self.topk_proposals = 300
        self.num_propagated = 300
        self.can_bus_embed = nn.Sequential(
            nn.Linear(89, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4096),
        )
        self.reset()

    def reset(self):
        B = 1
        # memory 创建在 can_bus_embed 权重所在 device（支持 GPU 优化，S4）
        _dev = next(self.can_bus_embed.parameters()).device
        _dtype = next(self.can_bus_embed.parameters()).dtype
        self.memory_canbus = torch.zeros(B, self.can_bus_len, 19, device=_dev, dtype=_dtype)
        self.his_memory_canbus_len = torch.zeros(B, dtype=torch.int64, device=_dev)
        self.memory_egopose = torch.zeros(B, self.memory_len, 4, 4, device=_dev, dtype=_dtype)

    def pre_update(self, is_first_frame, scene_same, ego_pose_inv, can_bus):
        """对应 orion_head.py pre_update_memory（forward 开头）。"""
        B = 1
        x = (1.0 - is_first_frame.float()) * scene_same.float()  # (1,)
        if is_first_frame.item() == 1:
            self.his_memory_canbus_len = torch.zeros(B, dtype=torch.int64,
                                                      device=self.memory_canbus.device)
            # orion_head: 新场景第一帧, memory_egopose 相对变换后 + num_propagated padding
            # memory_egopose 已在 reset 时全零; 此处与官方一致: 仅对 :num_propagated 加 eye(4)
            self.memory_egopose[:, :self.num_propagated] += (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=self.memory_egopose.device)
            return
        # memory_egopose 相对变换（orion 486 行）
        self.memory_egopose = ego_pose_inv @ self.memory_egopose
        # memory_egopose refresh + 截断到 memory_len（orion 491 行）
        self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
        # num_propagated padding（orion 502-505 行）
        if self.num_propagated > 0:
            self.memory_egopose[:, :self.num_propagated] += (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=self.memory_egopose.device)
        # memory_canbus relative 到当前帧
        his_len = self.his_memory_canbus_len.to(torch.int64)
        for i in range(B):
            n = int(his_len[i].item())
            if n > 0:
                self.memory_canbus[i:i+1, :n, 1:4] -= can_bus[i:i+1, :3].unsqueeze(1)
                self.memory_canbus[i:i+1, :n, -1] -= can_bus[i:i+1, -1].unsqueeze(1)
        self.memory_canbus = memory_refresh(self.memory_canbus[:, :self.can_bus_len], x)
        self.his_memory_canbus_len = memory_refresh(self.his_memory_canbus_len, x)

    def post_update(self, ego_pose, command, can_bus):
        """对应 orion_head.py post_update_memory（forward 结尾）。
        1. memory_egopose = cat([identity(topk), memory_egopose])（orion 533 行）
        2. memory_egopose = ego_pose @ memory_egopose（orion 541 行）
        3. memory_canbus = cat([rec_can_bus, memory_canbus]) 后截取 can_bus_len。
        """
        B = 1
        # 1. 注入 topk identity rec_ego_pose（orion 533 行; rec_ego_pose 为 identity 重复 topk_proposals 行）
        rec_ego_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(
            B, self.topk_proposals, 1, 1).to(self.memory_egopose.dtype).to(self.memory_egopose.device)
        self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        # 2. ego_pose 变换（orion 541 行）
        self.memory_egopose = ego_pose.unsqueeze(1) @ self.memory_egopose
        # 3. memory_canbus 更新
        rec_can_bus = can_bus.clone()
        rec_can_bus[:, :3] = 0
        rec_can_bus[:, -1] = 0
        rec_can_bus = torch.cat([command.unsqueeze(-1), rec_can_bus], dim=-1)  # (1, 19)
        new_mem = torch.cat([rec_can_bus.unsqueeze(1), self.memory_canbus], dim=1)[:, :self.can_bus_len]
        self.memory_canbus = new_mem
        self.his_memory_canbus_len += 1
        # 加回绝对 can_bus（orion 第 543-544 行）
        his_len = self.his_memory_canbus_len.to(torch.int64)
        for i in range(1):
            n = int(his_len[i].item())
            if n > 0:
                self.memory_canbus[i:i+1, :n, 1:4] += can_bus[i:i+1, :3].unsqueeze(1)
                self.memory_canbus[i:i+1, :n, -1] += can_bus[i:i+1, -1].unsqueeze(1)

    def compute(self, command, can_bus):
        """计算当前帧 can_bus_embed: (1, 4096)。用 pre_update 后的 memory。"""
        B = 1
        rec_can_bus = can_bus.clone()
        rec_can_bus[:, :3] = 0
        rec_can_bus[:, -1] = 0
        rec_can_bus = torch.cat([command.unsqueeze(-1), rec_can_bus], dim=-1)  # (1, 19)
        memory_ego_pose = self.memory_egopose.reshape(B, -1, self.topk_proposals, 4, 4).flatten(-2)
        can_bus_input = torch.cat(
            [rec_can_bus, self.memory_canbus.flatten(-2), memory_ego_pose.mean(-2).flatten(-2)],
            dim=-1).float()
        return self.can_bus_embed(can_bus_input)

    def step(self, is_first_frame, scene_same, ego_pose_inv, ego_pose, command, can_bus):
        """每帧完整流程：pre_update → compute → post_update。返回 can_bus_embed。"""
        self.pre_update(is_first_frame, scene_same, ego_pose_inv, can_bus)
        with torch.no_grad():
            embed = self.compute(command, can_bus)
        self.post_update(ego_pose, command, can_bus)
        return embed


def load_can_bus_fixer(orion_pth, device='cpu'):
    if isinstance(orion_pth, str):
        sd = torch.load(orion_pth, map_location='cpu', weights_only=False)
        if 'state_dict' in sd:
            sd = sd['state_dict']
    else:
        sd = orion_pth

    prefix = 'pts_bbox_head.can_bus_embed.'
    cb_sd = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            cb_sd[k[len(prefix):]] = v

    model = CanBusFixer()
    model.eval()
    missing, unexpected = model.can_bus_embed.load_state_dict(cb_sd, strict=True)
    model.to(device)
    nbytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    print(f"[canbus] loaded {len(cb_sd)} keys, {nbytes / 2**20:.2f} MB -> {device}")
    return model
