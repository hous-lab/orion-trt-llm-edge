#!/usr/bin/env python
"""EVAViT 视觉 backbone 蒸馏（24层 → 12层）Feature Mimic。

设计：
  - teacher: 24 层 EVAViT（冻结，从 fused ckpt 加载 img_backbone）
  - student: 12 层 EVAViT（depth=12，用 teacher 前 12 层 blocks 初始化）
  - 数据: bench2drive 图像在线前向（无需预采集 npz）
  - Loss: 层对齐（student layer_i ↔ teacher layer_2i）+ 最终特征对齐 + 检测头 loss 可选

用法:
  CUDA_VISIBLE_DEVICES=0 python distill/train_vision_distill.py \
      --config adzoo/orion/configs/orion_distill_new_agent_with_mimic.py \
      --ckpt eval/fused_ckpts/orion_lite_subset8.pth \
      --out-dir work_dirs/vision_distill_12l \
      --epochs 1 --train-frames 173 --val-frames 50
"""
import argparse, os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('scripts'))

os.environ.setdefault('ORION_QFORMER_PATH', os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'ckpts', 'hf_models', 'pretrain_qformer'))

# 进程名掩码：统一显示为 Running-{short_hash}，避免 nvidia-smi 暴露项目路径
try:
    from utils.process_mask import mask_from_env
    mask_from_env()
except Exception:
    pass

from mmcv.utils import Config, load_checkpoint
from mmcv.models import build_model
from mmcv.datasets import build_dataset, build_dataloader


def build_student_backbone(teacher_backbone, new_depth=12):
    """从 teacher(24层) 构造 student(12层): 取前 new_depth 个 blocks + 共享 patch/pos/rope。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('eva_vit', 'mmcv/models/backbones/eva_vit.py')
    # 复用 teacher 的配置，改 depth
    cfg = teacher_backbone
    student = copy.deepcopy(cfg)
    # EVAViT 由 config 构建，这里直接用 teacher 的模块构造新实例
    # 更简单：clone teacher 后只保留前 new_depth blocks
    from torch import nn as _nn
    new_bb = copy.deepcopy(cfg)
    # 深拷贝后截断 blocks
    keep = list(new_bb.blocks)[:new_depth]
    new_bb.blocks = _nn.ModuleList(keep)
    # 调整 window_block_indexes / residual_block_indexes（若引用旧索引会越界）
    for attr in ['window_block_indexes', 'residual_block_indexes']:
        if hasattr(new_bb, attr) and getattr(new_bb, attr):
            orig = list(getattr(new_bb, attr))
            filtered = [i for i in orig if i < new_depth]
            setattr(new_bb, attr, filtered)
    return new_bb


class VisionFeatureMimic(nn.Module):
    """teacher(24层) + student(12层) 双 backbone，Feature Mimic 蒸馏。"""

    def __init__(self, teacher_backbone, student_backbone, teacher_layers, student_layers):
        super().__init__()
        self.teacher = teacher_backbone
        self.student = student_backbone
        self.teacher_layers = teacher_layers   # e.g. [0,2,4,...,22]
        self.student_layers = student_layers   # e.g. [0..11]

    def _forward_with_features(self, backbone, layers):
        x = backbone.patch_embed(backbone.img)
        # 简化：直接调用 backbone 内部，但 EVAViT forward 只返回最后特征
        # 这里需要在 block 间捕获特征。我们 clone forward 逻辑。
        return self._run_backbone(backbone, x, layers)

    def _run_backbone(self, bb, x, capture_layers):
        x = x.to(bb.patch_embed.proj.weight.data.dtype)
        x = bb.patch_embed(x)
        if bb.pos_embed is not None:
            from mmcv.models.backbones.eva_vit import get_abs_pos
            x = x + get_abs_pos(bb.pos_embed, bb.pretrain_use_cls_token, (x.shape[1], x.shape[2])).to(x.dtype)
        feats = []
        for i, blk in enumerate(bb.blocks):
            x = blk(x)
            if i in capture_layers:
                feats.append(x)
        x = x.permute(0, 3, 1, 2)
        return x, feats

    def forward(self, img):
        """img: (B, N, 3, H, W) 或 (N,3,H,W)。"""
        if img.dim() == 5:
            B, N, C, H, W = img.shape
            img = img.view(B * N, C, H, W)
        # teacher 前向（冻结，no_grad 可选但保留给 loss）
        t_out, t_feats = self._run_backbone(self.teacher, img, self.teacher_layers)
        s_out, s_feats = self._run_backbone(self.student, img, self.student_layers)
        return (t_out, t_feats), (s_out, s_feats)

    def _forward_teacher(self, img):
        if img.dim() == 5:
            img = img.view(-1, *img.shape[2:])
        t_out, t_feats = self._run_backbone(self.teacher, img, self.teacher_layers)
        return t_out, t_feats

    def _forward_student(self, img):
        if img.dim() == 5:
            img = img.view(-1, *img.shape[2:])
        s_out, s_feats = self._run_backbone(self.student, img, self.student_layers)
        return s_out, s_feats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='adzoo/orion/configs/orion_distill_new_agent_with_mimic.py')
    p.add_argument('--ckpt', default='eval/fused_ckpts/orion_lite_subset8.pth')
    p.add_argument('--out-dir', default='work_dirs/vision_distill_12l')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--train-frames', type=int, default=173)
    p.add_argument('--val-frames', type=int, default=50)
    p.add_argument('--new-depth', type=int, default=12)
    p.add_argument('--img-size', type=int, default=512)
    p.add_argument('--layer-weight', type=float, default=1.0, help='中间层 loss 权重')
    p.add_argument('--final-weight', type=float, default=1.0, help='最终特征 loss 权重')
    p.add_argument('--fp16', action='store_true', default=True)
    p.add_argument('--train-ann', default='distill_data_subset8/b2d_infos_train_subset8.pkl',
                   help='训练数据 ann file（subset8 或全量）')
    p.add_argument('--val-ann', default='distill_data_subset8/b2d_infos_val_subset8.pkl',
                   help='验证数据 ann file')
    p.add_argument('--data-root', default=os.environ.get('B2D_ROOT', ''),
                   help='Bench2Drive dataset root (env B2D_ROOT, or pass --data-root)')
    p.add_argument('--map-file', default=None,
                   help='map file（默认同 val-ann 目录）')
    p.add_argument('--batch-size', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载完整模型（teacher 24 层）
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    for mname in ['img_backbone', 'pts_bbox_head', 'map_head']:
        cfg.model[mname]["flash_attn"] = False
        cfg.model[mname]["with_cp"] = False
        if mname == 'pts_bbox_head':
            for sub in ['transformer', 'motion_transformer_decoder', 'memory_decoder_transformer']:
                if sub in cfg.model[mname]:
                    cfg.model[mname][sub]["flash_attn"] = False
                    cfg.model[mname][sub]["with_cp"] = False
    os.environ.pop('ORION_STUDENT_MODEL_PATH', None)
    cfg.model['student_model_path'] = None
    cfg.model['student_load_strict'] = False
    cfg.model['img_backbone']['img_size'] = args.img_size

    print("Building teacher model (24-layer)...")
    teacher_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(teacher_model, args.ckpt, map_location='cpu')
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_bb = teacher_model.img_backbone

    # 2. 构造 student backbone (12层)
    print(f"Building student backbone (depth={args.new_depth})...")
    student_bb = build_student_backbone(teacher_bb, new_depth=args.new_depth)
    student_bb.train()

    # 3. Feature Mimic 模块
    teacher_layers = list(range(0, 24, 24 // args.new_depth))[:args.new_depth]  # 每2层采1
    student_layers = list(range(args.new_depth))
    # 启用 student 梯度（深拷贝自冻结 teacher）
    for p in student_bb.parameters():
        p.requires_grad = True
    model = VisionFeatureMimic(teacher_bb, student_bb, teacher_layers, student_layers)
    if args.fp16:
        # 不手动 .half()；用 autocast + GradScaler（标准 AMP）
        model = model.to(torch.float32)

    # 4. 数据（bench2drive 在线前向）
    map_root = os.path.join(args.data_root, 'maps')
    map_file = args.map_file or os.path.join(os.path.dirname(os.path.abspath(args.train_ann)), 'b2d_map_infos.pkl')
    cfg.data.test.data_root = args.data_root
    cfg.data.test.ann_file = os.path.abspath(args.train_ann)
    cfg.data.test.map_root = map_root
    cfg.data.test.map_file = map_file
    for step in cfg.data.test.pipeline:
        if step.get('type') == 'ResizeMultiview3D':
            step['img_scale'] = (args.img_size, args.img_size)

    dataset = build_dataset(cfg.data.test)
    print(f"dataset size: {len(dataset)}")
    # 取前 train_frames 训练（不 shuffle 保证确定性）
    indices = list(range(min(args.train_frames, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = build_dataloader(subset, samples_per_gpu=args.batch_size, workers_per_gpu=2,
                              dist=False, shuffle=False)

    # 5. 优化器（只训 student backbone）
    optimizer = torch.optim.AdamW(student_bb.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    device = 'cuda'

    # 6. 训练
    print(f"Training {args.new_depth}-layer student (feature mimic)...")
    model.to(device)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for i, batch in enumerate(loader):
            # batch['img'] 可能是 list(每sample) 或 tensor；统一为 (B*N,3,H,W)
            if isinstance(batch['img'], (list, tuple)):
                imgs = torch.cat([b[0] for b in batch['img']], dim=0)  # (B,6,3,H,W)
            else:
                imgs = batch['img']
            if imgs.dim() == 5:
                B, N, C, H, W = imgs.shape
                img = imgs.view(B * N, C, H, W)
            else:
                img = imgs
            img = img.to(device)
            if args.fp16:
                img = img.float()  # autocast 会自动转 fp16
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                # teacher 前向不保留计算图（节省显存），student 前向保留用于反向
                with torch.no_grad():
                    t_out, t_feats = model._forward_teacher(img)
                s_out, s_feats = model._forward_student(img)
                # 中间层对齐
                layer_loss = 0.0
                for tf, sf in zip(t_feats, s_feats):
                    layer_loss += F.mse_loss(sf.float(), tf.float())
                layer_loss = layer_loss / max(len(t_feats), 1)
                # 最终特征对齐
                final_loss = F.mse_loss(s_out.float(), t_out.float())
                loss = args.layer_weight * layer_loss + args.final_weight * final_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            global_step += 1
            if global_step % 20 == 0:
                print(f"  epoch{epoch} step{global_step}: loss={loss.item():.4f} "
                      f"layer={layer_loss.item():.4f} final={final_loss.item():.4f} "
                      f"({(time.time()-t0)/max(global_step,1):.2f}s/step)", flush=True)

        avg = epoch_loss / max(len(loader), 1)
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Epoch {epoch} avg_loss={avg:.4f} ({(time.time()-t0)/60:.1f}min) "
                  f"peak_gpu={peak_mb:.0f}MB", flush=True)
        else:
            print(f"Epoch {epoch} avg_loss={avg:.4f} ({(time.time()-t0)/60:.1f}min)", flush=True)

        # 保存
        torch.save({
            'student_backbone': student_bb.state_dict(),
            'epoch': epoch,
            'avg_loss': avg,
            'new_depth': args.new_depth,
        }, os.path.join(args.out_dir, f'student_backbone_{args.new_depth}l_epoch{epoch}.pth'))
        print(f"Saved: {args.out_dir}/student_backbone_{args.new_depth}l_epoch{epoch}.pth", flush=True)

    print("DONE")


if __name__ == '__main__':
    main()
