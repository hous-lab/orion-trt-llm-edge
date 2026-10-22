"""
Training script for the VLA student model (OrionVLAModel with Qwen3.5-0.8B).

Usage (single GPU):
    python train_student_vla.py \
        --qwen-model-path /path/to/Qwen3.5-0.8B-Base \
        --train-dir distill_data/train \
        --val-dir distill_data/val \
        --orion-ckpt /path/to/orion_teacher.pth \
        --device cuda:0

Usage (multi-GPU DDP):
    torchrun --nproc_per_node=4 train_student_vla.py \
        --qwen-model-path /path/to/Qwen3.5-0.8B-Base \
        --train-dir distill_data/train \
        --val-dir distill_data/val \
        --orion-ckpt /path/to/orion_teacher.pth
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
import json
import logging
import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

_DISTILL_DIR = os.path.dirname(os.path.abspath(__file__))
if _DISTILL_DIR not in sys.path:
    sys.path.insert(0, _DISTILL_DIR)


class NpzDistillDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise RuntimeError(f"No npz files found in {data_dir}")
        self.files = all_files
        self._corrupted_count = 0
        print(f"Found {len(self.files)} npz files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            with np.load(path, allow_pickle=True) as d:
                vision_embeded = d["visual_queries"]
                planning_token = d["planning_token"]
                ego_fut_trajs = d["ego_fut_trajs"]
                ego_fut_masks = d["ego_fut_masks"]
                ego_fut_cmd = d["ego_fut_cmd"]
                lane_scores = d["lane_scores"]
                lane_preds = d["lane_preds"]

                agent_preds_np = d["agent_preds"] if "agent_preds" in d.files else None
                agent_fut_preds_np = d["agent_fut_preds"] if "agent_fut_preds" in d.files else None
                agent_score_preds_np = d["agent_score_preds"] if "agent_score_preds" in d.files else None
                agent_fut_cls_preds_np = d["agent_fut_cls_preds"] if "agent_fut_cls_preds" in d.files else None
        except Exception as e:
            self._corrupted_count += 1
            if self._corrupted_count <= 5:
                print(f"[WARN] Corrupted npz file: {os.path.basename(path)}, error: {e}")
            next_idx = (idx + 1) % len(self.files)
            return self.__getitem__(next_idx)

        vision_embeded = torch.from_numpy(vision_embeded).float().squeeze(0)
        planning_token = torch.from_numpy(planning_token).float().squeeze(0)
        ego_fut_trajs = torch.from_numpy(ego_fut_trajs).float().squeeze(0)
        ego_fut_masks = torch.from_numpy(ego_fut_masks).float()
        ego_fut_cmd = torch.from_numpy(ego_fut_cmd).float()
        lane_scores = torch.from_numpy(lane_scores).float()
        lane_preds = torch.from_numpy(lane_preds).float()

        sample = {
            "vision_embeded": vision_embeded,
            "target_ego_feature": planning_token[0],
            "ego_fut_trajs": ego_fut_trajs,
            "ego_fut_masks": ego_fut_masks,
            "ego_fut_cmd": ego_fut_cmd,
            "lane_scores": lane_scores,
            "lane_preds": lane_preds,
        }

        for key, np_arr in [
            ("agent_preds", agent_preds_np),
            ("agent_fut_preds", agent_fut_preds_np),
            ("agent_score_preds", agent_score_preds_np),
            ("agent_fut_cls_preds", agent_fut_cls_preds_np),
        ]:
            if np_arr is not None:
                t = torch.from_numpy(np_arr).float()
                if t.dim() > 0 and t.shape[0] == 1:
                    t = t.squeeze(0)
                sample[key] = t

        return sample


def collate_fn(batch):
    return torch.utils.data._utils.collate.default_collate(batch)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train VLA student model (Qwen3.5-0.8B) with distillation from Orion.",
    )

    # ── Data ────────────────────────────────────────────────────────────
    parser.add_argument("--train-dir", type=str, default="distill_data/train")
    parser.add_argument("--val-dir", type=str, default="distill_data/val")

    # ── Model ───────────────────────────────────────────────────────────
    parser.add_argument("--qwen-model-path", type=str, required=True,
                        help="Path to Qwen3.5-0.8B-Base weights")
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--output-dim", type=int, default=4096)
    parser.add_argument("--freeze-llm", action="store_true", default=True)
    parser.add_argument("--no-freeze-llm", action="store_false", dest="freeze_llm")
    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-col-loss", action="store_true", default=True)
    parser.add_argument("--mimic-loss", type=str, default="l1",
                        choices=["l1", "l2", "kl", "huber"])

    # ── Training ────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vision-noise-std", type=float, default=0.0)
    parser.add_argument("--mimic-weight-decay-epochs", type=int, default=0)
    parser.add_argument("--use-feature-mimic-loss", action="store_true", default=True)
    parser.add_argument("--no-feature-mimic-loss", action="store_false",
                        dest="use_feature_mimic_loss")
    parser.add_argument("--use-bf16", action="store_true", default=True,
                        help="Use BFloat16 mixed precision for training")

    # ── System ──────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g. '2,3'). "
                             "Sets CUDA_VISIBLE_DEVICES before any CUDA init.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dist-backend", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=0)

    # ── Orion checkpoint for VAE weights ────────────────────────────────
    parser.add_argument("--orion-ckpt", type=str, default=None,
                        help="Path to original Orion checkpoint to load VAE weights")

    return parser.parse_args()


def set_seed(seed):
    if seed <= 0:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed_run():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed(args):
    if not is_distributed_run():
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        return
    args.distributed = True
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = getattr(args, "dist_backend", None) or (
        "nccl" if torch.cuda.is_available() else "gloo"
    )
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(minutes=30),
    )


def setup_device(args):
    if args.distributed and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        return torch.device("cuda", args.local_rank)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(args.device if torch.cuda.is_available() else "cpu")


def setup_logger(log_path, rank=0):
    logger = logging.getLogger(f"vla_train.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = "%(asctime)s - %(levelname)s - %(message)s" if rank == 0 else \
          f"%(asctime)s - [rank {rank}] %(message)s"
    formatter = logging.Formatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path, model, optimizer, epoch, global_step, best_val_loss, args,
                    val_loss=None):
    save_dict = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    if args.use_lora:
        save_dict["lora_state_dict"] = {
            k: v for k, v in unwrap_model(model).qwen_model.named_parameters()
            if v.requires_grad
        }
    torch.save(save_dict, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"]
    unwrap_model(model).load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def reduce_losses(losses, world_size, device):
    if world_size <= 1:
        return losses
    reduced = {}
    for k, v in losses.items():
        t = torch.tensor([v], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[k] = (t / world_size).item()
    return reduced


def train_one(args, train_loader, val_loader, train_sampler, device, run_dir):
    from student_vla_model import OrionVLAModel

    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    run_args = dict(vars(args))
    if args.rank == 0:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(run_args, f, indent=2, sort_keys=True)

    logger = setup_logger(os.path.join(log_dir, "train.log"), rank=args.rank)
    if args.rank == 0:
        logger.info(f"Config: {json.dumps(run_args, indent=2, default=str)}")

    model = OrionVLAModel(
        qwen_model_path=args.qwen_model_path,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        freeze_llm=args.freeze_llm,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_col_loss=args.use_col_loss,
        mimic_loss_type=args.mimic_loss,
    ).to(device)

    if args.orion_ckpt:
        if args.rank == 0:
            logger.info(f"Loading Orion VAE weights from {args.orion_ckpt}")
        model.load_orion_vae_weights(args.orion_ckpt, device=str(device))

    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        logger.info(f"Resumed from {args.resume}, epoch={start_epoch}")

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_bf16)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=args.rank != 0,
        )
        epoch_losses = {}

        for step, batch in enumerate(train_bar):
            batch_gpu = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            agent_outs = None
            if "agent_preds" in batch:
                agent_outs = {
                    k: batch_gpu[k]
                    for k in ("agent_preds", "agent_fut_preds",
                              "agent_score_preds", "agent_fut_cls_preds")
                    if k in batch_gpu
                }

            lane_preds = batch_gpu["lane_preds"]
            lane_scores = batch_gpu["lane_scores"]
            if lane_preds.dim() == 5 and lane_preds.shape[1] == 1:
                lane_preds = lane_preds.squeeze(1)
            if lane_scores.dim() == 4 and lane_scores.shape[1] == 1:
                lane_scores = lane_scores.squeeze(1)

            vision_in = batch_gpu["vision_embeded"]
            if getattr(args, "vision_noise_std", 0) > 0:
                vision_in = vision_in + torch.randn_like(vision_in) * args.vision_noise_std

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.use_bf16):
                outputs = model(
                    vision_embeded=vision_in,
                    ego_fut_trajs=batch_gpu["ego_fut_trajs"],
                    ego_fut_masks=batch_gpu["ego_fut_masks"],
                    ego_fut_cmd=batch_gpu["ego_fut_cmd"],
                    lane_preds=lane_preds,
                    lane_scores=lane_scores,
                    agent_outs=agent_outs,
                    target_ego_feature=batch_gpu["target_ego_feature"],
                    return_loss=True,
                )

            losses = outputs["losses"]
            mimic_weight = 1.0
            if getattr(args, "mimic_weight_decay_epochs", 0) > 0 and \
               "loss_feature_mimic" in losses:
                mimic_weight = max(
                    0.0, 1.0 - (epoch + 1) / args.mimic_weight_decay_epochs
                )

            total_loss = 0.0
            for k, v in losses.items():
                if k == "loss_feature_mimic":
                    total_loss += mimic_weight * v if args.use_feature_mimic_loss else 0.0
                else:
                    total_loss += v

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            loss_scalars = {k: v.item() for k, v in losses.items()}
            loss_scalars["total"] = total_loss.item()
            for k, v in loss_scalars.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            if args.rank == 0:
                postfix = {"loss": f"{total_loss.item():.4f}"}
                for k in ["loss_feature_mimic", "loss_plan_reg",
                           "loss_plan_bound", "loss_plan_col", "loss_vae_gen"]:
                    if k in losses:
                        short_k = k.replace("loss_", "").replace("feature_", "").replace("plan_", "")
                        postfix[short_k] = f"{losses[k].item():.4f}"
                train_bar.set_postfix(postfix)

        avg_train_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        if args.distributed:
            avg_train_losses = reduce_losses(avg_train_losses, args.world_size, device)
            dist.barrier()

        if args.rank == 0:
            logger.info(f"Epoch {epoch + 1} Train Loss: {avg_train_losses['total']:.4f}")
            logger.info(f"Train Details: {avg_train_losses}")

        # ── Validation ──────────────────────────────────────────────────
        if args.rank == 0 and val_loader:
            model.eval()
            val_losses = {}
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    batch_gpu = {
                        k: v.to(device)
                        for k, v in batch.items() if isinstance(v, torch.Tensor)
                    }
                    agent_outs = None
                    if "agent_preds" in batch:
                        agent_outs = {
                            k: batch_gpu[k]
                            for k in ("agent_preds", "agent_fut_preds",
                                      "agent_score_preds", "agent_fut_cls_preds")
                            if k in batch_gpu
                        }
                    lane_preds = batch_gpu["lane_preds"]
                    lane_scores = batch_gpu["lane_scores"]
                    if lane_preds.dim() == 5 and lane_preds.shape[1] == 1:
                        lane_preds = lane_preds.squeeze(1)
                    if lane_scores.dim() == 4 and lane_scores.shape[1] == 1:
                        lane_scores = lane_scores.squeeze(1)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=args.use_bf16):
                        outputs = model(
                            vision_embeded=batch_gpu["vision_embeded"],
                            ego_fut_trajs=batch_gpu["ego_fut_trajs"],
                            ego_fut_masks=batch_gpu["ego_fut_masks"],
                            ego_fut_cmd=batch_gpu["ego_fut_cmd"],
                            lane_preds=lane_preds,
                            lane_scores=lane_scores,
                            agent_outs=agent_outs,
                            target_ego_feature=batch_gpu["target_ego_feature"],
                            return_loss=True,
                        )

                    losses = outputs["losses"]
                    total_loss = (
                        sum(losses.values())
                        if args.use_feature_mimic_loss
                        else sum(v for k, v in losses.items()
                                 if k != "loss_feature_mimic")
                    )
                    loss_scalars = {k: v.item() for k, v in losses.items()}
                    loss_scalars["total"] = total_loss.item()
                    for k, v in loss_scalars.items():
                        val_losses[k] = val_losses.get(k, 0.0) + v

            avg_val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
            logger.info(f"Epoch {epoch + 1} Val Loss: {avg_val_losses['total']:.4f}")
            logger.info(f"Val Details: {avg_val_losses}")

            if avg_val_losses["total"] < best_val_loss:
                best_val_loss = avg_val_losses["total"]
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    model, optimizer, epoch + 1, global_step, best_val_loss, args,
                    val_loss=avg_val_losses["total"],
                )
                logger.info("Saved best checkpoint")

            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                model, optimizer, epoch + 1, global_step, best_val_loss, args,
            )

            if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt"),
                    model, optimizer, epoch + 1, global_step, best_val_loss, args,
                )

        if args.distributed:
            dist.barrier()

    if args.rank == 0:
        logger.info("Training completed!")
    if args.distributed:
        dist.barrier()


def main():
    # Parse --gpu-ids BEFORE any CUDA init to set CUDA_VISIBLE_DEVICES early
    _gpu_ids = None
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu-ids" and i + 1 < len(sys.argv):
            _gpu_ids = sys.argv[i + 1]
            break
    if _gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_ids
        print(f"[GPU] Set CUDA_VISIBLE_DEVICES={_gpu_ids}")

    args = parse_args()
    init_distributed(args)
    set_seed(args.seed + (args.rank if args.seed > 0 else 0))
    device = setup_device(args)

    train_ds = NpzDistillDataset(args.train_dir)
    val_ds = NpzDistillDataset(args.val_dir)
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
    ) if not args.distributed or args.rank == 0 else None

    script_dir = os.path.dirname(__file__)
    out_dir = args.out_dir or os.path.join(script_dir, "results_vla")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = os.path.join(out_dir, run_name)
    os.makedirs(base_run_dir, exist_ok=True)

    train_one(args, train_loader, val_loader, train_sampler, device, base_run_dir)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
