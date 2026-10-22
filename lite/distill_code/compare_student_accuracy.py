#!/usr/bin/env python
"""对比原 student vs 新 student（6L+384 重训练）的轨迹精度。

用 distill_data_6l384/val 的 npz 数据：
  visual_queries (6L+384 特征) → student → ego_feature + ego_fut_preds
  planning_token (原版 7B teacher 参考)

对比指标：
  1. ego_feature cos（vs planning_token）
  2. ego_fut_preds L2 误差（vs ego_fut_trajs GT）
"""
import os, sys, glob, json
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("ORION_LITE_PROJECT_ROOT")
if not REPO:
    raise SystemExit("error: ORION_LITE_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (Lite variant, see ckpts/README.md)")
sys.path.insert(0, os.path.join(REPO, "distill"))
from student_model_with_orion_loss import OrionMimicModel


def load_student(ckpt_path, prefix=""):
    """加载 student 模型权重。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    else:
        sd = ck.get("state_dict", ck)
        # 过滤 student_model 前缀
        sd = {k.replace("student_model.", ""): v for k, v in sd.items() if k.startswith("student_model.")}

    model = OrionMimicModel(
        input_dim=4096, hidden_dim=1024, output_dim=4096,
        num_layers=6, num_heads=16, dropout=0.0,
        with_bound_loss=False, use_col_loss=False,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def main():
    val_dir = os.path.join(REPO, "distill_data_6l384/val")
    files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))[:50]
    print(f"评估 {len(files)} 个 val 样本")

    # 加载两个 student
    orig_ckpt = os.path.join(REPO, "eval/fused_ckpts/orion_lite_subset8.pth")
    new_ckpt = os.path.join(REPO, "distill/results/20260812_173853/checkpoints/best.pt")

    print("加载原 student...")
    orig_model = load_student(orig_ckpt)
    print("加载新 student（6L+384 重训练）...")
    new_model = load_student(new_ckpt)

    # 逐样本评估
    orig_ego_cos = []
    new_ego_cos = []
    orig_traj_l2 = []
    new_traj_l2 = []

    for f in files:
        d = np.load(f, allow_pickle=True)
        vq = torch.from_numpy(d["visual_queries"]).float()  # (1, 529, 4096)
        pt = torch.from_numpy(d["planning_token"]).float()   # (1, 1, 4096)
        ego_trajs = torch.from_numpy(d["ego_fut_trajs"]).float()  # (1, cmds, 6, 2)
        ego_cmd = torch.from_numpy(d["ego_fut_cmd"]).float()
        lane_scores = torch.from_numpy(d["lane_scores"]).float()
        lane_preds = torch.from_numpy(d["lane_preds"]).float()

        with torch.no_grad():
            # 原 student（return_loss=False 纯推理）
            try:
                out_orig = orig_model(vq, return_loss=False)
                ego_feat_orig = out_orig.get("ego_feature", None)
                ego_pred_orig = out_orig.get("ego_fut_preds", None)
            except Exception as e:
                print(f"  orig err: {e}")
                ego_feat_orig = None
                ego_pred_orig = None

            # 新 student
            try:
                out_new = new_model(vq, return_loss=False)
                ego_feat_new = out_new.get("ego_feature", None)
                ego_pred_new = out_new.get("ego_fut_preds", None)
            except Exception as e:
                print(f"  new err: {e}")
                ego_feat_new = None
                ego_pred_new = None

        # ego_feature cos（vs teacher planning_token）
        if ego_feat_orig is not None and pt is not None:
            a = ego_feat_orig.reshape(-1)
            b = pt.reshape(-1)
            orig_ego_cos.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        if ego_feat_new is not None and pt is not None:
            a = ego_feat_new.reshape(-1)
            b = pt.reshape(-1)
            new_ego_cos.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

        # ego_fut_preds L2（vs GT ego_trajs）
        # student 输出 (1, 6_mode, 6_ts, 2)，GT 是 (1, 1, 6, 2) → squeeze → (6, 2)
        if ego_pred_orig is not None and ego_trajs is not None:
            pred = ego_pred_orig[0]  # (6_mode, 6_ts, 2)
            gt = ego_trajs.squeeze().reshape(-1, 2)  # (6, 2)
            if gt.shape[0] != 6: gt = ego_trajs.reshape(-1, 2)[:6]
            best_l2 = min(torch.norm(pred[m].reshape(-1) - gt.reshape(-1)).item() for m in range(pred.shape[0]))
            orig_traj_l2.append(best_l2)
        if ego_pred_new is not None and ego_trajs is not None:
            pred = ego_pred_new[0]
            gt = ego_trajs.squeeze().reshape(-1, 2)
            if gt.shape[0] != 6: gt = ego_trajs.reshape(-1, 2)[:6]
            best_l2 = min(torch.norm(pred[m].reshape(-1) - gt.reshape(-1)).item() for m in range(pred.shape[0]))
            new_traj_l2.append(best_l2)

    print(f"\n=== 精度对比（{len(files)} 样本）===")
    if orig_ego_cos and new_ego_cos:
        print(f"ego_feature cos (vs teacher):")
        print(f"  原 student: {np.mean(orig_ego_cos):.4f}")
        print(f"  新 student: {np.mean(new_ego_cos):.4f}")
        print(f"  改善: {np.mean(new_ego_cos) - np.mean(orig_ego_cos):+.4f}")
    if orig_traj_l2 and new_traj_l2:
        print(f"ego_fut_preds L2 (vs GT, 越小越好):")
        print(f"  原 student: {np.mean(orig_traj_l2):.4f}")
        print(f"  新 student: {np.mean(new_traj_l2):.4f}")
        print(f"  改善: {np.mean(new_traj_l2) - np.mean(orig_traj_l2):+.4f} ({(np.mean(new_traj_l2)/max(np.mean(orig_traj_l2),1e-9)-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
