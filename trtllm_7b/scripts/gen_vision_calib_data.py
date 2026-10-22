#!/usr/bin/env python
"""生成 Vision engine INT8 校准数据（.npz 格式）。

从 bench2drive 真实数据提取 vision engine 的输入 binding，保存为 .npz 供
gen_calib.py 的 MyEntropyCalibrator 使用。

用法: python gen_vision_calib_data.py [--n-frames 32]
产出: calib_data/*.npz
"""
import os, sys, argparse
import numpy as np
import torch

ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not ROOT:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "calib_data"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from mmcv import Config
    from mmcv.datasets import build_dataset, build_dataloader
    import tensorrt as trt
    from adzoo.orion.test_trt import InferTrt

    cfg = Config.fromfile("adzoo/orion/configs/orion_stage3_fp16.py")
    cfg.data.test.test_mode = True
    cfg.data.test.data_root = os.path.join(ROOT, "data/bench2drive")
    cfg.data.test.ann_file = os.path.join(ROOT, "data/infos-tiny/b2d_infos_val.pkl")
    cfg.data.test.map_root = os.path.join(ROOT, "data/bench2drive/maps")
    cfg.data.test.map_file = os.path.join(ROOT, "data/infos-tiny/b2d_map_infos.pkl")
    dataset = build_dataset(cfg.data.test)
    dl = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    # 加载 vision engine 获取 binding 结构
    logger = trt.Logger(trt.Logger.WARNING)
    engine = InferTrt(logger, qa_save_path=None, LLM_engine=None, torch_ref_model=None, ego_engine_pth=None)
    vision_engine = os.environ.get("ORION_VISION_ENGINE", "work_dirs/vision_mixed_precision.engine")
    engine.read(vision_engine)
    engine.eval()

    # 收集输入 binding 名称
    input_names = []
    for i in range(engine.engine.num_io_tensors):
        name = engine.engine.get_tensor_name(i)
        if engine.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
    print(f"Vision engine 输入 binding: {len(input_names)} 个")
    print(input_names)

    # 遍历数据，提取每帧输入
    dtype = torch.float32
    device_id = 0
    saved = 0
    for fi, batch in enumerate(dl):
        if saved >= args.n_frames:
            break
        # 从 test_trt.py 的逻辑提取 binding 数据
        try:
            img = batch["img"][0].to(f"cuda:{device_id}", dtype=dtype).contiguous()
            lidar2img = batch["lidar2img"][0].to(f"cuda:{device_id}", dtype=dtype).inverse()
            ego_pose = batch["ego_pose"][0].to(f"cuda:{device_id}", dtype=dtype).contiguous()
            ego_pose_inv = batch["ego_pose_inv"][0].to(f"cuda:{device_id}", dtype=dtype).contiguous()
            command = batch["command"][0].to(f"cuda:{device_id}", dtype=dtype).contiguous()
            can_bus = batch["can_bus"][0].to(f"cuda:{device_id}", dtype=dtype).contiguous()
            timestamp = torch.zeros([1]).to(f"cuda:{device_id}", dtype=dtype).contiguous()
            is_first = torch.ones([1]).to(f"cuda:{device_id}", dtype=dtype).contiguous()
            input_ids = batch["input_ids"][0][0][0].unsqueeze(0)
        except Exception as e:
            print(f"  帧 {fi} 数据提取失败: {e}")
            continue

        # 填充 engine binding（复用 test_trt.py 逻辑）
        bindings_data = {}
        for k, v in [("img", img), ("img2lidars", lidar2img), ("command", command),
                     ("can_bus", can_bus), ("is_first_frame", is_first),
                     ("ego_pose", ego_pose), ("ego_pose_inv", ego_pose_inv),
                     ("timestamp", timestamp)]:
            if k in input_names:
                bindings_data[k] = v.cpu().numpy()

        # memory binding 初始化为 0（第一帧）
        for name in input_names:
            if name not in bindings_data and "_in" in name:
                shape = tuple(engine.engine.get_tensor_shape(name))
                bindings_data[name] = np.zeros(shape, dtype=np.float32)

        # 保存 .npz
        out_path = os.path.join(args.out_dir, f"calib_{saved:04d}.npz")
        np.savez(out_path, **bindings_data)
        saved += 1
        if saved % 8 == 0:
            print(f"  已保存 {saved}/{args.n_frames} 帧")

    print(f"\n完成：保存 {saved} 帧校准数据到 {args.out_dir}")


if __name__ == "__main__":
    main()
