#!/usr/bin/env python
"""Vision engine 延迟测量（CUDA events，30 帧均值）。

用法: python profile_vision.py <engine_path>
对比 mixed_precision 与 int8 的 Vision 延迟。
"""
import os, sys, time, json
import numpy as np
import torch

PROJECT_ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from mmcv import Config
from mmcv.datasets import build_dataset, build_dataloader
import tensorrt as trt
from adzoo.orion.test_trt import InferTrt

def main():
    engine_pth = sys.argv[1] if len(sys.argv) > 1 else "work_dirs/vision_int8.engine"
    n_frames_mean = 30

    cfg = Config.fromfile("adzoo/orion/configs/orion_stage3_fp16.py")
    cfg.data.test.test_mode = True
    cfg.data.test.data_root = os.path.join(PROJECT_ROOT, "data/bench2drive")
    cfg.data.test.ann_file = os.path.join(PROJECT_ROOT, "data/infos-tiny/b2d_infos_val.pkl")
    cfg.data.test.map_root = os.path.join(PROJECT_ROOT, "data/bench2drive/maps")
    cfg.data.test.map_file = os.path.join(PROJECT_ROOT, "data/infos-tiny/b2d_map_infos.pkl")
    dataset = build_dataset(cfg.data.test)
    dl = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    logger = trt.Logger(trt.Logger.WARNING)
    engine = InferTrt(logger, qa_save_path=None, LLM_engine=None, torch_ref_model=None, ego_engine_pth=None)
    engine.read(engine_pth)
    engine.eval()

    print(f"Engine: {engine_pth}")

    # warmup
    for fi, batch in enumerate(dl):
        if fi >= n_frames_mean:
            break
        img = batch["img"][0]
        # 简单触发一次 inference（warmup）
        break

    # CUDA events 计时
    times = []
    for fi, batch in enumerate(dl):
        if fi >= n_frames_mean:
            break
        img = batch["img"][0].to("cuda:0", dtype=torch.float32).contiguous()
        lidar2img = batch["lidar2img"][0].to("cuda:0", dtype=torch.float32).inverse()
        ego_pose = batch["ego_pose"][0].to("cuda:0", dtype=torch.float32).contiguous()
        ego_pose_inv = batch["ego_pose_inv"][0].to("cuda:0", dtype=torch.float32).contiguous()
        command = batch["command"][0].to("cuda:0", dtype=torch.float32).contiguous()
        can_bus = batch["can_bus"][0].to("cuda:0", dtype=torch.float32).contiguous()
        timestamp = torch.zeros([1]).to("cuda:0", dtype=torch.float32).contiguous()
        is_first = torch.ones([1]).to("cuda:0", dtype=torch.float32).contiguous()

        for k, v in [("img", img), ("img2lidars", lidar2img), ("command", command),
                     ("can_bus", can_bus), ("is_first_frame", is_first),
                     ("ego_pose", ego_pose), ("ego_pose_inv", ego_pose_inv),
                     ("timestamp", timestamp)]:
            engine.bindings[k].copy_(v)
        for key in engine.bindings:
            if "_in" in key and key.replace("_in", "_out") in engine.bindings:
                engine.bindings[key].copy_(engine.bindings[key.replace("_in", "_out")])

        engine.cuda_ctx.push()
        for i in range(len(engine.names)):
            engine.context.set_tensor_address(engine.names[i], engine.bindings[str(engine.names[i])].data_ptr())
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        engine.context.execute_async_v3(stream_handle=engine.stream.handle)
        end.record()
        engine.stream.synchronize()
        engine.cuda_ctx.pop()
        times.append(start.elapsed_time(end))

    times = times[3:]  # 丢弃前 3 帧 warmup
    if times:
        arr = np.array(times)
        print(f"\n=== Vision 延迟（{len(times)} 帧均值）===")
        print(f"  mean={arr.mean():.2f}ms  p50={np.percentile(arr,50):.2f}  "
              f"p95={np.percentile(arr,95):.2f}  min={arr.min():.2f}  max={arr.max():.2f}")

if __name__ == "__main__":
    main()