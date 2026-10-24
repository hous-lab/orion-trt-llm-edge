#!/usr/bin/env python
"""端到端真实延迟测量：从数据输入到轨迹输出的完整单帧墙钟时间。

与模块 profile（各模块独立计时之和）不同，这里测真实串行流水线总耗时，
反映实际部署延迟（含模块间传递、同步点）。
"""
import os, sys, time
import numpy as np
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
PROJECT_ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not PROJECT_ROOT:
    sys.exit("error: ORION_PROJECT_ROOT is not set - point it at the patched training repo (see trtllm_7b/env_template.sh)")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from mmcv import Config
from mmcv.datasets import build_dataset, build_dataloader
import tensorrt as trt
from adzoo.orion.test_trt import InferTrt, InferTrtLLM

def main():
    engine_pth = os.environ.get("VISION_ENGINE", "ckpts/onnxs_hous/orion_stage3_vision_mixed_precision.engine")
    llm_pth = os.environ.get("LLM_ENGINE", "ckpts/onnxs/llm_safetensor/x86_1gpu_w8abf16_patched")

    cfg = Config.fromfile("adzoo/orion/configs/orion_stage3_fp16.py")
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    dl = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    logger = trt.Logger(trt.Logger.WARNING)
    engine = InferTrt(logger, qa_save_path=None, LLM_engine=None, torch_ref_model=None, ego_engine_pth=None)
    engine.read(engine_pth)
    engine.eval()
    llm = InferTrtLLM(llm_engine_pth=llm_pth, tokenizer_pth="ckpts/hf_models/pretrain_qformer/")

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "lightweight_ego", os.path.join(PROJECT_ROOT, "adzoo", "orion", "lightweight_ego.py"))
    _ego = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ego)
    _ego_dev = "cuda:0" if os.environ.get("EGO_ON_GPU", "0") == "1" else "cpu"
    torch_ego = _ego.load_lightweight_ego("ckpts/Orion.pth", device=_ego_dev)
    torch_ego.eval()
    engine.torch_ego_model = torch_ego

    # 逐帧端到端计时（含全部模块 + 同步）
    lat = []
    prev_scene = None
    start_ts = None
    for fi, batch in enumerate(dl):
        if fi >= 33:
            break
        scene_token = batch["img_metas"][0][0].get("scene_token")
        ts_raw = batch["timestamp"][0].item()
        if scene_token != prev_scene:
            prev_scene = scene_token; start_ts = ts_raw
            is_first = torch.ones([1])
        else:
            is_first = torch.zeros([1])
        ts = torch.tensor([ts_raw - start_ts])

        t0 = time.time()
        # ---- 模拟 test_trt forward 全流程 ----
        device = "cuda:0"; dtype = torch.float32
        img = batch["img"][0].to(device, dtype=dtype).contiguous()
        lidar2img = batch["lidar2img"][0].to(device, dtype=torch.float32).inverse()
        ego_pose = batch["ego_pose"][0].to(device, dtype=dtype).contiguous()
        ego_pose_inv = batch["ego_pose_inv"][0].to(device, dtype=dtype).contiguous()
        command = batch["command"][0].to(device, dtype=dtype).contiguous()
        can_bus = batch["can_bus"][0].to(device, dtype=dtype).contiguous()
        timestamp = ts.to(device, dtype=dtype).contiguous()
        input_ids = batch["input_ids"][0][0][0].unsqueeze(0)

        # Vision 输入
        for k, v in [("img", img), ("img2lidars", lidar2img), ("command", command),
                     ("can_bus", can_bus), ("is_first_frame", is_first.to(device, dtype=dtype)),
                     ("ego_pose", ego_pose), ("ego_pose_inv", ego_pose_inv), ("timestamp", timestamp)]:
            engine.bindings[k].copy_(v)
        for key in ["memory_embedding_bbox", "memory_reference_point_bbox", "memory_timestamp_bbox",
                    "memory_egopose_bbox", "memory_canbus_bbox", "sample_time_bbox",
                    "memory_scene_query_bbox", "scene_memory_timestamp_bbox", "memory_timestamp_map",
                    "sample_time_map", "memory_egopose_map", "memory_embedding_map", "memory_reference_point_map"]:
            in_k, out_k = f"{key}_in", f"{key}_out"
            if in_k in engine.bindings and out_k in engine.bindings:
                engine.bindings[in_k].copy_(engine.bindings[out_k])

        # Vision 推理
        engine.cuda_ctx.push()
        for i in range(len(engine.names)):
            engine.context.set_tensor_address(engine.names[i], engine.bindings[str(engine.names[i])].data_ptr())
        engine.context.execute_async_v3(stream_handle=engine.stream.handle)
        engine.stream.synchronize()
        engine.cuda_ctx.pop()
        vision_embeded = engine.bindings["vision_embeded"].clone()

        # canbus
        if engine.canbus_fixer is not None:
            scene_same = torch.tensor([1.0]) if is_first.item() == 0 else torch.tensor([0.0])
            if getattr(engine, "_canbus_scene", None) != scene_token:
                engine.canbus_fixer.reset(); engine._canbus_scene = scene_token
            _fd = next(engine.canbus_fixer.can_bus_embed.parameters()).device
            _td = lambda t: t.to(_fd) if _fd.type == 'cuda' else t.cpu()
            with torch.no_grad():
                cb = engine.canbus_fixer.step(_td(is_first.float()), _td(scene_same),
                                              _td(ego_pose_inv.float()), _td(ego_pose.float()),
                                              _td(command.float()), _td(can_bus.float()))
            vision_embeded[0, 272, :] = cb[0].to(vision_embeded.device)

        # LLM
        inputs_embeds, new_ids = llm.prepare_inputs_labels_for_multimodal_(input_ids, vision_embeded)
        output_ids, ego_feature, _ = llm.generate_test(inputs_embeds, new_ids)

        # Ego
        with torch.no_grad():
            _ed = next(torch_ego.parameters()).device
            if _ed.type == 'cuda':
                ef = ego_feature.to(torch.float32).to(_ed)
                cs = ef.unsqueeze(1)
                sample, _ = torch_ego.distribution_forward(cs)
                hs = ef.unsqueeze(1)
                shs, _ = torch_ego.future_states_predict(1, sample, hs, cs)
                eqh = shs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                stacked = torch.cat([eqh[i] for i in range(6)], dim=0)
                torch_ego.ego_fut_decoder(stacked)
            else:
                ef = ego_feature.to(torch.float32).cpu()
                cs = ef.unsqueeze(1)
                sample, _ = torch_ego.distribution_forward(cs)
                hs = ef.unsqueeze(1)
                shs, _ = torch_ego.future_states_predict(1, sample, hs, cs)
                eqh = shs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                for i in range(6):
                    torch_ego.ego_fut_decoder(eqh[i])

        torch.cuda.synchronize()
        lat.append((time.time() - t0) * 1000)
        if fi >= 3 and fi % 5 == 0:
            print(f"  frame {fi}: {lat[-1]:.1f}ms", flush=True)

    lat = np.array(lat[3:])  # 跳过 warmup
    print("\n" + "=" * 60)
    print(f"  端到端真实延迟 (skip warmup):")
    print(f"    mean={lat.mean():.1f}ms  p50={np.percentile(lat,50):.1f}  p95={np.percentile(lat,95):.1f}  max={lat.max():.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
