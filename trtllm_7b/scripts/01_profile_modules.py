#!/usr/bin/env python
"""Orion 端到端模块级延迟 profile。

对每个模块用 CUDA events 精确计时（GPU 时间，非 wall clock）：
  [T1] 数据预处理（H2D + 格式化）
  [T2] Vision engine 推理
  [T3] canbus_fixer（PyTorch CPU）
  [T4] LLM inputs_embeds 构建（prepare_inputs_labels_for_multimodal_）
  [T5] LLM engine 推理（generate_test，含 full_hidden_states）
  [T6] ego 预测头（lightweight_ego，CPU）
  [T7] 输出后处理（get_bbox/get_lane）

用法:
  先 export ORION_PROJECT_ROOT 指向打补丁后的训练 repo（见 ../env_template.sh）
  python 01_profile_modules.py [--frames N] [--engine PATH] [--llm PATH]
"""
import os, sys, time, json, argparse
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=30, help="profile 帧数")
    p.add_argument("--engine", default="ckpts/onnxs_hous/orion_stage3_vision_mixed_precision.engine")
    p.add_argument("--llm", default="ckpts/onnxs/llm_safetensor/x86_1gpu_w8abf16_patched")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_modules.json"))
    return p.parse_args()

def main():
    args = parse_args()

    cfg = Config.fromfile("adzoo/orion/configs/orion_stage3_fp16.py")
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    dl = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    logger = trt.Logger(trt.Logger.WARNING)
    engine = InferTrt(logger, qa_save_path=None, LLM_engine=None, torch_ref_model=None, ego_engine_pth=None)
    engine.read(args.engine)
    engine.eval()

    llm = InferTrtLLM(llm_engine_pth=args.llm, tokenizer_pth="ckpts/hf_models/pretrain_qformer/")

    # 加载 lightweight_ego（默认 CPU；EGO_ON_GPU=1 时 GPU + 批处理，与 test_trt.py 一致）
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "lightweight_ego",
        os.path.join(PROJECT_ROOT, "adzoo", "orion", "lightweight_ego.py"))
    _ego = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ego)
    _ego_dev = "cuda:0" if os.environ.get("EGO_ON_GPU", "0") == "1" else "cpu"
    torch_ego = _ego.load_lightweight_ego("ckpts/Orion.pth", device=_ego_dev)
    torch_ego.eval()

    device = "cuda:0"
    N = args.frames
    # 先跳过前几帧建立 memory（canbus/场景状态）
    timings = {k: [] for k in ["T1_data", "T2_vision", "T3_canbus", "T4_llm_inputs",
                               "T5_llm", "T6_ego", "T7_postproc", "T_total"]}

    prev_scene = None
    start_ts = None
    for fi, batch in enumerate(dl):
        if fi >= N + 3:  # 前 3 帧 warmup
            break
        scene_token = batch["img_metas"][0][0].get("scene_token", "unknown")
        ts_raw = batch["timestamp"][0].item()
        if scene_token != prev_scene:
            prev_scene = scene_token
            start_ts = ts_raw
            is_first = torch.ones([1])
        else:
            is_first = torch.zeros([1])
        ts = torch.tensor([ts_raw - start_ts])

        img = batch["img"][0].to("cuda:0", dtype=torch.float32).contiguous()
        lidar2img = batch["lidar2img"][0].to("cuda:0", dtype=torch.float32).inverse()
        ego_pose = batch["ego_pose"][0].to("cuda:0", dtype=torch.float32).contiguous()
        ego_pose_inv = batch["ego_pose_inv"][0].to("cuda:0", dtype=torch.float32).contiguous()
        command = batch["command"][0].to("cuda:0", dtype=torch.float32).contiguous()
        can_bus = batch["can_bus"][0].to("cuda:0", dtype=torch.float32).contiguous()
        timestamp = ts.to("cuda:0", dtype=torch.float32).contiguous()
        input_ids = batch["input_ids"][0][0][0].unsqueeze(0)

        if fi < 3:
            continue  # warmup

        events = {}

        # ===== T1 数据预处理（H2D）=====
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for k, v in [("img", img), ("img2lidars", lidar2img), ("command", command),
                     ("can_bus", can_bus), ("is_first_frame", is_first.to("cuda:0", dtype=torch.float32)),
                     ("ego_pose", ego_pose), ("ego_pose_inv", ego_pose_inv), ("timestamp", timestamp)]:
            engine.bindings[k].copy_(v)
        for key in ["memory_embedding_bbox", "memory_reference_point_bbox", "memory_timestamp_bbox",
                    "memory_egopose_bbox", "memory_canbus_bbox", "sample_time_bbox",
                    "memory_scene_query_bbox", "scene_memory_timestamp_bbox", "memory_timestamp_map",
                    "sample_time_map", "memory_egopose_map", "memory_embedding_map", "memory_reference_point_map"]:
            in_k, out_k = f"{key}_in", f"{key}_out"
            if in_k in engine.bindings and out_k in engine.bindings:
                engine.bindings[in_k].copy_(engine.bindings[out_k])
        e1.record(); torch.cuda.synchronize()
        timings["T1_data"].append(e0.elapsed_time(e1))

        # ===== T2 Vision engine =====
        e0.record()
        engine.cuda_ctx.push()
        for i in range(len(engine.names)):
            engine.context.set_tensor_address(engine.names[i], engine.bindings[str(engine.names[i])].data_ptr())
        engine.context.execute_async_v3(stream_handle=engine.stream.handle)
        engine.stream.synchronize()
        engine.cuda_ctx.pop()
        e1.record(); torch.cuda.synchronize()
        timings["T2_vision"].append(e0.elapsed_time(e1))

        vision_embeded = engine.bindings["vision_embeded"].clone()

        # ===== T3 canbus_fixer（CANBUS_ON_GPU=1 时 GPU，否则 CPU）=====
        e0.record()
        if engine.canbus_fixer is not None:
            scene_same = torch.tensor([1.0]) if is_first.item() == 0 else torch.tensor([0.0])
            if getattr(engine, "_canbus_scene", None) != scene_token:
                engine.canbus_fixer.reset()
                engine._canbus_scene = scene_token
            _fixer_dev = next(engine.canbus_fixer.can_bus_embed.parameters()).device
            _to_dev = lambda t: t.to(_fixer_dev) if _fixer_dev.type == 'cuda' else t.cpu()
            with torch.no_grad():
                cb_embed = engine.canbus_fixer.step(
                    _to_dev(is_first.float()), _to_dev(scene_same),
                    _to_dev(ego_pose_inv.float()), _to_dev(ego_pose.float()),
                    _to_dev(command.float()), _to_dev(can_bus.float()))
            vision_embeded[0, 272, :] = cb_embed[0].to(vision_embeded.device)
        e1.record(); torch.cuda.synchronize()
        timings["T3_canbus"].append(e0.elapsed_time(e1))

        # ===== T4 LLM inputs_embeds 构建 =====
        e0.record()
        inputs_embeds, new_ids = llm.prepare_inputs_labels_for_multimodal_(input_ids, vision_embeded)
        e1.record(); torch.cuda.synchronize()
        timings["T4_llm_inputs"].append(e0.elapsed_time(e1))

        # ===== T5 LLM 推理 =====
        e0.record()
        output_ids, ego_feature, _per_layer = llm.generate_test(inputs_embeds, new_ids)
        e1.record(); torch.cuda.synchronize()
        timings["T5_llm"].append(e0.elapsed_time(e1))

        # ===== T6 ego 预测头（EGO_ON_GPU=1 时 GPU 批处理，否则 CPU 循环）=====
        e0.record()
        with torch.no_grad():
            _dev = next(torch_ego.parameters()).device
            if _dev.type == "cuda":
                ef = ego_feature.to(torch.float32).to(_dev)
                current_states = ef.unsqueeze(1)
                sample, _ = torch_ego.distribution_forward(current_states)
                hidden_states = ef.unsqueeze(1)
                states_hs, _ = torch_ego.future_states_predict(1, sample, hidden_states, current_states)
                ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                stacked = torch.cat([ego_query_hs[ti] for ti in range(6)], dim=0)
                torch_ego.ego_fut_decoder(stacked)
            else:
                ego_feature_cpu = ego_feature.to(torch.float32).cpu()
                current_states = ego_feature_cpu.unsqueeze(1)
                sample, _ = torch_ego.distribution_forward(current_states)
                hidden_states = ego_feature_cpu.unsqueeze(1)
                states_hs, _ = torch_ego.future_states_predict(1, sample, hidden_states, current_states)
                ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                for ti in range(6):
                    torch_ego.ego_fut_decoder(ego_query_hs[ti])
        e1.record(); torch.cuda.synchronize()
        timings["T6_ego"].append(e0.elapsed_time(e1))

        # ===== T7 输出后处理 =====
        e0.record()
        engine.get_bbox(engine.bindings["all_cls_scores"].clone(),
                        engine.bindings["all_bbox_preds"].clone(),
                        engine.bindings["all_traj_preds"].clone(),
                        engine.bindings["all_traj_cls_scores"].clone(),
                        batch["img_metas"][0])
        engine.get_lane(engine.bindings["all_lane_cls_one2one"].clone(),
                        engine.bindings["all_lane_preds_one2one"].clone(),
                        batch["img_metas"][0])
        e1.record(); torch.cuda.synchronize()
        timings["T7_postproc"].append(e0.elapsed_time(e1))

        total = sum(timings[k][-1] for k in timings if k != "T_total")
        timings["T_total"].append(total)

        if fi % 5 == 0:
            print(f"  frame {fi}: T_total={total:.1f}ms", flush=True)

    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print("  模块级延迟 profile（CUDA events，GPU 时间）")
    print("=" * 70)
    result = {}
    for k, v in timings.items():
        if len(v) == 0:
            continue
        arr = np.array(v)
        mean, p50, p95, mx = arr.mean(), np.percentile(arr, 50), np.percentile(arr, 95), arr.max()
        print(f"  {k:16s}  mean={mean:7.2f}ms  p50={p50:7.2f}  p95={p95:7.2f}  max={mx:7.2f}")
        result[k] = {"mean_ms": round(float(mean), 3), "p50_ms": round(float(p50), 3),
                     "p95_ms": round(float(p95), 3), "max_ms": round(float(mx), 3), "n": len(arr)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  已保存: {args.out}")

if __name__ == "__main__":
    main()
