#!/usr/bin/env python
"""Vision engine INT8 量化编译（标准 TRT INT8 calibrator）。

用 gen_vision_calib_data.py 生成的 .npz 校准数据 + entropy calibrator，
对 vision ONNX 做 INT8+FP16 混合精度编译。

产出: work_dirs/vision_int8.engine
"""
import os, sys, argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ROOT = os.environ.get("ORION_PROJECT_ROOT")
if not ROOT:
    raise SystemExit("error: ORION_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (see ckpts/README.md)")
os.chdir(ROOT)


class VisionCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, input_names, batch_size=1):
        super().__init__()
        self.calib_files = sorted([os.path.join(calib_dir, f)
                                   for f in os.listdir(calib_dir) if f.endswith('.npz')])
        self.input_names = input_names
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = {}
        first = np.load(self.calib_files[0])
        for name in self.input_names:
            if name in first:
                arr = first[name]
            else:
                arr = np.zeros((1, 1), dtype=np.float32)
            self.device_input[name] = cuda.mem_alloc(arr.nbytes)
        print(f"校准数据: {len(self.calib_files)} 帧, 输入: {len(self.input_names)} binding")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.calib_files):
            return None
        cuda.Context.synchronize()
        batch = np.load(self.calib_files[self.current_index])
        for name in names:
            if name in batch:
                arr = batch[name]
                cuda.memcpy_htod(self.device_input[name], arr)
        self.current_index += 1
        return [int(self.device_input[n]) for n in names]

    def read_calibration_cache(self):
        cache_path = os.path.join(ROOT, "calib_data", "vision_int8.cache")
        if os.path.exists(cache_path):
            print(f"读取校准缓存: {cache_path}")
            with open(cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        cache_path = os.path.join(ROOT, "calib_data", "vision_int8.cache")
        with open(cache_path, "wb") as f:
            f.write(cache)
        print(f"校准缓存已保存: {cache_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(ROOT, "work_dirs/orion_stage3_vision_mixed_precision.onnx"))
    ap.add_argument("--output", default = os.path.join(ROOT, "work_dirs/vision_int8.engine"))
    ap.add_argument("--calib-dir", default=os.path.join(ROOT, "calib_data"))
    ap.add_argument("--workspace-gb", type=int, default=16)
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"解析 ONNX: {args.onnx}")
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX error: {parser.get_error(i)}")
            raise RuntimeError("ONNX 解析失败")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 20))

    # INT8 + FP16 混合精度（INT8 为主，部分层 FP16 fallback）
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # 收集输入名称
    input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    print(f"ONNX 输入: {input_names}")

    # 设置 calibrator
    config.int8_calibrator = VisionCalibrator(args.calib_dir, input_names)

    print("开始 INT8 编译（含校准，约 5-15 分钟）...")
    import time
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("INT8 编译失败")
    dt = time.time() - t0
    print(f"编译完成，耗时 {dt:.0f}s")

    with open(args.output, "wb") as f:
        f.write(serialized)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"Engine 已保存: {args.output} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
