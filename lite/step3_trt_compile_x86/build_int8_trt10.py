"""
Build INT8 calibrated engine using TRT 10 with proper multi-input calibration data.

FIX (2026-07-30): Previous version read calibration_data/*.npy which only contained
the 'img' input (shape 6,3,640,640), causing all other 20 inputs to be zero-filled.
This made TensorRT unable to compute valid INT8 scale factors → all layers fell back
to FP16, resulting in engine size == FP16 engine (704MB).

This version reads calibration_samples/*.npz which contains ALL 21 inputs per sample.

Usage:
    cd $ORION_LITE_PROJECT_ROOT
    python3 scripts/build_int8_trt10.py \
        --onnx eval/fused_ckpts/quant_benchmark/orion_lite_vision_sim_mixed.onnx \
        --calib-dir eval/fused_ckpts/quant_benchmark/calibration_samples \
        --output eval/fused_ckpts/quant_benchmark/vision_int8_calibrated_trt10.engine \
        --num-calib 50
"""

import argparse
import numpy as np
import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class NpzCalibrationFeeder(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator that reads multi-input .npz files.

    Each .npz file contains all 21 ONNX inputs as named arrays.
    This ensures TensorRT sees real activation distributions for every layer,
    enabling proper INT8 quantization instead of FP16 fallback.
    """

    def __init__(self, calib_dir, input_shapes, cache_path, num_calib=50):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_shapes = input_shapes  # {name: tuple(shape)} from ONNX
        self.cache_path = cache_path
        self.current_idx = 0

        self.calib_files = sorted([
            f for f in os.listdir(calib_dir) if f.endswith('.npz')
        ])
        self.calib_files = self.calib_files[:num_calib]
        self.num_samples = len(self.calib_files)
        self.calib_dir = calib_dir

        print(f"  Calibration samples: {self.num_samples} .npz files from {calib_dir}")

        # Verify first sample has expected keys
        if self.num_samples > 0:
            check = np.load(os.path.join(calib_dir, self.calib_files[0]),
                            allow_pickle=True)
            keys = sorted(check.keys())
            print(f"  Sample keys ({len(keys)}): {keys}")
            missing = set(input_shapes.keys()) - set(keys)
            if missing:
                print(f"  WARNING: {len(missing)} inputs missing in npz: {sorted(missing)}")

        self.device_buffers = {}   # {name: DeviceAllocation}
        self.buffer_sizes = {}    # {name: int nbytes} — pycuda DeviceAllocation has no .size

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_idx >= self.num_samples:
            return None

        f = self.calib_files[self.current_idx]
        data = np.load(os.path.join(self.calib_dir, f), allow_pickle=True)

        bindings = []
        for name in names:
            if name in data:
                arr = data[name]
            else:
                # Fallback: zero-fill if missing (should not happen with correct npz)
                shape = self.input_shapes.get(name, (1,))
                arr = np.zeros(shape, dtype=np.float32)

            arr = arr.astype(np.float32)

            # Reshape scalar () to [1] to match ONNX input shape [1]
            target_shape = self.input_shapes.get(name)
            if target_shape is not None:
                target_prod = int(np.prod(target_shape))
                if arr.size == target_prod and arr.shape != tuple(target_shape):
                    arr = arr.reshape(target_shape)

            arr = np.ascontiguousarray(arr)

            if name not in self.device_buffers:
                self.device_buffers[name] = cuda.mem_alloc(arr.nbytes)
                self.buffer_sizes[name] = arr.nbytes
            elif self.buffer_sizes.get(name, 0) < arr.nbytes:
                # Reallocate if buffer too small (shouldn't happen with fixed shapes)
                self.device_buffers[name] = cuda.mem_alloc(arr.nbytes)
                self.buffer_sizes[name] = arr.nbytes

            cuda.memcpy_htod(self.device_buffers[name], arr)
            bindings.append(int(self.device_buffers[name]))

        self.current_idx += 1
        if self.current_idx % 10 == 0:
            print(f"    Calibrated {self.current_idx}/{self.num_samples} samples")

        return bindings

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            print(f"  Reading calibration cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"  Writing calibration cache: {self.cache_path} ({len(cache)} bytes)")
        with open(self.cache_path, 'wb') as f:
            f.write(cache)


def build_int8_engine(onnx_path, calib_dir, output_path, cache_path, num_calib=50):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Loading ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            return False

    print(f"Network: {network.num_inputs} inputs, {network.num_outputs} outputs")

    input_shapes = {}
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        input_shapes[inp.name] = tuple(inp.shape)
        print(f"  Input {i}: {inp.name} shape={inp.shape} dtype={inp.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    calibrator = NpzCalibrationFeeder(
        calib_dir, input_shapes, cache_path, num_calib=num_calib)
    config.int8_calibrator = calibrator

    print(f"\nBuilding INT8 engine with {num_calib} calibration samples...")
    print(f"  (FP16 fallback enabled for non-quantizable layers)")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: Failed to build engine!")
        return False

    with open(output_path, 'wb') as f:
        f.write(engine_bytes)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nEngine saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  (Compare to FP16 engine ~705MB. True INT8 should be significantly smaller.)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Build INT8 calibrated TRT engine with proper multi-input calibration')
    parser.add_argument('--onnx',
                        default='eval/fused_ckpts/quant_benchmark/orion_lite_vision_sim_mixed.onnx')
    parser.add_argument('--calib-dir',
                        default='eval/fused_ckpts/quant_benchmark/calibration_samples',
                        help='Directory with sample_XXXX.npz files (21 inputs each)')
    parser.add_argument('--output',
                        default='eval/fused_ckpts/quant_benchmark/vision_int8_calibrated_trt10.engine')
    parser.add_argument('--cache',
                        default='eval/fused_ckpts/quant_benchmark/calibration_trt10.cache')
    parser.add_argument('--num-calib', type=int, default=50)
    args = parser.parse_args()

    build_int8_engine(args.onnx, args.calib_dir, args.output,
                      args.cache, args.num_calib)


if __name__ == '__main__':
    main()
