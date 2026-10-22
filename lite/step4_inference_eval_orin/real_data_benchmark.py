"""
Orion-Lite Quantization Benchmark - Using REAL test data.

Uses the complete test data saved by export_vision_onnx.py (--save-test-data),
which includes all 21 ONNX inputs and 25 reference outputs from PyTorch FP32.

This provides accurate, reproducible precision and performance measurements.

Usage (4090 Dev Machine):
    python scripts/real_data_benchmark.py \
        --test-data-dir eval/fused_ckpts/quant_benchmark/test_data \
        --output-dir eval/fused_ckpts/quant_benchmark

Usage (Orin):
    python3 scripts/real_data_benchmark.py \
        --test-data-dir eval/fused_ckpts/quant_benchmark/test_data \
        --output-dir eval/fused_ckpts/quant_benchmark
"""

import argparse
import json
import numpy as np
import os
import subprocess
import sys
import time
from datetime import datetime

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ONNX_INPUTS = [
    'img', 'img2lidars', 'command', 'can_bus', 'is_first_frame',
    'ego_pose', 'timestamp', 'ego_pose_inv',
    'memory_embedding_bbox_in', 'memory_reference_point_bbox_in',
    'memory_timestamp_bbox_in', 'memory_egopose_bbox_in',
    'memory_canbus_bbox_in', 'sample_time_bbox_in',
    'memory_scene_query_bbox_in', 'scene_memory_timestamp_bbox_in',
    'memory_timestamp_map_in', 'sample_time_map_in',
    'memory_egopose_map_in', 'memory_embedding_map_in',
    'memory_reference_point_map_in',
]

CORE_OUTPUTS = [
    'vision_embeded', 'all_cls_scores', 'all_bbox_preds',
    'all_traffic_states', 'all_traj_preds', 'all_traj_cls_scores',
    'all_lane_cls_one2one', 'all_lane_preds_one2one',
    'all_lane_cls_one2many', 'all_lane_preds_one2many',
    'outs_dec_one2one', 'outs_dec_one2many',
]

MEMORY_OUTPUTS = [
    'memory_embedding_bbox_out', 'memory_reference_point_bbox_out',
    'memory_embedding_map_out', 'memory_reference_point_map_out',
    'memory_timestamp_bbox_out', 'memory_timestamp_map_out',
    'memory_egopose_bbox_out', 'memory_egopose_map_out',
    'memory_canbus_bbox_out', 'memory_scene_query_bbox_out',
    'sample_time_bbox_out', 'sample_time_map_out',
    'scene_memory_timestamp_bbox_out',
]

ENGINE_CONFIGS = {
    'fp16': {
        'pattern': 'vision_fp16.engine',
        'label': 'FP16',
    },
    'int8_calibrated': {
        'pattern': 'vision_int8_calibrated_trt10.engine',
        'label': 'INT8 (calibrated)',
    },
}


class TRTEngine:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer_with_inputs(self, inputs_dict):
        device_buffers = {}

        for name in self.input_names:
            data = inputs_dict.get(name)
            if data is None:
                shape = tuple(self.engine.get_tensor_shape(name))
                data = np.zeros(shape, dtype=np.float32)
            data = np.ascontiguousarray(data.astype(np.float32))
            d_mem = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(d_mem, data)
            self.context.set_tensor_address(name, int(d_mem))
            device_buffers[name] = d_mem

        output_results = {}
        for name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            out_size = int(np.prod(out_shape))
            out_data = np.zeros(out_size, dtype=np.float32)
            d_out = cuda.mem_alloc(out_data.nbytes)
            self.context.set_tensor_address(name, int(d_out))
            device_buffers[name] = d_out
            output_results[name] = (out_data, d_out, out_shape)

        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()

        results = {}
        for name, (out_data, d_out, out_shape) in output_results.items():
            cuda.memcpy_dtoh(out_data, d_out)
            results[name] = out_data.reshape(out_shape)

        return results


def get_platform_info():
    info = {}
    try:
        info['trt_version'] = trt.__version__
    except Exception:
        pass
    try:
        info['cuda_version'] = cuda.get_version()
    except Exception:
        pass
    try:
        dev = cuda.Device(0)
        info['gpu_name'] = dev.name()
        info['gpu_memory_mb'] = dev.total_memory() // (1024 * 1024)
    except Exception:
        pass
    try:
        out = subprocess.check_output(['uname', '-a'], text=True).strip()
        info['kernel'] = out
    except Exception:
        pass
    return info


def load_test_data(test_data_dir):
    inputs = {}
    references = {}

    for f in os.listdir(test_data_dir):
        if not f.endswith('.npy'):
            continue
        path = os.path.join(test_data_dir, f)
        name = f[:-4]

        if name.endswith('_ref'):
            key = name[:-4]
            references[key] = np.load(path)
        else:
            if name in ONNX_INPUTS:
                inputs[name] = np.load(path)

    return inputs, references


def compute_metrics(engine_output, reference_output):
    metrics = {}

    all_keys = set(engine_output.keys()) & set(reference_output.keys())

    for name in sorted(all_keys):
        a = reference_output[name].flatten()
        b = engine_output[name].flatten()

        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        l2 = np.sqrt(np.mean((a - b) ** 2))

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_sim = np.dot(a, b) / (norm_a * norm_b + 1e-10)

        max_diff = np.max(np.abs(a - b))
        mean_diff = np.mean(np.abs(a - b))
        rel_diff = mean_diff / (np.mean(np.abs(a)) + 1e-10) * 100

        metrics[name] = {
            'cosine_similarity': float(cos_sim),
            'l2_distance': float(l2),
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'relative_diff_pct': float(rel_diff),
            'shape': list(reference_output[name].shape),
            'ref_mean': float(np.mean(reference_output[name])),
            'ref_std': float(np.std(reference_output[name])),
        }

    return metrics


def benchmark_latency(engine_path, inputs, warmup=5, runs=30):
    engine = TRTEngine(engine_path)

    for _ in range(warmup):
        engine.infer_with_inputs(inputs)

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        engine.infer_with_inputs(inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'fps': float(1000 / np.mean(latencies)),
        'runs': runs,
    }


def find_engines(engine_dir):
    found = {}
    for key, cfg in ENGINE_CONFIGS.items():
        path = os.path.join(engine_dir, cfg['pattern'])
        if os.path.exists(path):
            found[key] = {
                'path': path,
                'label': cfg['label'],
                'size_mb': os.path.getsize(path) / (1024 * 1024),
            }
    return found


def print_metrics_table(metrics, output_list, title):
    print(f"\n  {title}")
    print(f"  {'Output':<35} {'CosSim':>8} {'L2 Dist':>10} {'RelErr%':>8} {'MaxDiff':>10} {'Status':>6}")
    print(f"  {'-'*80}")
    for name in output_list:
        if name in metrics:
            m = metrics[name]
            if m['cosine_similarity'] > 0.999:
                status = "OK"
            elif m['cosine_similarity'] > 0.99:
                status = "WARN"
            else:
                status = "BAD"
            print(f"  {name:<35} {m['cosine_similarity']:>8.6f} {m['l2_distance']:>10.6f} {m['relative_diff_pct']:>8.2f} {m['max_diff']:>10.6f} {status:>6}")


def parse_args():
    parser = argparse.ArgumentParser(description='Orion-Lite Real Data Benchmark')
    parser.add_argument('--test-data-dir', default='eval/fused_ckpts/quant_benchmark/test_data')
    parser.add_argument('--engine-dir', default='eval/fused_ckpts/quant_benchmark')
    parser.add_argument('--output-dir', default='eval/fused_ckpts/quant_benchmark')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--runs', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    platform = get_platform_info()

    print("=" * 80)
    print("  Orion-Lite Quantization Benchmark (REAL DATA)")
    print("=" * 80)
    print(f"  GPU: {platform.get('gpu_name', 'Unknown')}")
    print(f"  GPU Memory: {platform.get('gpu_memory_mb', '?')} MB")
    print(f"  TensorRT: {platform.get('trt_version', '?')}")
    print(f"  CUDA: {platform.get('cuda_version', '?')}")
    print(f"  Test data: {args.test_data_dir}")
    print(f"  Engine dir: {args.engine_dir}")

    inputs, references = load_test_data(args.test_data_dir)
    print(f"\n  Loaded {len(inputs)} inputs, {len(references)} reference outputs (PyTorch FP32)")

    missing = [n for n in ONNX_INPUTS if n not in inputs]
    if missing:
        print(f"  WARNING: Missing inputs: {missing}")

    engines = find_engines(args.engine_dir)
    if not engines:
        print("\n  ERROR: No engine files found!")
        sys.exit(1)

    print(f"\n  Found {len(engines)} engine(s):")
    for key, info in engines.items():
        print(f"    {info['label']}: {info['path']} ({info['size_mb']:.1f} MB)")

    results = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform,
        'test_data_dir': args.test_data_dir,
        'num_inputs': len(inputs),
        'num_references': len(references),
        'engines': {},
    }

    all_metrics = {}

    for key, info in engines.items():
        label = info['label']
        path = info['path']
        print(f"\n{'=' * 80}")
        print(f"  Testing: {label}")
        print(f"  Engine: {path} ({info['size_mb']:.1f} MB)")
        print(f"{'=' * 80}")

        try:
            engine = TRTEngine(path)
        except Exception as e:
            print(f"  ERROR: Failed to load engine: {e}")
            results['engines'][key] = {
                'label': label,
                'path': path,
                'size_mb': info['size_mb'],
                'error': str(e),
            }
            continue
        output = engine.infer_with_inputs(inputs)
        metrics = compute_metrics(output, references)
        all_metrics[key] = metrics

        print_metrics_table(metrics, CORE_OUTPUTS, "Core Outputs (Detection + Planning)")
        print_metrics_table(metrics, MEMORY_OUTPUTS, "Memory Outputs (Temporal)")

        latency = benchmark_latency(path, inputs, args.warmup, args.runs)
        print(f"\n  Latency: {latency['mean_ms']:.1f} +/- {latency['std_ms']:.1f} ms  |  "
              f"FPS: {latency['fps']:.2f}  |  "
              f"P50: {latency['p50_ms']:.1f} ms  |  P95: {latency['p95_ms']:.1f} ms")

        core_cos_sims = [metrics[n]['cosine_similarity'] for n in CORE_OUTPUTS if n in metrics]
        avg_core_cos = np.mean(core_cos_sims) if core_cos_sims else 0

        results['engines'][key] = {
            'label': label,
            'path': path,
            'size_mb': info['size_mb'],
            'avg_core_cosine_similarity': float(avg_core_cos),
            'latency': latency,
            'metrics': metrics,
        }

    print(f"\n{'=' * 80}")
    print("  SUMMARY: Cross-Engine Comparison")
    print(f"{'=' * 80}")
    print(f"  {'Engine':<25} {'Size(MB)':>10} {'Avg CosSim':>12} {'Latency(ms)':>12} {'FPS':>8} {'Speedup':>8}")
    print(f"  {'-' * 80}")

    fp16_latency = None
    for key, info in engines.items():
        eng_data = results['engines'][key]
        if 'error' in eng_data:
            print(f"  {info['label']:<25} {info['size_mb']:>10.1f} {'FAILED':>12} {'-':>12} {'-':>8} {'-':>8}")
            continue
        lat = eng_data['latency']
        cos = eng_data['avg_core_cosine_similarity']
        if key == 'fp16':
            fp16_latency = lat['mean_ms']
        speedup = f"{fp16_latency / lat['mean_ms']:.2f}x" if fp16_latency and key != 'fp16' else "1.00x"
        print(f"  {info['label']:<25} {info['size_mb']:>10.1f} {cos:>12.6f} {lat['mean_ms']:>12.1f} {lat['fps']:>8.2f} {speedup:>8}")

    print(f"\n{'=' * 80}")
    print("  DETAILED: Core Output Comparison Across Engines")
    print(f"{'=' * 80}")

    engine_keys = [k for k in engines.keys() if k in all_metrics]
    header = f"  {'Output':<30}"
    for key in engine_keys:
        header += f" {ENGINE_CONFIGS[key]['label']:>16}"
    print(header)
    print(f"  {'-' * (30 + 17 * len(engine_keys))}")

    for name in CORE_OUTPUTS:
        row = f"  {name:<30}"
        for key in engine_keys:
            if name in all_metrics[key]:
                m = all_metrics[key][name]
                row += f" {m['cosine_similarity']:>10.6f}/{m['relative_diff_pct']:>4.1f}%"
            else:
                row += f" {'N/A':>16}"
        print(row)

    results_path = os.path.join(args.output_dir, 'benchmark_results_real_data.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
