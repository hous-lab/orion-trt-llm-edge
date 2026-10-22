"""
Per-layer latency profiling for Orion-Lite vision module.

Uses TensorRT's IProfiler interface to get per-layer GPU time,
then aggregates by module prefix (img_backbone, pts_bbox_head, map_head, etc.)

Usage:
    cd $ORION_LITE_PROJECT_ROOT
    python3 scripts/profile_layers.py \
        --engine eval/fused_ckpts/quant_benchmark/vision_fp16.engine \
        --test-data-dir eval/fused_ckpts/quant_benchmark/test_data \
        --output eval/fused_ckpts/quant_benchmark/fp16_layer_profile.json
"""

import argparse
import json
import numpy as np
import os
import re
import time
from collections import defaultdict

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


class Profiler(trt.IProfiler):
    """Collects per-layer timing from TensorRT."""

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layers = []

    def report_layer_time(self, layer_name, ms):
        self.layers.append({'name': layer_name, 'time_ms': ms})


class ProfiledEngine:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

        self.profiler = Profiler()
        self.context = self.engine.create_execution_context()
        self.context.profiler = self.profiler
        self.stream = cuda.Stream()

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer_once(self, inputs_dict):
        """Run one inference with profiling enabled."""
        self.profiler.layers = []  # reset

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

        for name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            out_size = int(np.prod(out_shape))
            out_data = np.zeros(out_size, dtype=np.float32)
            d_out = cuda.mem_alloc(out_data.nbytes)
            self.context.set_tensor_address(name, int(d_out))
            device_buffers[name] = d_out

        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()

        return list(self.profiler.layers)


def load_test_data(test_data_dir):
    inputs = {}
    for f in os.listdir(test_data_dir):
        if not f.endswith('.npy'):
            continue
        name = f[:-4]
        if name in ONNX_INPUTS:
            inputs[name] = np.load(os.path.join(test_data_dir, f))
    return inputs


def classify_layer(name):
    """Classify a layer into a module category based on its name."""
    # TRT layer names often contain the original PyTorch module path
    if 'img_backbone' in name or 'backbone' in name.lower():
        # Check if attention or MLP
        if 'attn' in name.lower() or 'attention' in name.lower():
            return 'ViT: Attention'
        elif 'mlp' in name.lower() or 'fc' in name.lower() or 'ffn' in name.lower():
            return 'ViT: MLP/FFN'
        elif 'norm' in name.lower() or 'layernorm' in name.lower():
            return 'ViT: LayerNorm'
        elif 'patch' in name.lower() or 'embed' in name.lower():
            return 'ViT: PatchEmbed'
        else:
            return 'ViT: Other'
    elif 'pts_bbox_head' in name or 'bbox_head' in name:
        return 'Agent Head (Detection+Track)'
    elif 'map_head' in name:
        return 'Map Head (Lane)'
    elif 'position_encoder' in name:
        return 'Position Encoder'
    elif 'memory' in name.lower():
        return 'Memory Module'
    elif 'student' in name.lower():
        return 'Student Model'
    else:
        return 'Other/Misc'


def classify_layer_fine(name):
    """Finer-grained classification for ViT layers."""
    if 'img_backbone' not in name and 'backbone' not in name.lower():
        return classify_layer(name)

    # Try to extract block index
    block_match = re.search(r'blocks?\.(\d+)', name)
    block_idx = int(block_match.group(1)) if block_match else -1

    if 'attn' in name.lower() or 'attention' in name.lower():
        if 'qkv' in name.lower() or 'matmul' in name.lower():
            return f'ViT Block {block_idx}: Attn QKV/MatMul'
        elif 'proj' in name.lower():
            return f'ViT Block {block_idx}: Attn Proj'
        else:
            return f'ViT Block {block_idx}: Attn Other'
    elif 'mlp' in name.lower():
        if 'fc1' in name.lower() or 'fc2' in name.lower():
            return f'ViT Block {block_idx}: MLP FC'
        else:
            return f'ViT Block {block_idx}: MLP Other'
    elif 'norm' in name.lower():
        return f'ViT Block {block_idx}: LayerNorm'
    else:
        return f'ViT Block {block_idx}: Other'


def run_profile(engine_path, inputs, warmup=3, runs=5):
    engine = ProfiledEngine(engine_path)

    # Warmup (without profiling data collection)
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        engine.infer_once(inputs)

    # Profiled runs
    all_runs = []
    for i in range(runs):
        layers = engine.infer_once(inputs)
        total = sum(l['time_ms'] for l in layers)
        all_runs.append({'run': i, 'total_ms': total, 'layers': layers})
        print(f"  Run {i}: {total:.2f} ms, {len(layers)} layers")

    return all_runs


def analyze_profile(all_runs, output_path):
    """Aggregate layer times by module category."""
    # Average across runs
    layer_times = defaultdict(list)
    for run in all_runs:
        for layer in run['layers']:
            layer_times[layer['name']].append(layer['time_ms'])

    # Compute mean per layer
    layer_means = {}
    for name, times in layer_times.items():
        layer_means[name] = np.mean(times)

    # Classify and aggregate
    category_times = defaultdict(float)
    fine_category_times = defaultdict(float)
    for name, t in layer_means.items():
        cat = classify_layer(name)
        category_times[cat] += t
        fine_cat = classify_layer_fine(name)
        fine_category_times[fine_cat] += t

    total_time = sum(layer_means.values())

    # Sort categories by time
    sorted_cats = sorted(category_times.items(), key=lambda x: -x[1])
    sorted_fine = sorted(fine_category_times.items(), key=lambda x: -x[1])

    # Print summary
    print(f"\n{'='*80}")
    print(f"  Layer Profile Summary (averaged over {len(all_runs)} runs)")
    print(f"  Total GPU time: {total_time:.2f} ms")
    print(f"  Total layers: {len(layer_means)}")
    print(f"{'='*80}")

    print(f"\n  {'Category':<45} {'Time(ms)':>10} {'Pct':>8}")
    print(f"  {'-'*65}")
    for cat, t in sorted_cats:
        pct = t / total_time * 100
        bar = '#' * int(pct / 2)
        print(f"  {cat:<45} {t:>10.2f} {pct:>7.1f}% {bar}")

    print(f"\n  {'Fine-grained (top 20)':<55} {'Time(ms)':>10} {'Pct':>8}")
    print(f"  {'-'*75}")
    for cat, t in sorted_fine[:20]:
        pct = t / total_time * 100
        print(f"  {cat:<55} {t:>10.2f} {pct:>7.1f}%")

    # Top 20 individual layers
    sorted_layers = sorted(layer_means.items(), key=lambda x: -x[1])
    print(f"\n  {'Top 20 Individual Layers':<70} {'Time(ms)':>10}")
    print(f"  {'-'*82}")
    for name, t in sorted_layers[:20]:
        short_name = name[:70] if len(name) > 70 else name
        print(f"  {short_name:<70} {t:>10.3f}")

    # Save JSON
    result = {
        'total_ms': total_time,
        'num_layers': len(layer_means),
        'categories': {k: v for k, v in sorted_cats},
        'fine_categories': {k: v for k, v in sorted_fine},
        'top_20_layers': [{'name': n, 'time_ms': t} for n, t in sorted_layers[:20]],
        'all_layers': [{'name': n, 'time_ms': t} for n, t in sorted_layers],
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Per-layer TRT profiling')
    parser.add_argument('--engine', required=True)
    parser.add_argument('--test-data-dir',
                        default='eval/fused_ckpts/quant_benchmark/test_data')
    parser.add_argument('--output', default='layer_profile.json')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    print(f"Engine: {args.engine}")
    inputs = load_test_data(args.test_data_dir)
    print(f"Loaded {len(inputs)} inputs")

    all_runs = run_profile(args.engine, inputs, args.warmup, args.runs)
    analyze_profile(all_runs, args.output)


if __name__ == '__main__':
    main()
