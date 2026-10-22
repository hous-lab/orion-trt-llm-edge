"""Verify that a TRT-LLM engine exposes the full_hidden_states output binding.

This is the build gate for the hidden-states export: it deserializes the engine,
lists every I/O tensor, confirms full_hidden_states is a real output binding, and
cross-checks the engine config vocab_size (the vocab_size=32001 waypoint-token
fix, see docs/hidden_states_export.md and the checkpoint-prep chain).

The functional gate is separate: verify_hs_compare.py compares engine hidden
states against the PyTorch reference (cos gate), and run_trt_eval.sh runs the
173-frame planning eval.

Usage (mirrors run_verify.sh):
    python trtllm_7b/verify_hidden_states.py --engine_dir <engine dir> \
        [--output_dir <dir>]
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import tensorrt as trt


def parse_args():
    ap = argparse.ArgumentParser(description='Verify full_hidden_states engine output.')
    ap.add_argument('--engine_dir', required=True,
                    help='TRT-LLM engine directory (contains rank0.engine + config.json)')
    ap.add_argument('--output_dir', default=None,
                    help='optional directory for the JSON report')
    return ap.parse_args()


def main():
    args = parse_args()
    engine_path = os.path.join(args.engine_dir, 'rank0.engine')
    if not os.path.isfile(engine_path):
        print(f"ERROR: engine not found: {engine_path}")
        return 1

    # Register the TRT-LLM plugins (WeightOnlyQuantMatmul etc.) before
    # deserializing; a W8A16 engine fails to deserialize without them.
    from tensorrt_llm.plugin import _load_plugin_lib
    _load_plugin_lib()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("ERROR: failed to deserialize engine")
        return 1

    print(f"Engine: {engine_path} "
          f"({os.path.getsize(engine_path) / 1024**3:.1f} GB)")
    print(f"\nEngine I/O tensors ({engine.num_io_tensors}):")
    io_table = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = ('INPUT' if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                else 'OUTPUT')
        shape = list(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        io_table.append({'name': name, 'mode': mode,
                         'shape': shape, 'dtype': str(dtype)})
        print(f"  [{mode}] {name}: shape={shape}, dtype={dtype}")

    has_hs = any(t['name'] == 'full_hidden_states' for t in io_table)
    if has_hs:
        hs = next(t for t in io_table if t['name'] == 'full_hidden_states')
        print("\nPASS: full_hidden_states is an engine output "
              f"(shape={hs['shape']}, dtype={hs['dtype']})")
    else:
        print("\nFAIL: full_hidden_states not found in engine outputs")
        print("  The modeling_utils mark_output patch was not applied when the")
        print("  engine was built - re-apply trtllm_7b/patches/modeling_utils_v0x.patch")
        print("  and rebuild (trtllm_7b/run_convert_and_build.sh).")
        return 1

    # Cross-check the engine config: vocab_size must be 32001 (waypoint token).
    vocab = None
    cfg_path = os.path.join(args.engine_dir, 'config.json')
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        pretrained = cfg.get('pretrained_config', {})
        vocab = pretrained.get('vocab_size', cfg.get('vocab_size'))
        print(f"\nEngine config vocab_size: {vocab}")
        if vocab != 32001:
            print("FAIL: vocab_size != 32001 - the waypoint token (id 32000) would")
            print("  fall back to the pad embedding and corrupt planning outputs.")
            print("  Regenerate the checkpoint with patch_llm_checkpoint.py and rebuild.")
            return 1
        print("PASS: vocab_size=32001 (real waypoint-token embedding)")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        report = {
            'engine_dir': args.engine_dir,
            'engine_path': engine_path,
            'engine_size_gb': round(os.path.getsize(engine_path) / 1024**3, 3),
            'num_io_tensors': engine.num_io_tensors,
            'io_tensors': io_table,
            'full_hidden_states': hs if has_hs else None,
            'vocab_size': vocab,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        out = os.path.join(args.output_dir, 'verify_hidden_states_report.json')
        with open(out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport: {out}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
