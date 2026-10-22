"""
Remove all _FORCEFP32 suffixes from ONNX node names.

The original export marked ~600 nodes with _FORCEFP32 to hint TensorRT
to use FP32 for precision-sensitive layers. However, when compiling with
plain --fp16 (without --layerPrecisions), these suffixes don't change TRT's
precision selection but DO fragment the computation graph, causing extra
Cast/Convert operations between named boundaries.

By removing all suffixes, TRT gets a clean graph and can freely fuse/ optimize
without artificial boundaries, potentially eliminating the 18.8ms (15.7%) Cast overhead.

Usage:
    cd $ORION_LITE_PROJECT_ROOT
    python3 scripts/strip_forcefp32.py \
        --input eval/fused_ckpts/quant_benchmark/orion_lite_vision_sim_mixed.onnx \
        --output eval/fused_ckpts/quant_benchmark/orion_lite_vision_sim_clean.onnx
"""

import argparse
import os

import onnx


def strip_forcefp32(input_path, output_path):
    print(f"Loading: {input_path}")
    model = onnx.load(input_path)

    # Remove _FORCEFP32 suffixes from all node names
    count = 0
    for node in model.graph.node:
        if node.name and node.name.endswith('_FORCEFP32'):
            node.name = node.name[:-len('_FORCEFP32')]
            count += 1

    print(f"  Stripped {count} _FORCEFP32 suffixes")

    # Save (handle large models with external data)
    model_size = os.path.getsize(input_path) / (1024 * 1024)
    save_external = model_size > 1.8 * 1024
    onnx.save(model, output_path, save_as_external_data=save_external)

    out_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({out_size:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default='eval/fused_ckpts/quant_benchmark/orion_lite_vision_sim_mixed.onnx')
    parser.add_argument('--output',
                        default='eval/fused_ckpts/quant_benchmark/orion_lite_vision_sim_clean.onnx')
    args = parser.parse_args()
    strip_forcefp32(args.input, args.output)


if __name__ == '__main__':
    main()
