#!/usr/bin/env python
"""Build an FP16 TensorRT engine from the exported vision ONNX.

Usage:
  python build_fp16.py --onnx orion_lite_vision_sim_mixed.onnx --output vision_fp16.engine
"""
import argparse

import tensorrt as trt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workspace-gb", type=float, default=8.0)
    args = p.parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("error: ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb) << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("error: engine build failed")
    with open(args.output, "wb") as f:
        f.write(engine)
    print(f"engine written: {args.output}")


if __name__ == "__main__":
    main()
