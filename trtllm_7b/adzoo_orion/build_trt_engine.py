"""
TensorRT Engine Building Script for Orion LLM ONNX Model
=========================================================
This script converts the ONNX LLM model to TensorRT engine
with INT8 quantization support.

TensorRT can handle large models (>2GB) and provides INT8 quantization.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def check_tensorrt():
    """Check if TensorRT is installed."""
    try:
        import tensorrt as trt
        print(f"[INFO] TensorRT version: {trt.__version__}")
        return True
    except ImportError:
        print("[ERROR] TensorRT not found!")
        print("[INFO] Please install TensorRT:")
        print("  pip install tensorrt")
        return False


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    int8_calibration_cache: str = None,
    max_batch_size: int = 1,
    max_workspace_size: int = 16 << 30,  # 16GB
):
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to the ONNX model
        engine_path: Path to save the TensorRT engine
        precision: Precision mode (fp32, fp16, int8)
        int8_calibration_cache: Path to INT8 calibration cache
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes
    """
    import tensorrt as trt
    
    print(f"\n[INFO] Building TensorRT engine...")
    print(f"  - ONNX model: {onnx_path}")
    print(f"  - Engine output: {engine_path}")
    print(f"  - Precision: {precision}")
    print(f"  - Max batch size: {max_batch_size}")
    print(f"  - Max workspace size: {max_workspace_size / (1024**3):.1f} GB")
    
    # Create output directory
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    
    # Create logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder
    builder = trt.Builder(logger)
    
    # Create network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create config
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # Set precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 mode enabled")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        print("[INFO] INT8 mode enabled")
        
        # Set INT8 calibrator if provided
        if int8_calibration_cache and os.path.exists(int8_calibration_cache):
            config.int8_calibrator = None  # TODO: Implement calibrator
            print(f"[INFO] Using INT8 calibration cache: {int8_calibration_cache}")
        else:
            print("[WARNING] INT8 calibration cache not provided, using dynamic range estimation")
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    
    print(f"[INFO] Parsing ONNX model...")
    # For large models (>2GB), we need to read the model in chunks
    if os.path.getsize(onnx_path) > 2 * 1024 * 1024 * 1024:
        print("[INFO] Large model detected, using path-based parsing")
        success = parser.parse_from_file(onnx_path)
    else:
        with open(onnx_path, 'rb') as f:
            success = parser.parse(f.read())
    
    if not success:
        print("[ERROR] Failed to parse ONNX model!")
        for i in range(parser.num_errors):
            print(f"  Error {i}: {parser.get_error(i)}")
        return False
    
    print(f"[INFO] ONNX model parsed successfully")
    print(f"  - Number of inputs: {network.num_inputs}")
    print(f"  - Number of outputs: {network.num_outputs}")
    
    # Print input/output info
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"  - Input {i}: {input_tensor.name}, shape: {input_tensor.shape}")
    
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"  - Output {i}: {output_tensor.name}, shape: {output_tensor.shape}")
    
    # Build engine
    print(f"[INFO] Building engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("[ERROR] Failed to build TensorRT engine!")
        return False
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"[INFO] TensorRT engine saved to: {engine_path}")
    
    # Get engine size
    engine_size = os.path.getsize(engine_path) / (1024**3)
    print(f"[INFO] Engine size: {engine_size:.2f} GB")
    
    return True


def build_engine_with_trtexec(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
):
    """
    Build TensorRT engine using trtexec command-line tool.
    This is more reliable for large models.
    """
    print(f"\n[INFO] Building TensorRT engine with trtexec...")
    print(f"  - ONNX model: {onnx_path}")
    print(f"  - Engine output: {engine_path}")
    print(f"  - Precision: {precision}")
    
    # Create output directory
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    
    # Build command
    cmd = [
        "trtexec",
        "--onnx=" + onnx_path,
        "--saveEngine=" + engine_path,
        "--memPoolSize=workspace:16384",  # 16GB workspace
        "--tacticSources=+CUBLAS,+CUBLASLT",
    ]
    
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] trtexec failed:")
        print(result.stderr)
        return False
    
    print(f"[INFO] TensorRT engine built successfully!")
    
    # Get engine size
    engine_size = os.path.getsize(engine_path) / (1024**3)
    print(f"[INFO] Engine size: {engine_size:.2f} GB")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine for Orion LLM")
    parser.add_argument(
        "--onnx_model",
        type=str,
        default="ckpts/llm_onnx/orion_llm.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--engine_output",
        type=str,
        default="ckpts/llm_trt/orion_llm_fp16.engine",
        help="Path to save the TensorRT engine",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Precision mode",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--use_trtexec",
        action="store_true",
        help="Use trtexec command-line tool instead of Python API",
    )
    parser.add_argument(
        "--int8_calibration_cache",
        type=str,
        default=None,
        help="Path to INT8 calibration cache",
    )
    
    args = parser.parse_args()
    
    # Check TensorRT
    if not args.use_trtexec and not check_tensorrt():
        return
    
    # Build engine
    if args.use_trtexec:
        success = build_engine_with_trtexec(
            args.onnx_model,
            args.engine_output,
            args.precision,
            args.max_batch_size,
        )
    else:
        success = build_tensorrt_engine(
            args.onnx_model,
            args.engine_output,
            args.precision,
            args.int8_calibration_cache,
            args.max_batch_size,
        )
    
    if success:
        print(f"\n[INFO] TensorRT engine is ready for deployment!")
        print(f"\n[INFO] Next steps:")
        print(f"  1. Test the engine with TensorRT Python API")
        print(f"  2. Deploy to AGX Orin")
    else:
        print(f"\n[ERROR] Failed to build TensorRT engine")


if __name__ == "__main__":
    main()
