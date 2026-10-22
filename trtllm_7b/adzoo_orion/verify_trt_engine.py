"""
TensorRT Engine Inference Verification for Orion LLM
=====================================================
This script verifies that the TensorRT engine produces the same
hidden_states as the original PyTorch model.
"""

import argparse
import os
import sys
import numpy as np
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tensorrt as trt
import torch
from mmcv.utils.llava_llama import LlavaLlamaForCausalLM, LlavaConfig


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_trt_engine(engine_path: str):
    """Load TensorRT engine."""
    print(f"[INFO] Loading TensorRT engine from: {engine_path}")
    
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")
    
    print(f"[INFO] Engine loaded successfully")
    print(f"[INFO] Number of bindings: {engine.num_bindings}")
    
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        is_input = engine.binding_is_input(i)
        print(f"  - Binding {i}: {name}, shape={shape}, dtype={dtype}, input={is_input}")
    
    return engine


def allocate_buffers(engine, input_shapes: Dict[str, tuple]):
    """Allocate host and device buffers for inference."""
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        is_input = engine.binding_is_input(i)
        
        if is_input and name in input_shapes:
            shape = input_shapes[name]
        else:
            shape = engine.get_binding_shape(i)
        
        # Handle dynamic shapes
        if -1 in shape:
            if name in input_shapes:
                shape = input_shapes[name]
            else:
                shape = tuple(max(1, s) for s in shape)
        
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if is_input:
            inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': name})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': name})
    
    return inputs, outputs, bindings, stream


def infer_trt(engine, inputs_embeds: np.ndarray, new_input_ids: np.ndarray):
    """Run inference with TensorRT engine."""
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    context = engine.create_execution_context()
    
    # Set input shapes
    input_shape = inputs_embeds.shape
    ids_shape = new_input_ids.shape
    
    context.set_binding_shape(0, input_shape)
    context.set_binding_shape(1, ids_shape)
    
    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(
        engine, 
        {'inputs_embeds': input_shape, 'new_input_ids': ids_shape}
    )
    
    # Copy input data
    np.copyto(inputs[0]['host'], inputs_embeds.ravel())
    np.copyto(inputs[1]['host'], new_input_ids.ravel())
    
    # Transfer to device
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer from device
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    
    # Synchronize
    stream.synchronize()
    
    # Get output shapes
    selected_shape = context.get_binding_shape(2)
    full_shape = context.get_binding_shape(3)
    
    # Reshape outputs
    selected_hidden_states = outputs[0]['host'].reshape(selected_shape)
    full_hidden_states = outputs[1]['host'].reshape(full_shape)
    
    return selected_hidden_states, full_hidden_states


def load_pytorch_model(model_path: str, device: str = "cuda"):
    """Load PyTorch LLM model."""
    print(f"[INFO] Loading PyTorch model from: {model_path}")
    
    config = LlavaConfig.from_pretrained(model_path)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        use_gen_token=False,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    
    if not hasattr(model.config, 'waypoint_token_idx'):
        model.config.waypoint_token_idx = 32000
    
    print(f"[INFO] PyTorch model loaded successfully")
    return model, config


def extract_hidden_states_pytorch(
    model, inputs_embeds: torch.Tensor, new_input_ids: torch.Tensor,
    waypoint_token_idx: int = 32000,
):
    """Extract hidden_states using PyTorch model."""
    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs[0]
        
        batch_size = new_input_ids.shape[0]
        selected_hidden_states = []
        
        for i in range(batch_size):
            loc_positions = (new_input_ids[i] == waypoint_token_idx)
            if loc_positions.sum() > 0:
                selected_hs = hidden_states[i][loc_positions][0]
            else:
                selected_hs = hidden_states[i][-1]
            selected_hidden_states.append(selected_hs)
        
        selected_hidden_states = torch.stack(selected_hidden_states, dim=0)
    
    return selected_hidden_states, hidden_states


def main():
    parser = argparse.ArgumentParser(description="Verify Orion LLM TensorRT Engine")
    parser.add_argument("--engine_path", type=str, default="ckpts/llm_trt/orion_llm_fp16.engine")
    parser.add_argument("--pytorch_model_path", type=str, default="ckpts/llm_clean_no_lora/")
    parser.add_argument("--waypoint_token_idx", type=int, default=32000)
    parser.add_argument("--seq_len", type=int, default=600)
    parser.add_argument("--num_tests", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load TensorRT engine
    engine = load_trt_engine(args.engine_path)
    
    # Load PyTorch model
    pt_model, config = load_pytorch_model(args.pytorch_model_path, args.device)
    
    # Run verification tests
    print(f"\n[INFO] Running {args.num_tests} verification tests...")
    
    all_selected_diffs = []
    all_full_diffs = []
    
    for i in range(args.num_tests):
        print(f"\n{'='*60}")
        print(f"[INFO] Test {i+1}/{args.num_tests}")
        print(f"{'='*60}")
        
        # Create random inputs
        inputs_embeds_np = np.random.randn(1, args.seq_len, 4096).astype(np.float32)
        new_input_ids_np = np.zeros((1, args.seq_len), dtype=np.int64)
        waypoint_pos = np.random.randint(args.seq_len // 2, args.seq_len)
        new_input_ids_np[0, waypoint_pos] = args.waypoint_token_idx
        
        # TensorRT inference
        trt_selected, trt_full = infer_trt(engine, inputs_embeds_np, new_input_ids_np)
        
        # PyTorch inference
        inputs_embeds_pt = torch.from_numpy(inputs_embeds_np).to(torch.float16).cuda()
        new_input_ids_pt = torch.from_numpy(new_input_ids_np).cuda()
        pt_selected, pt_full = extract_hidden_states_pytorch(
            pt_model, inputs_embeds_pt, new_input_ids_pt, args.waypoint_token_idx
        )
        
        # Compare
        selected_diff = np.abs(pt_selected.cpu().numpy() - trt_selected).max()
        full_diff = np.abs(pt_full.cpu().numpy() - trt_full).max()
        
        print(f"  - selected_hidden_states max diff: {selected_diff:.6f}")
        print(f"  - full_hidden_states max diff: {full_diff:.6f}")
        
        all_selected_diffs.append(selected_diff)
        all_full_diffs.append(full_diff)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[INFO] Verification Summary")
    print(f"{'='*60}")
    print(f"  - selected_hidden_states max diff: {max(all_selected_diffs):.6f}")
    print(f"  - full_hidden_states max diff: {max(all_full_diffs):.6f}")
    
    threshold = 0.1
    if max(all_selected_diffs) < threshold and max(all_full_diffs) < threshold:
        print(f"\n[INFO] ✅ TensorRT engine verification PASSED! (threshold: {threshold})")
    else:
        print(f"\n[INFO] ❌ TensorRT engine verification FAILED! (threshold: {threshold})")


if __name__ == "__main__":
    main()
