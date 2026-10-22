"""
Orion-Lite Student Model → ONNX Export + Test Data Preparation
===============================================================

Exports the OrionMimicModel (student model from the distilled Orion-Lite)
to ONNX format for TensorRT compilation.

Also prepares test input data as numpy files for cross-platform benchmarking.

This follows the same pattern as:
  Orion-main/adzoo/orion/export_vision_orion_wolm.py

Usage:
    python export_student_onnx.py --ckpt ../eval/fused_ckpts/orion_lite_subset8.pth
    python export_student_onnx.py --ckpt ../eval/fused_ckpts/orion_lite_subset8.pth --fp16
"""

import sys
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic

# Register custom symbolic function for aten::unflatten
# This is needed for GRU export in PyTorch 2.0
def unflatten_symbolic(g, self, dim, sizes):
    # unflatten(self, dim, sizes) -> reshape
    return g.op("Reshape", self, sizes)

register_custom_op_symbolic("aten::unflatten", unflatten_symbolic, 18)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "distill_code"))
sys.path.insert(0, os.environ.get("ORION_LITE_PROJECT_ROOT", os.path.join(os.path.dirname(__file__), "..", "..")))
from student_model_with_orion_loss import OrionMimicModel


class MultiHeadAttentionONNX(nn.Module):
    """ONNX-friendly multi-head attention without SDPA."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, N_q, C = query.shape
        B, N_kv, C = key.shape  # key and value have same shape

        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).contiguous().view(B, N_q, C)
        return self.out_proj(x)


class TransformerDecoderLayerONNX(nn.Module):
    """ONNX-friendly transformer decoder layer."""

    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionONNX(d_model, nhead)
        self.multihead_attn = MultiHeadAttentionONNX(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderONNX(nn.Module):
    """ONNX-friendly transformer decoder."""

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayerONNX(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt


class StudentModelONNXProxy(nn.Module):
    """Wrapper for ONNX export with ONNX-friendly attention.

    Replaces nn.TransformerDecoder with custom ONNX-friendly implementation.
    """

    def __init__(self, student_model):
        super().__init__()
        self.input_proj = student_model.input_proj
        self.norm_input = student_model.norm_input
        self.planning_query = student_model.planning_query

        # Replace transformer decoder with ONNX-friendly version
        self.transformer_decoder = TransformerDecoderONNX(
            d_model=1024, nhead=16, dim_feedforward=4096, num_layers=6, dropout=0.0
        )
        # Copy weights from original decoder
        self._copy_decoder_weights(student_model.transformer_decoder)

        self.output_proj = student_model.output_proj
        self.norm_output = student_model.norm_output

        # VAE components
        self.present_distribution = student_model.present_distribution
        self.predict_model = student_model.predict_model

        # Unroll ego_fut_decoder (nn.Sequential with 5 layers)
        self.ego_fut_decoder_layers = nn.ModuleList([
            layer for layer in student_model.ego_fut_decoder
        ])

        self.fut_ts = 6
        self.ego_fut_mode = student_model.ego_fut_mode
        self.latent_dim = student_model.latent_dim
        self.layer_dim = student_model.layer_dim
        self.with_cur = student_model.with_cur

    def _copy_decoder_weights(self, original_decoder):
        """Copy weights from original nn.TransformerDecoder to ONNX-friendly version."""
        for i, orig_layer in enumerate(original_decoder.layers):
            new_layer = self.transformer_decoder.layers[i]

            # Self attention
            new_layer.self_attn.q_proj.weight.data = orig_layer.self_attn.in_proj_weight[:1024].clone()
            new_layer.self_attn.q_proj.bias.data = orig_layer.self_attn.in_proj_bias[:1024].clone()
            new_layer.self_attn.k_proj.weight.data = orig_layer.self_attn.in_proj_weight[1024:2048].clone()
            new_layer.self_attn.k_proj.bias.data = orig_layer.self_attn.in_proj_bias[1024:2048].clone()
            new_layer.self_attn.v_proj.weight.data = orig_layer.self_attn.in_proj_weight[2048:].clone()
            new_layer.self_attn.v_proj.bias.data = orig_layer.self_attn.in_proj_bias[2048:].clone()
            new_layer.self_attn.out_proj.weight.data = orig_layer.self_attn.out_proj.weight.clone()
            new_layer.self_attn.out_proj.bias.data = orig_layer.self_attn.out_proj.bias.clone()

            # Cross attention
            new_layer.multihead_attn.q_proj.weight.data = orig_layer.multihead_attn.in_proj_weight[:1024].clone()
            new_layer.multihead_attn.q_proj.bias.data = orig_layer.multihead_attn.in_proj_bias[:1024].clone()
            new_layer.multihead_attn.k_proj.weight.data = orig_layer.multihead_attn.in_proj_weight[1024:2048].clone()
            new_layer.multihead_attn.k_proj.bias.data = orig_layer.multihead_attn.in_proj_bias[1024:2048].clone()
            new_layer.multihead_attn.v_proj.weight.data = orig_layer.multihead_attn.in_proj_weight[2048:].clone()
            new_layer.multihead_attn.v_proj.bias.data = orig_layer.multihead_attn.in_proj_bias[2048:].clone()
            new_layer.multihead_attn.out_proj.weight.data = orig_layer.multihead_attn.out_proj.weight.clone()
            new_layer.multihead_attn.out_proj.bias.data = orig_layer.multihead_attn.out_proj.bias.clone()

            # FFN
            new_layer.linear1.weight.data = orig_layer.linear1.weight.clone()
            new_layer.linear1.bias.data = orig_layer.linear1.bias.clone()
            new_layer.linear2.weight.data = orig_layer.linear2.weight.clone()
            new_layer.linear2.bias.data = orig_layer.linear2.bias.clone()

            # LayerNorm
            new_layer.norm1.weight.data = orig_layer.norm1.weight.clone()
            new_layer.norm1.bias.data = orig_layer.norm1.bias.clone()
            new_layer.norm2.weight.data = orig_layer.norm2.weight.clone()
            new_layer.norm2.bias.data = orig_layer.norm2.bias.clone()
            new_layer.norm3.weight.data = orig_layer.norm3.weight.clone()
            new_layer.norm3.bias.data = orig_layer.norm3.bias.clone()

    def forward(self, vision_embeded):
        # Student decoder
        memory = self.norm_input(self.input_proj(vision_embeded))
        batch_size = memory.shape[0]
        tgt = self.planning_query.expand(batch_size, -1, -1)
        out = self.transformer_decoder(tgt, memory)
        predicted_states = self.norm_output(self.output_proj(out))
        ego_feature = predicted_states.squeeze(1)

        # VAE Planner - distribution_forward (inference mode, no future_distribution)
        current_states = ego_feature.unsqueeze(1)
        present_sample, _ = self.present_distribution(current_states)
        sample = present_sample

        # future_states_predict
        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
        future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)

        hidden_states = ego_feature.unsqueeze(1).permute(1, 0, 2)
        hidden_state = hidden_states.reshape(self.layer_dim, -1, 4096 // self.layer_dim)
        future_states = self.predict_model(future_prediction_input, hidden_state)

        current_states_hs = current_states.unsqueeze(0).repeat(self.fut_ts, 1, 1, 1)
        future_states_hs = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2])

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
        else:
            states_hs = future_states_hs

        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)

        # Unrolled ego_fut_decoder loop
        ego_fut_trajs_list = []
        for i in range(self.fut_ts):
            x = ego_query_hs[i]
            for layer in self.ego_fut_decoder_layers:
                x = layer(x)
            pred_i = x.reshape(batch_size, self.ego_fut_mode, 2)
            ego_fut_trajs_list.append(pred_i)

        ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)
        return ego_feature, ego_fut_preds


def load_student_weights(ckpt_path, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    student_state = {}
    for k, v in state_dict.items():
        if k.startswith("student_model."):
            student_state[k[len("student_model."):]] = v
    return student_state


def export_onnx(ckpt_path, output_dir, fp16=False, opset=17, seq_len=513):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Orion-Lite Student Model → ONNX Export")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading student model...")
    student_state = load_student_weights(ckpt_path, device="cpu")
    model = OrionMimicModel(
        input_dim=4096, hidden_dim=1024, output_dim=4096,
        num_layers=6, num_heads=16, dropout=0.1,
        with_bound_loss=True, use_col_loss=False,
    )
    model.load_state_dict(student_state, strict=False)
    model.eval()

    dtype = torch.float16 if fp16 else torch.float32
    model = model.to(dtype=dtype)

    proxy = StudentModelONNXProxy(model)
    proxy.eval()

    # Create dummy input
    B, N = 1, seq_len
    vision_embeded = torch.randn(B, N, 4096, dtype=dtype)

    # Verify forward pass
    print("\n[2/4] Verifying forward pass...")
    with torch.no_grad():
        ego_feature, ego_fut_preds = proxy(vision_embeded)
    print(f"  ego_feature: {ego_feature.shape}")
    print(f"  ego_fut_preds: {ego_fut_preds.shape}")

    # Export ONNX
    onnx_path = os.path.join(output_dir, "orion_lite_student.onnx")
    print(f"\n[3/4] Exporting ONNX to: {onnx_path}")
    print(f"  opset={opset}, fp16={fp16}")

    torch.onnx.export(
        proxy,
        (vision_embeded,),
        onnx_path,
        input_names=["vision_embeded"],
        output_names=["ego_feature", "ego_fut_preds"],
        opset_version=opset,
        do_constant_folding=True,
        # Fixed shapes for TensorRT compatibility
        dynamic_axes=None,
        verbose=False,
    )

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ONNX size: {onnx_size_mb:.1f} MB")

    # Try onnxsim (skip if fails due to custom ops)
    try:
        import onnxsim
        import onnx

        sim_path = onnx_path.replace(".onnx", "_sim.onnx")
        print(f"\n  Running onnxsim...")
        onnx_model = onnx.load(onnx_path)
        onnx_model, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(onnx_model, sim_path)
            sim_size_mb = os.path.getsize(sim_path) / (1024 * 1024)
            print(f"  Simplified ONNX: {sim_path} ({sim_size_mb:.1f} MB)")
            onnx_path = sim_path
        else:
            print(f"  onnxsim check failed, using original")
    except Exception as e:
        print(f"  onnxsim failed: {e}, using original ONNX")

    # Save test data as numpy
    print(f"\n[4/4] Saving test data (numpy)...")
    npy_dir = os.path.join(output_dir, "test_data")
    os.makedirs(npy_dir, exist_ok=True)

    # Generate multiple test samples
    np.random.seed(42)
    torch.manual_seed(42)

    for i in range(5):
        ve = torch.randn(B, N, 4096, dtype=dtype)
        with torch.no_grad():
            ego_feat, ego_fut = proxy(ve)

        np.save(os.path.join(npy_dir, f"vision_embeded_{i}.npy"),
                ve.cpu().float().numpy())
        np.save(os.path.join(npy_dir, f"ego_feature_ref_{i}.npy"),
                ego_feat.cpu().float().numpy())
        np.save(os.path.join(npy_dir, f"ego_fut_preds_ref_{i}.npy"),
                ego_fut.cpu().float().numpy())

    print(f"  Saved 5 test samples to: {npy_dir}")

    # Save model info
    info = {
        "onnx_path": onnx_path,
        "onnx_size_mb": onnx_size_mb,
        "fp16": fp16,
        "input_shape": [B, N, 4096],
        "output_shapes": {
            "ego_feature": list(ego_feature.shape),
            "ego_fut_preds": list(ego_fut_preds.shape),
        },
        "test_data_dir": npy_dir,
        "test_samples": 5,
        "trtexec_command": generate_trtexec_command(onnx_path, output_dir),
    }

    info_path = os.path.join(output_dir, "export_info.json")
    import json
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Export info saved to: {info_path}")

    print(f"\n{'='*70}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"  ONNX: {onnx_path} ({onnx_size_mb:.1f} MB)")
    print(f"  Test data: {npy_dir}/")
    print(f"\n  Next step: Transfer to Orin and compile with trtexec:")
    print(f"  {info['trtexec_command']}")

    return info


def generate_trtexec_command(onnx_path, output_dir):
    engine_path = onnx_path.replace(".onnx", ".engine")
    if "_sim" in engine_path:
        pass
    else:
        engine_path = os.path.join(output_dir, "orion_lite_student.engine")

    cmd = f"""trtexec \\
  --onnx={onnx_path} \\
  --saveEngine={engine_path} \\
  --fp16 \\
  --useCudaGraph \\
  --verbose"""
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Export Orion-Lite Student Model to ONNX")
    parser.add_argument("--ckpt", type=str,
                        default="../eval/fused_ckpts/orion_lite_subset8.pth")
    parser.add_argument("--output-dir", type=str,
                        default="../eval/fused_ckpts/student_onnx")
    parser.add_argument("--fp16", action="store_true",
                        help="Export with FP16 weights")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seq-len", type=int, default=513,
                        help="Sequence length N of vision_embeded input (default 513, real=529)")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    export_onnx(ckpt_path, args.output_dir, fp16=args.fp16, opset=args.opset, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
