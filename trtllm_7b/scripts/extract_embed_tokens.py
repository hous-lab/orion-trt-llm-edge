"""
从 Orion.pth 提取 LlavaLlama 的 embed_tokens 权重。

根因：save_llm_checkpoint.py 因 PeftModel key 不匹配，
load_checkpoint 静默跳过了 embed_tokens，导致保存的 safetensors
里 embed_tokens 是 pretrain_qformer 初始值（错的）且只有 32000 行。

本脚本绕过 save_llm_checkpoint.py 的 bug，直接从 Orion.pth 提取
正确的 embed_tokens (32001, 4096)，含真实 waypoint_token embedding。

Usage:
    $CONDA_PREFIX/bin/python extract_embed_tokens.py
"""
import torch
import os

ORION_PTH = 'ckpts/Orion.pth'
OUTPUT_PATH = 'ckpts/embed_tokens_orion_fp16.pt'
EXPECTED_KEY = 'lm_head.base_model.model.model.embed_tokens.weight'

def main():
    print(f"Loading {ORION_PTH} (38GB, may take a few minutes)...")
    sd = torch.load(ORION_PTH, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    if EXPECTED_KEY not in sd:
        raise KeyError(f"'{EXPECTED_KEY}' not found in Orion.pth. "
                       f"Available embed_tokens keys: "
                       f"{[k for k in sd.keys() if 'embed_tokens' in k][:5]}")

    emb = sd[EXPECTED_KEY]
    print(f"Found: {EXPECTED_KEY}")
    print(f"  shape: {tuple(emb.shape)}, dtype: {emb.dtype}")
    print(f"  BOS(1)  first 5: {emb[1, :5].float().tolist()}")
    print(f"  EOS(2)  first 5: {emb[2, :5].float().tolist()}")
    print(f"  WP(32000) first 5: {emb[32000, :5].float().tolist()}")
    print(f"  mean={emb.float().mean().item():.6f}, std={emb.float().std().item():.6f}")

    # 转为 fp16 保存（与 TRT engine 推理精度一致）
    emb_fp16 = emb.half().contiguous()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(emb_fp16, OUTPUT_PATH)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"  shape: {tuple(emb_fp16.shape)}, dtype: {emb_fp16.dtype}")
    print(f"  size: {os.path.getsize(OUTPUT_PATH) / 1024**2:.1f} MB")

    # 验证保存的文件
    loaded = torch.load(OUTPUT_PATH, map_location='cpu')
    assert loaded.shape == (32001, 4096), f"Shape mismatch: {loaded.shape}"
    assert loaded.dtype == torch.float16, f"Dtype mismatch: {loaded.dtype}"
    assert torch.allclose(loaded, emb_fp16), "Content mismatch"
    print("  Verification PASSED.")

if __name__ == '__main__':
    main()
