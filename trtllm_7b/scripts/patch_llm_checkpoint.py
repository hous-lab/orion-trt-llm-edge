"""
Patch llm_clean_no_lora 的 safetensors，替换 embed_tokens 和 lm_head 为
Orion.pth 中微调后的正确权重（含 waypoint_token 行，vocab_size=32001）。

根因：save_llm_checkpoint.py 因 PeftModel key 不匹配，load_checkpoint 静默
跳过了 embed_tokens/lm_head，导致保存的 safetensors 里这两个权重仍是
pretrain_qformer 初始值（错的）且只有 32000 行。

修复后输出到 ckpts/llm_clean_no_lora_patched/，可直接用于 convert_llm_checkpoint.py。

Usage:
    $CONDA_PREFIX/bin/python patch_llm_checkpoint.py
"""
import os
import json
import shutil
import torch
from safetensors.torch import load_file, save_file

ORION_PTH = 'ckpts/Orion.pth'
SRC_DIR = 'ckpts/llm_clean_no_lora'
DST_DIR = 'ckpts/llm_clean_no_lora_patched'

# Orion.pth 中的 key（PeftModel 包裹下的路径）
ORION_EMBED_KEY = 'lm_head.base_model.model.model.embed_tokens.weight'  # (32001, 4096)
ORION_LM_HEAD_KEY = 'lm_head.base_model.model.lm_head.weight'           # (32001, 4096)

# safetensors 中的 key
ST_EMBED_KEY = 'model.embed_tokens.weight'
ST_LM_HEAD_KEY = 'lm_head.weight'


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # ===== 1. 加载 Orion.pth 提取正确权重 =====
    print(f"Loading {ORION_PTH} (38GB, may take a few minutes)...")
    sd = torch.load(ORION_PTH, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    orion_embed = sd[ORION_EMBED_KEY].half().contiguous()  # (32001, 4096) fp16
    orion_lm_head = sd[ORION_LM_HEAD_KEY].half().contiguous()  # (32001, 4096) fp16
    del sd

    print(f"Orion embed_tokens: {tuple(orion_embed.shape)}, BOS(1)[:3]={orion_embed[1,:3].tolist()}")
    print(f"Orion lm_head:      {tuple(orion_lm_head.shape)}, row1[:3]={orion_lm_head[1,:3].tolist()}")

    # ===== 2. 复制源目录所有文件到目标目录 =====
    print(f"\nCopying {SRC_DIR} -> {DST_DIR}")
    for f in os.listdir(SRC_DIR):
        src = os.path.join(SRC_DIR, f)
        dst = os.path.join(DST_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  copied: {f}")

    # ===== 3. Patch model-00001-of-00003.safetensors（embed_tokens）=====
    shard1 = os.path.join(DST_DIR, 'model-00001-of-00003.safetensors')
    print(f"\nPatching {os.path.basename(shard1)} (embed_tokens)...")
    state = load_file(shard1)
    old_emb = state[ST_EMBED_KEY]
    print(f"  old embed_tokens: {tuple(old_emb.shape)}, BOS(1)[:3]={old_emb[1,:3].tolist()}")
    state[ST_EMBED_KEY] = orion_embed
    save_file(state, shard1, metadata={'format': 'pt'})
    print(f"  new embed_tokens: {tuple(orion_embed.shape)}, BOS(1)[:3]={orion_embed[1,:3].tolist()}")
    del state

    # ===== 4. Patch model-00003-of-00003.safetensors（lm_head）=====
    shard3 = os.path.join(DST_DIR, 'model-00003-of-00003.safetensors')
    print(f"\nPatching {os.path.basename(shard3)} (lm_head)...")
    state = load_file(shard3)
    old_lh = state[ST_LM_HEAD_KEY]
    print(f"  old lm_head: {tuple(old_lh.shape)}, row1[:3]={old_lh[1,:3].tolist()}")
    state[ST_LM_HEAD_KEY] = orion_lm_head
    save_file(state, shard3, metadata={'format': 'pt'})
    print(f"  new lm_head: {tuple(orion_lm_head.shape)}, row1[:3]={orion_lm_head[1,:3].tolist()}")
    del state

    # ===== 5. 修改 config.json: vocab_size 32000 -> 32001 =====
    cfg_path = os.path.join(DST_DIR, 'config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    old_vs = cfg.get('vocab_size', 'N/A')
    cfg['vocab_size'] = 32001
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"\nconfig.json: vocab_size {old_vs} -> 32001")

    # ===== 6. 更新 model.safetensors.index.json 的 total_size =====
    idx_path = os.path.join(DST_DIR, 'model.safetensors.index.json')
    with open(idx_path, 'r') as f:
        idx = json.load(f)
    # total_size 需要反映新增的 2 行（embed_tokens + lm_head 各加 1 行 * 4096 * 2 bytes）
    # 旧: 32000*4096*2 = 262144000 bytes per tensor
    # 新: 32001*4096*2 = 262212096 bytes per tensor
    # 增加: (32001-32000)*4096*2 = 8192 bytes per tensor, 两个 tensor 共 16384 bytes
    old_total = idx['metadata']['total_size']
    new_total = old_total + 2 * (32001 - 32000) * 4096 * 2  # 两个 tensor 各加 1 行
    idx['metadata']['total_size'] = new_total
    with open(idx_path, 'w') as f:
        json.dump(idx, f, indent=2)
    print(f"index.json: total_size {old_total} -> {new_total} (+{new_total-old_total} bytes)")

    # ===== 7. 验证 =====
    print("\n=== Verification ===")
    for shard, key in [(shard1, ST_EMBED_KEY), (shard3, ST_LM_HEAD_KEY)]:
        with safe_open_verify(shard, key) as t:
            pass

    print(f"\n✅ Patched checkpoint saved to: {DST_DIR}")
    print(f"   Next steps:")
    print(f"   1. convert_llm_checkpoint.py --model_dir {DST_DIR} --output_dir {DST_DIR}/trt --dtype float16")
    print(f"   2. trtllm-build --checkpoint_dir {DST_DIR}/trt --output_dir <engine_dir> ...")


def safe_open_verify(shard_path, key):
    """验证 patch 后的 tensor shape 和值"""
    from safetensors import safe_open
    with safe_open(shard_path, framework='pt') as st:
        t = st.get_tensor(key)
        print(f"  {os.path.basename(shard_path)} | {key}: shape={tuple(t.shape)}, dtype={t.dtype}")
        if t.shape[0] >= 2:
            print(f"    row 1 [:3] = {t[1,:3].tolist()}")
        if t.shape[0] >= 32001:
            print(f"    row 32000 [:3] = {t[32000,:3].tolist()}")
        assert t.shape[0] == 32001, f"Expected vocab_size=32001, got {t.shape[0]}"
    return None


if __name__ == '__main__':
    main()
