"""
Run from the training-repo root (cwd = ORION_PROJECT_ROOT)
python adzoo/orion/merge_lora_completely.py \
    --config adzoo/orion/configs/orion_stage3_fp16.py \
    --checkpoint ckpts/Orion.pth \
    --llm_pretrain_path ckpts/hf_models/pretrain_qformer/ \
    --output_dir ckpts/llm_clean_no_lora/

"""

import torch
from transformers import AutoTokenizer
from mmcv.utils.llava_llama import LlavaLlamaForCausalLM
from peft import PeftModel
from mmcv.utils import Config, load_checkpoint
from mmcv.models import build_model

def load_orion_model_with_lora(config_path, checkpoint_path, llm_pretrain_path):
    """加载 Orion 模型（带 LoRA）"""
    cfg = Config.fromfile(config_path)
    cfg.model.train_cfg = None
    
    # 构建 Orion 模型
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载 LLM（带 LoRA）
    from adzoo.orion.save_llm_checkpoint import load_model
    lm_kwargs = dict(use_gen_token=True, use_critical_qa=False)
    model.lm_head = load_model(
        llm_pretrain_path,
        use_lora=True,
        frozen=False,
        lm_kwargs=lm_kwargs,
        fp16_infer=True
    )
    
    # 加载训练后的权重
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    return model.lm_head

def extract_and_merge_lora(peft_model):
    """
    从 PeftModel 中提取并合并 LoRA 权重
    返回不含 LoRA 结构的纯净模型
    """
    print("=" * 60)
    print("Extracting and merging LoRA weights...")
    print("=" * 60)
    
    # 检查模型结构
    if not isinstance(peft_model, PeftModel):
        print(f"Model type: {type(peft_model)}")
        # 尝试从嵌套结构中提取
        if hasattr(peft_model, 'base_model'):
            print("Found base_model attribute")
            peft_model = peft_model.base_model
    
    # 使用 PEFT 的 merge_and_unload
    print("Calling merge_and_unload()...")
    merged_model = peft_model.merge_and_unload()
    
    # 验证
    has_lora = any('lora' in name.lower() for name, _ in merged_model.named_modules())
    print(f"After merge_and_unload: has_lora = {has_lora}")
    
    if has_lora:
        print("⚠️ WARNING: Model still has LoRA layers!")
        print("Attempting manual cleanup...")
        
        # 手动移除 LoRA 层
        def remove_lora_layers(module):
            for name, child in list(module.named_children()):
                if hasattr(child, 'base_layer'):
                    # 这是一个 lora.Linear，替换为 base_layer
                    print(f"  Replacing {name} with base_layer")
                    setattr(module, name, child.base_layer)
                else:
                    remove_lora_layers(child)
        
        remove_lora_layers(merged_model)
        
        # 再次验证
        has_lora = any('lora' in name.lower() for name, _ in merged_model.named_modules())
        print(f"After manual cleanup: has_lora = {has_lora}")
    
    return merged_model

def verify_weights(model):
    """验证权重是否正常"""
    print("\n" + "=" * 60)
    print("Verifying weights...")
    print("=" * 60)
    
    # 检查一些关键层
    key_layers = [
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.self_attn.k_proj.weight',
        'model.layers.0.self_attn.v_proj.weight',
        'model.layers.0.self_attn.o_proj.weight',
    ]
    
    state_dict = model.state_dict()
    
    for key in key_layers:
        if key in state_dict:
            weight = state_dict[key]
            print(f"✅ {key}: shape={weight.shape}, dtype={weight.dtype}")
            print(f"   mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
        else:
            print(f"❌ {key}: NOT FOUND!")
            # 尝试找到相似的键
            similar = [k for k in state_dict.keys() if 'q_proj' in k][:3]
            print(f"   Similar keys: {similar}")
    
    # 列出所有权重键
    print(f"\nTotal weights: {len(state_dict)}")
    print("First 20 keys:")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        print(f"  {key}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--llm_pretrain_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    # 1. 加载模型（带 LoRA）
    print("Loading Orion model with LoRA...")
    peft_model = load_orion_model_with_lora(
        args.config,
        args.checkpoint,
        args.llm_pretrain_path
    )
    
    # 2. 提取并合并 LoRA
    merged_model = extract_and_merge_lora(peft_model)
    
    # 3. 验证权重
    verify_weights(merged_model)
    
    # 4. 保存
    print("\n" + "=" * 60)
    print(f"Saving to {args.output_dir}...")
    print("=" * 60)
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
        # 显式保存关键配置，覆盖默认值
        use_gen_token=True,
        vocab_size=32001 if hasattr(merged_model, 'use_gen_token') and merged_model.use_gen_token else 32000
    )

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_pretrain_path)
    tokenizer.save_pretrained(args.output_dir)
    
    print("✅ Done!")
    
    # 5. 重新加载验证
    print("\n" + "=" * 60)
    print("Reloading to verify...")
    print("=" * 60)
    
    # 关键：传递 use_gen_token=True，确保 weighted_mask 形状为 32001（与之前一致）
    reloaded = LlavaLlamaForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch.float16,
        device_map='cpu',
        # ignore_mismatched_sizes=True,  # 忽略形状不匹配的张量
        # exclude_keys=["weighted_mask"],  # 显式排除 weighted_mask，不参与加载检查
        trust_remote_code=True,  # 若需要加载自定义模型，添加此参数
        use_gen_token=True  # 与训练时一致，确保 weighted_mask 形状正确
    )
    
    has_lora_final = any('lora' in n.lower() for n, _ in reloaded.named_modules())
    print(f"Reloaded model has_lora: {has_lora_final}")
    
    if has_lora_final:
        print("❌ FAILED: Model still has LoRA!")
    else:
        print("✅ SUCCESS: Clean model saved!")

if __name__ == '__main__':
    main()