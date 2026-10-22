import argparse
import os
import torch
from mmcv.utils import wrap_fp16_model, set_random_seed, Config, DictAction, load_checkpoint
from mmcv.models import build_model
from mmcv.utils.llava_llama import LlavaLlamaForCausalLM, add_special_token
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer

def load_model(base_model, use_lora=False, frozen=False, lm_kwargs=dict(), fp16_infer=False):
    """加载带LoRA的LlavaLlamaForCausalLM模型"""
    dtype = torch.float16 if fp16_infer else torch.float32
    
    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model, 
        torch_dtype=dtype, 
        device_map='cpu', 
        **lm_kwargs
    )
    
    model.gradient_checkpointing_enable()
    
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    
    if use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
    
    return model

def merge_lora_to_base(model):
    """强制合并LoRA权重，非PeftModel直接报错（确保合并逻辑有效）"""
    print("🔄 Merging LoRA weights into base model...")
    
    if not isinstance(model, PeftModel):
        raise RuntimeError("❌ 模型不是PeftModel，无法合并LoRA权重！请检查use_lora是否为True及模型加载逻辑")
    
    merged_model = model.merge_and_unload()
    print("✅ LoRA weights merged successfully")
    return merged_model

def check_weight_consistency(orig_model, merged_model):
    """对比合并前后关键层权重差异，确保合并正确（核心验证步骤）"""
    print("\n📊 验证合并前后权重一致性...")
    
    # 提取原模型（带LoRA）动态融合后的权重（q_proj为核心注意力层，LoRA必修改）
    orig_q_proj = orig_model.base_model.model.model.layers[0].self_attn.q_proj.weight.data
    # 提取合并后模型的权重
    merged_q_proj = merged_model.model.layers[0].self_attn.q_proj.weight.data
    
    # 计算绝对值均值差异（float16下允许微小误差，阈值设为1e-3）
    diff = torch.mean(torch.abs(orig_q_proj - merged_q_proj)).item()
    print(f"第0层q_proj权重差异（绝对值均值）：{diff:.6f}")
    
    if diff > 1e-3:
        raise RuntimeError(f"❌ 权重差异过大（{diff} > 1e-3），合并失败！")
    print("✅ 权重一致性验证通过")

EGO_WAYPOINT_TOKEN = "<waypoint_ego>"

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    
    # ===== 步骤1：构建Orion模型，加载带LoRA的LLM头 =====
    print("=" * 50)
    print("Building Orion model with LoRA...")
    print("=" * 50)
    
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载带LoRA的lm_head（use_lora=True强制启用）
    lm_kwargs = dict(use_gen_token=True, use_critical_qa=False)
    orig_lm_head = load_model(
        args.llm_pretrain_path, 
        use_lora=True,
        frozen=False, 
        lm_kwargs=lm_kwargs, 
        fp16_infer=args.fp16_infer
    )
    model.lm_head = orig_lm_head
    
    # 添加特殊token（<waypoint_ego>），与原模型保持一致
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_pretrain_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    add_special_token([EGO_WAYPOINT_TOKEN], tokenizer=tokenizer, model=model.lm_head)
    model.lm_head.config.waypoint_token_idx = tokenizer(
        EGO_WAYPOINT_TOKEN, add_special_tokens=False
    ).input_ids[0]
    
    # 适配FP16推理（若启用）
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    # ===== 步骤2：加载checkpoint（含LoRA权重）=====
    print("\n" + "=" * 50)
    print("Loading checkpoint...")
    print("=" * 50)
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    print("✅ Checkpoint loaded")
    
    # 校验LoRA参数是否存在（必须>0，否则合并无意义）
    lora_count = sum(1 for name, _ in model.lm_head.named_parameters() if 'lora' in name.lower())
    print(f"Found {lora_count} LoRA parameters")
    if lora_count == 0:
        raise RuntimeError("❌ 未找到LoRA参数！请检查checkpoint是否正确包含LoRA权重")
    
    # ===== 步骤3：合并LoRA权重（强制校验）=====
    print("\n" + "=" * 50)
    print("Merging LoRA weights...")
    print("=" * 50)
    
    merged_model = merge_lora_to_base(orig_lm_head)
    
    # ===== 步骤4：验证合并权重一致性（关键步骤，避免合并失效）=====
    check_weight_consistency(orig_lm_head, merged_model)
    
    # ===== 步骤5：保存合并后的模型（含tokenizer）=====
    print("\n" + "=" * 50)
    print("Saving merged model...")
    print("=" * 50)
    
    os.makedirs(args.save_checkpoint_pth, exist_ok=True)
    # 保存主模型（safe_serialization=True确保权重完整性）
    merged_model.save_pretrained(args.save_checkpoint_pth, safe_serialization=True)
    # 保存tokenizer（与模型配套，确保后续推理一致）
    tokenizer.save_pretrained(args.save_checkpoint_pth)
    
    print(f"✅ 合并后的模型已保存至：{args.save_checkpoint_pth}")
    
    # ===== 步骤6：最终验证（确保无LoRA残留，合并完全）=====
    print("\n" + "=" * 50)
    print("Final verification (no LoRA残留)...")
    print("=" * 50)
    
    # 重新加载合并后的模型，验证结构
    verify_model = LlavaLlamaForCausalLM.from_pretrained(
        args.save_checkpoint_pth,
        torch_dtype=torch.float16 if args.fp16_infer else torch.float32,
        device_map='cpu'
    )
    
    # 检查是否存在LoRA相关模块/参数（必须无残留）
    has_lora = any(
        'lora' in name.lower() for name, _ in verify_model.named_modules()
    ) or any(
        'lora' in name.lower() for name, _ in verify_model.named_parameters()
    )
    
    if has_lora:
        raise RuntimeError("❌ 合并后的模型仍存在LoRA相关层/参数！合并失败")
    print("✅ 最终验证通过：无LoRA残留，模型结构纯净")
    
    # ===== 后续步骤提示（适配Orion模型转换流程）=====
    print("\n" + "=" * 50)
    print("Done! Next steps (Orion模型专属流程)：")
    print("=" * 50)
    print(f"1. 转换模型格式：python adzoo/orion/convert_llm_checkpoint.py \\")
    print(f"      --model_dir {args.save_checkpoint_pth} \\")
    print(f"      --output_dir {args.save_checkpoint_pth}/trt \\")
    print(f"      --dtype {'float16' if args.fp16_infer else 'float32'}")
    print(f"2. 构建TRT引擎：trtllm-build --checkpoint_dir {args.save_checkpoint_pth}/trt \\")
    print(f"      --output_dir trt_engine \\")
    print(f"      --model_type llama \\")
    print(f"      --max_batch_size 1 \\")
    print(f"      --max_input_len 600 \\")  # 适配你的输入序列长度（599）
    print(f"      --max_seq_len 1024 \\")
    print(f"      --gemm_plugin float16 \\")
    print(f"      --enable_debug_output \\")
    print(f"      --remove_input_padding disable")

def parse_args():
    parser = argparse.ArgumentParser(description='Save Orion LLM checkpoint with LoRA merged (适配无vision_proj结构)')
    parser.add_argument('--config', help='Orion模型配置文件路径', required=True)
    parser.add_argument('--checkpoint', help='含LoRA权重的checkpoint路径', required=True)
    parser.add_argument('--llm_pretrain_path', help='基础LLM预训练模型路径', required=True)
    parser.add_argument('--save_checkpoint_pth', help='合并后模型保存路径', required=True)
    parser.add_argument('--fp16_infer', action='store_true', help='是否使用FP16推理（与原模型一致）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--deterministic', action='store_true', help='是否启用确定性推理')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置文件参数')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()