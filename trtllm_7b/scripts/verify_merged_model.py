# 诊断脚本：verify_merged_model.py
import torch
from transformers import AutoTokenizer
from mmcv.utils.llava_llama import LlavaLlamaForCausalLM

# 1. 加载合并后的模型（你刚刚保存的）
print("=" * 60)
print("Loading merged model...")
print("=" * 60)

merged_model = LlavaLlamaForCausalLM.from_pretrained(
    "ckpts/llm_ckpt_fp16/",
    torch_dtype=torch.float16,
    device_map='cuda:3'
)
merged_model.eval()

tokenizer = AutoTokenizer.from_pretrained("ckpts/llm_ckpt_fp16/")

# 2. 准备相同的输入（使用你实际的 vision_embeds）
# 这里需要你从 test_trt.py 中提取一个真实的 vision_embeds
# 暂时用随机数模拟
vision_embeds = torch.randn(1, 599, 4096, dtype=torch.float16, device='cuda:3')

print(f"\nInput vision_embeds:")
print(f"  Shape: {vision_embeds.shape}")
print(f"  Mean: {vision_embeds.mean().item():.6f}")
print(f"  Std: {vision_embeds.std().item():.6f}")

# 3. 运行 PyTorch 推理
print("\n" + "=" * 60)
print("Running PyTorch inference...")
print("=" * 60)

with torch.no_grad():
    outputs = merged_model.model(
        inputs_embeds=vision_embeds,
        output_hidden_states=True,
        return_dict=True
    )
    hidden_states = outputs.hidden_states[-1]  # 最后一层

print(f"\nPyTorch hidden_states:")
print(f"  Shape: {hidden_states.shape}")
print(f"  Mean: {hidden_states.mean().item():.6f}")
print(f"  Std: {hidden_states.std().item():.6f}")
print(f"  Last token (first 10 values): {hidden_states[0, -1, :10]}")

# 4. 保存这个作为参考
torch.save({
    'vision_embeds': vision_embeds.cpu(),
    'hidden_states': hidden_states.cpu()
}, 'pytorch_reference.pt')

print("\n✅ Saved to pytorch_reference.pt")