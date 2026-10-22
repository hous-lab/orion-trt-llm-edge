"""
导出 ego_fut_pred_sim.onnx —— Orion 的 planning 轨迹预测头

从 Orion.pth 中提取 distribution_forward + future_states_predict + ego_fut_decoder，
导出为 ONNX，输入是 LLM 的 selected_hidden_states [1, 4096]，输出是 ego_fut_pred [1, 6, 6, 2]。

用法:
    cd $ORION_PROJECT_ROOT
    PYTHONPATH=. $CONDA_PREFIX/bin/python \
        export_ego_fut_pred.py
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv.utils import Config, load_checkpoint, set_random_seed
from mmcv.models import build_model


class EgoFeatureProcessor(nn.Module):
    """从 LLM hidden_states 到 planning 轨迹的预测头"""

    def __init__(self, mod):
        super().__init__()
        # 直接复制原模型的相关子模块引用
        self.distribution_forward = mod.distribution_forward
        self.future_states_predict = mod.future_states_predict
        self.ego_fut_decoder = mod.ego_fut_decoder
        self.fut_ts = mod.fut_ts
        self.ego_fut_mode = mod.ego_fut_mode

    def forward(self, ego_feature):
        """
        Args:
            ego_feature: [1, 4096] — LLM waypoint_token 位置的 hidden_state
        Returns:
            ego_fut_pred: [1, ego_fut_mode, fut_ts, 2] = [1, 6, 6, 2]
        """
        current_states = ego_feature.unsqueeze(1)  # [1, 1, 4096]

        # distribution_forward
        sample, _ = self.distribution_forward(current_states)

        # future_states_predict
        hidden_states = ego_feature.unsqueeze(1)  # [1, 1, 4096]
        states_hs, _ = self.future_states_predict(
            1,  # batch_size
            sample,
            hidden_states,
            current_states
        )

        # ego_fut_decoder（每个时间步预测一个轨迹）
        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
        ego_fut_trajs_list = []
        for i in range(self.fut_ts):
            outputs_ego_trajs = self.ego_fut_decoder(
                ego_query_hs[i]
            ).reshape(1, self.ego_fut_mode, 2)  # [1, 6, 2]
            ego_fut_trajs_list.append(outputs_ego_trajs)

        ego_fut_pred = torch.stack(ego_fut_trajs_list, dim=2)  # [1, 6, 6, 2]
        return ego_fut_pred


def main():
    config_path = 'adzoo/orion/configs/orion_stage3.py'
    ckpt_path = 'ckpts/Orion.pth'
    output_dir = 'ckpts/vision_onnx_only'
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Loading config: {config_path}")
    cfg = Config.fromfile(config_path)

    # 禁用 flash_attn 和 checkpoint
    cfg.model.img_backbone["flash_attn"] = False
    cfg.model.img_backbone["with_cp"] = False
    cfg.model.pts_bbox_head["transformer"]["flash_attn"] = False
    cfg.model.pts_bbox_head["transformer"]["with_cp"] = False
    if hasattr(cfg.model.pts_bbox_head, 'motion_transformer_decoder'):
        cfg.model.pts_bbox_head["motion_transformer_decoder"]["flash_attn"] = False
        cfg.model.pts_bbox_head["motion_transformer_decoder"]["with_cp"] = False
    if hasattr(cfg.model.pts_bbox_head, 'memory_decoder_transformer'):
        cfg.model.pts_bbox_head["memory_decoder_transformer"]["flash_attn"] = False
        cfg.model.pts_bbox_head["memory_decoder_transformer"]["with_cp"] = False
    cfg.model.map_head["transformer"]["flash_attn"] = False
    cfg.model.map_head["transformer"]["with_cp"] = False

    print(f"[2/4] Building model and loading checkpoint: {ckpt_path}")
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # fp16 wrapping（与 export_vision_orion_wolm.py 一致）
    custom_fp16 = dict(map_head=False, pts_bbox_head=False)
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for module_name, v in custom_fp16.items():
        model._modules[module_name].fp16_enabled = v

    load_checkpoint(model, ckpt_path, map_location='cpu')
    model.eval()

    print(f"[3/4] Creating EgoFeatureProcessor")
    ego_processor = EgoFeatureProcessor(model)
    ego_processor.eval()

    # 验证 forward
    with torch.no_grad():
        dummy_input = torch.randn(1, 4096, dtype=torch.float32)
        dummy_output = ego_processor(dummy_input)
        print(f"  Verification: input={dummy_input.shape}, output={dummy_output.shape}")
        assert dummy_output.shape == (1, 6, 6, 2), f"Unexpected output shape: {dummy_output.shape}"

    print(f"[4/4] Exporting ONNX")
    ego_onnx_path = os.path.join(output_dir, "ego_fut_pred.onnx")
    torch.onnx.export(
        ego_processor,
        (dummy_input,),
        ego_onnx_path,
        input_names=["ego_feature"],
        output_names=["ego_fut_pred"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,  # 固定 shape
        verbose=False
    )
    print(f"  Saved: {ego_onnx_path} ({os.path.getsize(ego_onnx_path) / 1024**2:.1f} MB)")
    print(f"\nDone! Next step:")
    print(f"  /usr/src/TensorRT-10.4.0.26/bin/trtexec \\")
    print(f"    --onnx={ego_onnx_path} \\")
    print(f"    --saveEngine={output_dir}/ego_fut_pred_sim.engine \\")
    print(f"    --skipInference --fp16 --precisionConstraints=obey \\")
    print(f"    --layerPrecisions=*_FORCEFP32:fp32")


if __name__ == '__main__':
    main()