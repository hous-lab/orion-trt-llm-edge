import argparse
import warnings
import os
import copy

import torch
# from torch.nn import DataParallel
# from torch.nn.parallel.distributed import DistributedDataParallel

import mmcv
from mmcv.utils import get_dist_info, init_dist, wrap_fp16_model, set_random_seed, Config, DictAction, load_checkpoint
from mmcv.models import build_model, fuse_conv_bn

from mmcv.datasets import build_dataset, build_dataloader, replace_ImageToTensor
from mmcv.models.utils import DistributionModule, PredictModel
from mmcv.models.utils.transformer import inverse_sigmoid
from mmcv.core import bbox3d2result
# from mmcv import NMSFreeCoder, CustomNMSFreeCoder
from mmcv.core.bbox.coder import CustomNMSFreeCoder

from adzoo.orion.apis.test import custom_multi_gpu_test, single_gpu_test, single_gpu_test_trt

import time
import tensorrt as trt
import pycuda.driver as cuda
import os.path as osp
import numpy as np
import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer
import json

from scipy.optimize import linear_sum_assignment
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric

from pathlib import Path
from tensorrt_llm.runtime import Session, ModelConfig
from mmcv.utils.llava_llama import add_special_token
from mmcv.utils.misc import load_model

# 修改 CustomModelRunner 类
class CustomModelRunner(ModelRunner):
    def _prepare_inputs(self, batch_input_ids, paddings=None):
        """确保所有 tensor 在正确的设备上，且格式符合模型要求"""
        # 确定设备（取第一个输入的设备）
        if isinstance(batch_input_ids, (list, tuple)):
            device = batch_input_ids[0].device
        else:
            device = batch_input_ids.device
            # 若输入是单个张量（[batch_size, seq_len]），转为列表方便处理单个样本
            batch_input_ids = list(batch_input_ids)

        # 处理填充（paddings）
        pad_id = getattr(self, '_padding_cache', {}).get(
            'pad_id', torch.tensor(0, dtype=torch.int32, device=device)
        )
        if paddings is None:
            # 计算batch中最长序列长度，统一填充到该长度
            max_input_length = max(ids.shape[-1] for ids in batch_input_ids)
            paddings = []
            for ids in batch_input_ids:
                pad_len = max_input_length - ids.shape[-1]
                # 生成与ids维度匹配的填充（确保是1维：[pad_len]）
                pad = torch.full((pad_len,), pad_id.item(), dtype=torch.int32, device=device)
                paddings.append(pad)
        elif isinstance(paddings, torch.Tensor):
            # 若paddings是单个张量，按batch_size拆分（假设已对齐长度）
            paddings = [paddings[i].to(device) for i in range(len(batch_input_ids))]
        elif isinstance(paddings, int):
            # 若paddings是整数（填充长度），生成对应填充
            max_input_length = max(ids.shape[-1] for ids in batch_input_ids)
            paddings = [
                torch.full((max_input_length - ids.shape[-1],), 0, dtype=torch.int32, device=device)
                for ids in batch_input_ids
            ]
        else:
            paddings = [p.to(device) for p in paddings]

        # 拼接样本和填充，得到每个样本的完整序列（长度均为max_input_length）
        padded_inputs = [
            torch.cat([x, pad], dim=0)  # x: [seq_len], pad: [pad_len] → [max_input_length]
            for x, pad in zip(batch_input_ids, paddings)
        ]

        # 将列表转换为二维张量（[batch_size, max_input_length]）
        batch_input_ids = torch.stack(padded_inputs, dim=0)  # 关键：堆叠成batch张量

        # 计算每个样本的原始长度（未填充前的长度），转为张量（[batch_size]）
        input_lengths = torch.tensor(
            [x.shape[-1] for x in batch_input_ids],  # 收集所有样本的长度（已填充后的长度，或用原始x的长度）
            dtype=torch.int32,
            device=device
        )

        return batch_input_ids, input_lengths  # 此时batch_input_ids是张量，input_lengths是[batch_size]的张量

    @classmethod
    def from_dir(cls, model_dir, rank=0, debug_mode=False, stream=None):
        model_runner = super().from_dir(model_dir, rank=rank, debug_mode=debug_mode, stream=stream)
        model_runner.__class__ = cls
        return model_runner

# generate ego token
EGO_WAYPOINT_TOKEN = "<waypoint_ego>"
class InferTrtLLM(object):
    def __init__(self, llm_engine_pth, tokenizer_pth) -> None:
        device_id = 0
        self.IMAGE_TOKEN_INDEX = -200
        self.llm_engine_pth = llm_engine_pth

        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)
        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, model_max_length=2048, padding_side="right", use_fast=False,)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # 加载 embed_tokens：必须用 Orion.pth 微调后的权重，不能用 pretrain_qformer 初始值
        # 根因：save_llm_checkpoint.py 因 PeftModel key 不匹配，load_checkpoint 静默跳过
        # embed_tokens，导致 pretrain_qformer 的初始 embed_tokens（错值）被编译进 TRT engine。
        # 这里用从 Orion.pth 提取的正确 embed_tokens (32001, 4096) fp16，含真实 waypoint embedding。
        # 提取脚本：extract_embed_tokens.py
        embed_tokens_path = 'ckpts/embed_tokens_orion_fp16.pt'
        print(f"Loading embed_tokens from {embed_tokens_path}...")
        embed_weight = torch.load(embed_tokens_path, map_location='cpu')
        print(f"  embed_tokens: shape={tuple(embed_weight.shape)}, dtype={embed_weight.dtype}")
        print(f"  BOS(1)[:3]={embed_weight[1,:3].tolist()}")
        print(f"  WP(32000)[:3]={embed_weight[32000,:3].tolist()}")
        assert embed_weight.shape == (32001, 4096), \
            f"Expected (32001, 4096), got {tuple(embed_weight.shape)}"
        self.embed_tokens = torch.nn.Embedding.from_pretrained(embed_weight.to(self.device))
        self.embed_tokens.eval()
        print(f"  embed_tokens loaded: {self.embed_tokens.weight.shape}")

        self.model_type = "llava_llama"
        self.vision_proj = None
        self.init_llm()

        # waypoint_token_idx
        self.tokenizer.add_tokens(["<waypoint_ego>"], special_tokens=True)
        self.model_config.waypoint_token_idx = self.tokenizer("<waypoint_ego>", add_special_tokens=False)["input_ids"][0]
    
    def init_llm(self):
        # self.model = ModelRunner.from_dir(str(self.llm_engine_pth), rank=0, debug_mode=True, stream=self.stream)
        self.model = CustomModelRunner.from_dir(str(self.llm_engine_pth), rank=0, debug_mode=True, stream=self.stream)
        self.model_config = self.model.session._model_config
        self.runtime_mapping = self.model.session.mapping

    def image_to_ptuning(self, input_ids, vision_embeded):
        updated_input_ids = []
        current_vocab_size = self.tokenizer.vocab_size
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                updated_input_ids.append(cur_input_ids)
                continue
            im_token_ids = torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0].tolist()
            im_token_ids = [-1] + im_token_ids + [cur_input_ids.shape[0]]
            im_idx = 0
            for i in range(len(im_token_ids) - 1):
                updated_input_ids.append(cur_input_ids[im_token_ids[i]+1:im_token_ids[i+1]].cuda())
                if im_idx < vision_embeded.shape[0]:
                    im = vision_embeded[im_idx]
                    im_size = im.shape[0]
                    im_indices = torch.from_numpy(np.arange(current_vocab_size, current_vocab_size + im_size)).cuda()
                    updated_input_ids.append(im_indices)
                    im_idx += 1
        return torch.cat(updated_input_ids).unsqueeze(0), vision_embeded.reshape(1, -1, vision_embeded.shape[2])


    def prepare_inputs_labels_for_multimodal_(
        self,
        input_ids, # [1, 71]
        image_features # [1, 529, 4096]
    ):
        """
        这个函数已经调试ok，与原版行为一致。251015
        将文本 token embedding 与视觉 patch embedding 交替拼接，返回 (input_embeds, new_input_ids)
        说明：
         - 在 engine 推理流程中无法直接调用 engine 的 embed layer，所以这里使用在初始化时用 load_model 加载的
           PyTorch lm_head（self.lm_head）来获取 token embedding。
         - 若视觉特征维度与语言模型 embedding 维度不一致，会自动建立一个线性投影 self.vision_proj。
        """
        # fallback IGNORE_INDEX（若全局已有定义会被覆盖）
        IGNORE_INDEX = getattr(self, 'IGNORE_INDEX', -100)
        IMAGE_TOKEN_INDEX = getattr(self, 'IMAGE_TOKEN_INDEX', -200)

        # 假设 batch_size = 1
        input_ids = input_ids[0]  # [L]
        image_features = image_features[0]  # [N, D_v]

        device = image_features.device if isinstance(image_features, torch.Tensor) else torch.device('cuda')
        # 1) 准备 labels
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 2) 找到所有 IMAGE_TOKEN_INDEX 的位置
        image_token_indices = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].tolist()
        image_token_indices = [-1] + image_token_indices + [input_ids.shape[0]]

        # 3) 获取语言模型的 embed 层
        embed_module = None
        if hasattr(self, 'embed_tokens') and self.embed_tokens is not None:
            embed_module = self.embed_tokens
        elif hasattr(self, 'lm_head') and getattr(self.lm_head, 'get_input_embeddings', None) is not None:
            embed_module = self.lm_head.get_input_embeddings()
        elif hasattr(self, 'lm_head') and getattr(self.lm_head, 'embed_tokens', None) is not None:
            embed_module = self.lm_head.embed_tokens
        else:
            embed_module = None

        if embed_module is None:
            raise RuntimeError("无法获取 LM embedding 模块，请确保 self.lm_head 已通过 load_model 加载或提供 embedding 层。")

        embed_dim = embed_module.embedding_dim if hasattr(embed_module, 'embedding_dim') else embed_module.weight.shape[1]

        # 4) 如视觉特征维度与 embed_dim 不一致，建立或复用线性投影
        if image_features.shape[1] != embed_dim:
            if not hasattr(self, 'vision_proj') or self.vision_proj is None or self.vision_proj.in_features != image_features.shape[1] or self.vision_proj.out_features != embed_dim:
                # 创建投影，并置为 eval、float16，用于推理拼接
                proj = torch.nn.Linear(image_features.shape[1], embed_dim, bias=True).to(device)
                proj.weight.data.uniform_(-0.02, 0.02)
                proj.bias.data.zero_()
                proj.eval()
                self.vision_proj = proj.half()
            # 投影视觉特征到 lm embed 空间
            vision_embeds = self.vision_proj(image_features.to(device).half()).to(dtype=embed_module.weight.dtype)
        else:
            vision_embeds = image_features.to(device).to(dtype=embed_module.weight.dtype)

        # 5) 依次拼接文本 embedding 和 vision_embeds（注意保持顺序）
        input_embeds_list = []
        new_labels = []
        new_input_ids = []

        for i in range(len(image_token_indices) - 1):
            # 文本块
            text_ids = input_ids[image_token_indices[i]+1:image_token_indices[i+1]]
            text_labels = labels[image_token_indices[i]+1:image_token_indices[i+1]]
            if text_ids.numel() > 0:
                # 确保 indices 与 embedding 在同一 device（避免 index_select 的 cpu/cuda mismatch）
                text_ids = text_ids.to(device)
                if hasattr(embed_module, 'weight'):
                    if embed_module.weight.device != text_ids.device:
                        embed_module = embed_module.to(text_ids.device)
                else:
                    # fallback: 尝试把 module 整体移动到目标 device
                    try:
                        embed_module = embed_module.to(text_ids.device)
                    except Exception:
                        pass
                # 使用 lm_head 的 embedding 模块生成文本 embedding
                text_embeds = embed_module(text_ids)
                input_embeds_list.append(text_embeds)
                new_labels.append(text_labels.to(device))
                new_input_ids.append(text_ids.to(device))

            # image 块：在非末尾 image token 后插入视觉 patch embeddings
            if i < len(image_token_indices) - 2:
                # vision_embeds: [N, D_emb]
                num_vis = vision_embeds.shape[0]
                input_embeds_list.append(vision_embeds)
                new_labels.append(torch.full((num_vis,), IGNORE_INDEX, device=device, dtype=text_labels.dtype))
                # p-tuning: 视觉 token ID 从 32001 开始
                # 原因：waypoint_token_idx=32000 已占用 vocab_embedding[32000]，
                #       视觉 token 必须从 32001 开始，否则第一个视觉 token (32000)
                #       会被 TRT engine 当作 waypoint 用 vocab_embedding 处理。
                #       修复后：waypoint(32000) < vocab_size(32001) → 用 vocab_embedding[32000]（真实 wp embedding）
                #               视觉(32001+) >= vocab_size(32001) → 用 prompt_table
                vocab_size = 32001
                vis_ids = torch.arange(vocab_size, vocab_size + num_vis, device=device, dtype=input_ids.dtype)
                new_input_ids.append(vis_ids)

        # 拼接所有块
        if len(input_embeds_list) == 0:
            # 没有任何有效 token，返回空序列
            return torch.zeros((1,0,embed_dim), device=device, dtype=embed_module.weight.dtype), torch.zeros((1,0), device=device, dtype=input_ids.dtype)

        input_embeds = torch.cat(input_embeds_list, dim=0)
        new_labels = torch.cat(new_labels, dim=0)
        new_input_ids = torch.cat(new_input_ids, dim=0)

        # 6) 加 batch 维度并返回
        input_embeds = input_embeds.unsqueeze(0)  # [1, L', D]
        new_input_ids = new_input_ids.unsqueeze(0)  # [1, L']
        return input_embeds, new_input_ids
    
    def generate(self, input_ids, vision_embeded):
        # input_ids, prompt_table = self.image_to_ptuning(input_ids, vision_embeded)
        prompt_table, input_ids = self.prepare_inputs_labels_for_multimodal_(input_ids, vision_embeded)
        
        input_ids = input_ids.contiguous().to(dtype=torch.int32)
        prompt_table = prompt_table.cuda().contiguous().to(dtype=torch.float16)

        t_start = time.time()
        output_ids = self.model.generate(
            input_ids, 
            prompt_table=prompt_table,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=320,
            use_cache=True
            )
        print(f"Generation time: {time.time() - t_start}s.")
        # 从 debug_buffer 提取 full_hidden_states（替代旧版 debug_buffer["hidden_states"]）
        full_hidden_states = self.model.session.debug_buffer["full_hidden_states"]
        
        # output_ids = torch.masked_select(output_ids, output_ids.lt(self.tokenizer.vocab_size)).reshape([1, -1])
        self.stream.synchronize()

        return output_ids, full_hidden_states
        

    def generate_test(self, inputs_embeds, new_input_ids):
        # p-tuning 方式（修复版：waypoint_token 用真实 embedding，不再替换为 pad）
        target_device = torch.device('cuda:0')
        # vocab_size=32001：waypoint_token(32000) 在 vocab_embedding 内，
        # 视觉 token(32001+) 用 prompt_table 替换。
        # 旧版 vocab_size=32000 且把 waypoint 替换为 0(pad)，导致 TRT engine
        # 在 waypoint 位置用 pad_embedding 而非真实 waypoint_embedding 计算 hidden_state。
        vocab_size = 32001
        wp_idx_val = self.model_config.waypoint_token_idx

        ids_flat = new_input_ids[0]
        wp_positions = (ids_flat == wp_idx_val).nonzero(as_tuple=True)
        wp_pos = wp_positions[0][-1].item() if len(wp_positions[0]) > 0 else -2

        # 视觉 token mask：>= vocab_size (32001) 的位置
        vis_mask = ids_flat >= vocab_size

        vis_embeds = inputs_embeds[0][vis_mask]
        prompt_table = vis_embeds.unsqueeze(0).contiguous().to(device=target_device, dtype=torch.float16)

        # 不再替换 waypoint_token：TRT engine 的 vocab_embedding 已扩展到 32001 行，
        # waypoint_token(32000) 会用 vocab_embedding[32000]（Orion.pth 中真实 waypoint embedding）
        input_ids_int = new_input_ids.contiguous().to(device=target_device, dtype=torch.int32)

        t_start = time.time()
        output_ids = self.model.generate(
            input_ids_int,
            prompt_table=prompt_table,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=1,
            use_cache=True
        )
        print(f"  Generation time: {time.time() - t_start:.2f}s.")

        full_hidden = self.model.session.debug_buffer["full_hidden_states"]
        print(f"  full_hidden_states: {full_hidden.shape}")

        # [hous: debug] 收集逐层 hidden states
        per_layer_hidden = {}
        for key in list(self.model.session.debug_buffer.keys()):
            if key.startswith("hidden_states_layer_"):
                layer_hs = self.model.session.debug_buffer[key]
                per_layer_hidden[key] = layer_hs
        print(f"  per_layer_hidden_states: {len(per_layer_hidden)} layers")

        wp_idx = wp_pos
        if wp_idx >= 0:
            if full_hidden.dim() == 3:
                selected_hidden_states = full_hidden[0:1, wp_idx:wp_idx+1, :].squeeze(0)
            else:
                selected_hidden_states = full_hidden[wp_idx:wp_idx+1, :]
        else:
            selected_hidden_states = full_hidden[-2:-1, :]

        self.stream.synchronize()
        return output_ids, selected_hidden_states, per_layer_hidden


class InferTrt(object):
    def __init__(self, logger, qa_save_path=None, LLM_engine=None, torch_ref_model=None, ego_engine_pth=None):
        self.device_id = 0
        self.cuda_ctx = cuda.Device(self.device_id).retain_primary_context()
        self.cuda_ctx.push()

        self.builder = trt.Builder(logger)
        self.logger = logger
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.opt = self.builder.create_optimization_profile()

        self.config = self.builder.create_builder_config()
        self.config.add_optimization_profile(self.opt)
        # self.config.max_workspace_size = 2 << 34
        self.config.builder_optimization_level = 5
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # self.config.set_flag(trt.BuilderFlag.FP16)  # control this
        self.stream = cuda.Stream()

        # 新增：ego_fut_pred_sim.engine相关初始化
        self.load_ego_engine(ego_engine_pth)

        self.cuda_ctx.pop()

        self.curr_scene_token = None
        self.start_timestamp = None
        self.bindings = {}
        self.bbox_coder = CustomNMSFreeCoder(
            post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num = 300,
            voxel_size = [0.2, 0.2, 8],
            num_classes=9
        )
        self.LLM_engine = LLM_engine
        self.qa_save_path = qa_save_path
        self.torch_ref_model = torch_ref_model
        self.fut_ts = 6
        self.torch_ego_model = None  # PyTorch ego predictor (fallback when ego engine not available)

        # can_bus_embed 修复：TRT vision engine 的 canbus token（vision_embeded[272]）输出错误，
        # 用 PyTorch can_bus_embed 权重替换（根因见 README §6.0.2）
        self.canbus_fixer = None
        try:
            _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            _cfg_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'canbus_fixer.py')
            if os.path.exists(_cfg_pth):
                import importlib.util
                _spec = importlib.util.spec_from_file_location('canbus_fixer', _cfg_pth)
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _orion_pth = os.path.join(_proj_root, 'ckpts', 'Orion.pth')
                # [Orion-Latency] CANBUS_ON_GPU=1 时 canbus_fixer 搬 GPU（S4）
                _cb_dev = 'cuda:0' if os.environ.get('CANBUS_ON_GPU', '0') == '1' else 'cpu'
                self.canbus_fixer = _mod.load_can_bus_fixer(_orion_pth, device=_cb_dev)
                print("[canbus] fixer loaded")
            else:
                print(f"[canbus] fixer not found: {_cfg_pth}")
        except Exception as e:
            print(f"[canbus] fixer load failed: {e}")

    # 新增：加载ego_fut_pred_sim.engine的方法
    def load_ego_engine(self, engine_path):
        self.ego_bindings = {}
        self.ego_names = []
        self.ego_loaded = False
        
        if engine_path is None:
            print("[TensorRT INFO] ego_engine_pth not provided, skipping ego engine loading.")
            print("[TensorRT INFO] Planning evaluation will be disabled.")
            return
        
        print("[TensorRT INFO] Loading ego_fut_pred_sim engine from: ", engine_path)
        self.cuda_ctx.push()
        with open(engine_path, "rb") as fp:
            buf = fp.read()
        runtime = trt.Runtime(self.logger)        
        self.ego_engine = runtime.deserialize_cuda_engine(buf)
        self.ego_context = self.ego_engine.create_execution_context()
        
        # 初始化ego引擎的绑定
        n_io = self.ego_engine.num_io_tensors
        for i in range(n_io):
            name = self.ego_engine.get_tensor_name(i)
            self.ego_names.append(name)
            shape = self.ego_engine.get_tensor_shape(name)
            dtype = self.ego_engine.get_tensor_dtype(name)
            
            # 根据数据类型创建绑定张量
            if str(dtype) == 'DataType.INT32':
                torch_dtype = torch.int32
            else:
                torch_dtype = torch.float32 if str(dtype) == 'DataType.FLOAT' else torch.float16
            
            self.ego_bindings[name] = torch.zeros(list(shape), dtype=torch_dtype, 
                                                device=f"cuda:{self.device_id}").contiguous()
        
        # 打印ego引擎的输入输出信息
        print("##### Ego Engine Input Bindings: ")
        for i in range(n_io):
            name = self.ego_engine.get_tensor_name(i)
            if self.ego_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                print(f"{i}\t{name}\t{self.ego_engine.get_tensor_shape(name)}\t{self.ego_engine.get_tensor_dtype(name)}")
        
        print("##### Ego Engine Output Bindings: ")
        for i in range(n_io):
            name = self.ego_engine.get_tensor_name(i)
            if self.ego_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                print(f"{i}\t{name}\t{self.ego_engine.get_tensor_shape(name)}\t{self.ego_engine.get_tensor_dtype(name)}")
        
        self.ego_loaded = True
        self.cuda_ctx.pop()

    def from_onnx(self, onnx_mod):
        parser = trt.OnnxParser(self.network, self.logger)
        result = parser.parse(onnx_mod.SerializeToString())
        if not result:
            print("failed parsing onnx")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(-1)
        self.buf = self.builder.build_serialized_network(self.network, self.config)
        self._build_engine()
        
    def _build_engine(self):
        self.runtime = trt.Runtime(self.logger)        
        self.engine = self.runtime.deserialize_cuda_engine(self.buf)
        self.context = self.engine.create_execution_context()
        # self.context.profiler = CustomProfiler()
        self.names = []
        n_io = self.engine.num_io_tensors
        for i in range(n_io):
            self.names.append(self.engine.get_tensor_name(i))

    def write(self, path):
        with open(path, "wb") as fp:
            fp.write(self.buf)

    def read(self, path):
        print("[TensorRT INFO] Loading engine from: ", path)
        with open(path, "rb") as fp:
            self.buf = fp.read()
        self._build_engine()
        
    def get_bbox(self, all_cls_scores, all_bbox_preds, all_traj_preds, all_traj_cls_scores, img_metas):
        bbox_results = []
        preds_dicts = self.bbox_coder.decode({
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'all_traj_preds': all_traj_preds,
        })
        num_samples = len(preds_dicts)
        self.num_classes_bbox = 9
        score_threshold = 0.0
        score_threshold = [score_threshold] * self.num_classes_bbox

        bbox_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            # trajs = preds['trajs']

            cur_bboxes = []
            cur_labels = []
            cur_scores = []
            # cur_trajs = []
            for cid in range(self.num_classes_bbox):
                cid_mask = labels == cid
                score_thrs = scores.new_ones(scores.shape) * score_threshold[cid]
                score_mask = scores > score_thrs
                mask = cid_mask & score_mask
                cur_bboxes.append(bboxes[mask].tensor)
                cur_labels.append(labels[mask])
                cur_scores.append(scores[mask])
                # cur_trajs.append(trajs[mask])

            bboxes = torch.cat(cur_bboxes)
            scores = torch.cat(cur_scores)
            labels = torch.cat(cur_labels)
            # trajs =  torch.cat(cur_trajs)
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))

            # bbox_list.append([bboxes, scores, labels, trajs])
            bbox_list.append([bboxes, scores, labels])
            
        # for bboxes, scores, labels, trajs in bbox_list:
        for bboxes, scores, labels in bbox_list:
            bbox_result = bbox3d2result(bboxes, scores, labels)
            # bbox_result['trajs_3d'] = trajs.cpu()
            bbox_results.append(bbox_result)

        return bbox_results

    def get_lane(self, all_lane_cls_one2one, all_lane_preds_one2one, img_metas):
        cls_scores = all_lane_cls_one2one[-1]
        bbox_preds = all_lane_preds_one2one[-1]

        predictions_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            predictions_list.append(self._get_bboxes_single_lane(cls_score, bbox_pred))
        return predictions_list
    
    def _get_bboxes_single_lane(self,
                           cls_score,
                           bbox_pred):
        # 修改为输出50条概率最大的，modify by fuhaoyu
        max_num = 50
        score_threshold = 0.0
        self.num_classes = 6
        self.n_control = 11
        score_threshold = [score_threshold] * self.num_classes

        assert len(cls_score) == len(bbox_pred)
        
        cls_score = cls_score.sigmoid()
        scores, indexs = cls_score.view(-1).topk(max_num)

        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes

        det_bboxes = bbox_pred
        for p in range(self.n_control):
            det_bboxes[..., 3 * p].clamp_(min=-51.2000, max=51.2000)
            det_bboxes[..., 3 * p + 1].clamp_(min=-51.2000, max=51.2000)
            
        # det_bboxes = self.control_points_to_lane_points(det_bboxes)
        det_bboxes = det_bboxes.reshape(det_bboxes.shape[0], -1, 3)
        
        det_bboxes = det_bboxes[bbox_index]

        cur_bboxes = []
        cur_labels = []
        cur_scores = []
        for cid in range(self.num_classes):
            cid_mask = labels == cid
            score_thrs = scores.new_ones(scores.shape) * score_threshold[cid]
            score_mask = scores > score_thrs
            mask = cid_mask & score_mask
            cur_bboxes.append(det_bboxes[mask])
            cur_labels.append(labels[mask])
            cur_scores.append(scores[mask])
        det_bboxes = torch.cat(cur_bboxes)
        scores = torch.cat(cur_scores)
        labels = torch.cat(cur_labels)

        predictions_dict = {
            'map_scores_3d': scores.cpu(),
            'map_labels_3d': labels.cpu(),
            'map_pts_3d': det_bboxes.cpu(),
        }

        return predictions_dict

    def eval(self):
        if self.torch_ref_model is not None:            
            self.torch_ref_model.eval()
        if len(self.bindings) == 0:
            create_bindings_tensor = True
        else:
            create_bindings_tensor = False
        n_io = self.engine.num_io_tensors
        metas_in = []
        metas_out = []
        for i in range(n_io):
            tname = self.engine.get_tensor_name(i)
            tshape = self.engine.get_tensor_shape(tname)
            tdtype = self.engine.get_tensor_dtype(tname)
            tmode = str(self.engine.get_tensor_mode(tname))
            m = f"{i}\t{tname}\t{str(tshape)}\t{str(tdtype)}"
            if "INPUT" in tmode:
                metas_in.append(m)
            elif "OUTPUT" in tmode:
                metas_out.append(m)
            else:
                assert False, f"Unrecognized tensor mode: {tname}: {tmode}."
            if create_bindings_tensor:
                if str(tdtype) in ['DataType.INT32']:
                    dtype = torch.int32
                else:
                    dtype = torch.float32
                self.bindings[tname] = torch.zeros(list(tshape), dtype=dtype, device=f"cuda:{self.device_id}").contiguous()

        print("##### Input Bindings: ")
        print("\n".join(metas_in))
        print("##### Output Bindings: ")
        print("\n".join(metas_out))
        return

    def __call__(self, img_metas, input_ids, img, lidar2img, cam_intrinsic, timestamp,
                ego_pose, ego_pose_inv, command, can_bus, ego_fut_cmd, 
                gt_attr_labels, gt_bboxes_3d, gt_labels_3d, ego_fut_trajs, fut_valid_flag,
                vlm_labels,
                return_loss=False, rescale=True, **kwargs):
        if self.torch_ref_model is not None:
            data_dict = {
                "img_metas": img_metas, "input_ids": input_ids, "img": img, "lidar2img": lidar2img, 
                "cam_intrinsic": cam_intrinsic,  "timestamp": timestamp, 
                "ego_pose": ego_pose, "ego_pose_inv": ego_pose_inv, "command": command, "can_bus": can_bus, "ego_fut_cmd": ego_fut_cmd,
            }
            ref_result_list = self.torch_ref_model(return_loss=False, rescale=True, **data_dict)
        else:
            ref_result_list = None
            
        return self.forward(img_metas=img_metas, 
                            input_ids=input_ids, 
                            img=img, 
                            lidar2img=lidar2img, 
                            cam_intrinsic=cam_intrinsic, 
                            timestamp=timestamp, 
                            ego_pose=ego_pose, 
                            ego_pose_inv=ego_pose_inv, 
                            command=command, 
                            can_bus=can_bus,
                            ego_fut_cmd=ego_fut_cmd,
                            gt_attr_labels=gt_attr_labels, 
                            gt_bboxes_3d=gt_bboxes_3d,
                            gt_labels_3d=gt_labels_3d,
                            ego_fut_trajs = ego_fut_trajs,
                            fut_valid_flag=fut_valid_flag
                            )

    def forward(self, img_metas, input_ids, img, lidar2img, cam_intrinsic, timestamp, 
                ego_pose, ego_pose_inv, command, can_bus, ego_fut_cmd,
                gt_attr_labels, gt_bboxes_3d, gt_labels_3d, ego_fut_trajs, fut_valid_flag
                ):
        if len(self.bindings) == 0:
            print("Need to call eval() before forward!.")
            exit(-1)
        # 添加输入数据类型控制
        dtype = torch.float32
        # re-format the data
        img_metas = img_metas[0]
        input_ids = input_ids[0][0][0].unsqueeze(0)

        img = img[0].to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        img2lidar = lidar2img[0].to(device=f"cuda:{self.device_id}", dtype=torch.float32).inverse()
        cam_intrinsic = cam_intrinsic[0].to(device=f"cuda:{self.device_id}", dtype=dtype)
        timestamp = timestamp[0]
        ego_pose = ego_pose[0].to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        ego_pose_inv = ego_pose_inv[0].to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        command = command[0].to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        can_bus = can_bus[0].to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        ego_fut_cmd = ego_fut_cmd[0].to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        ego_fut_trajs = ego_fut_trajs[0].to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        fut_valid_flag = bool(fut_valid_flag[0])
        gt_bboxes_3d = gt_bboxes_3d[0][0]
        gt_labels_3d = gt_labels_3d[0][0]
        gt_attr_labels = gt_attr_labels[0][0]

        # convert timestamp from fp64 to fp32
        if self.curr_scene_token is None or img_metas[0]["scene_token"] != self.curr_scene_token:
            self.start_timestamp = timestamp.item()
            self.curr_scene_token = img_metas[0]["scene_token"]
            is_first_frame = torch.ones([1]).to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        else:
            is_first_frame = torch.zeros([1]).to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()
        timestamp -= self.start_timestamp
        timestamp = timestamp.type(torch.float32).to(device=f"cuda:{self.device_id}", dtype=dtype).contiguous()

        # ---- 数据预处理（H2D）----
        for k, v in [("img", img), ("img2lidars", img2lidar), ("command", command),
                     ("can_bus", can_bus), ("is_first_frame", is_first_frame),
                     ("ego_pose", ego_pose), ("ego_pose_inv", ego_pose_inv), ("timestamp", timestamp)]:
            self.bindings[k].copy_(v)

        # [Orion-Latency S5-C] canbus 提前计算：不依赖 Vision 输出，
        # 与 Vision 推理并行执行，掩盖 canbus 延迟（~2ms）。
        _cb_embed_ready = None
        if self.canbus_fixer is not None:
            try:
                scene_same = torch.tensor([1.0]) if is_first_frame.item() == 0 else torch.tensor([0.0])
                if getattr(self, '_canbus_scene', None) != self.curr_scene_token:
                    self.canbus_fixer.reset()
                    self._canbus_scene = self.curr_scene_token
                with torch.no_grad():
                    _fixer_dev = next(self.canbus_fixer.can_bus_embed.parameters()).device
                    _to_dev = lambda t: t.to(_fixer_dev) if _fixer_dev.type == 'cuda' else t.cpu()
                    _cb_embed_ready = self.canbus_fixer.step(
                        _to_dev(is_first_frame.float()), _to_dev(scene_same),
                        _to_dev(ego_pose_inv.float()), _to_dev(ego_pose.float()),
                        _to_dev(command.float()), _to_dev(can_bus.float()))
            except Exception as e:
                print(f"[canbus] pre-fix failed: {e}")
                _cb_embed_ready = None

        self.bindings["memory_embedding_bbox_in"].copy_(self.bindings["memory_embedding_bbox_out"])
        self.bindings["memory_reference_point_bbox_in"].copy_(self.bindings["memory_reference_point_bbox_out"])
        self.bindings["memory_timestamp_bbox_in"].copy_(self.bindings["memory_timestamp_bbox_out"])
        self.bindings["memory_egopose_bbox_in"].copy_(self.bindings["memory_egopose_bbox_out"])
        self.bindings["memory_canbus_bbox_in"].copy_(self.bindings["memory_canbus_bbox_out"])
        self.bindings["sample_time_bbox_in"].copy_(self.bindings["sample_time_bbox_out"])
        self.bindings["memory_scene_query_bbox_in"].copy_(self.bindings["memory_scene_query_bbox_out"])
        self.bindings["scene_memory_timestamp_bbox_in"].copy_(self.bindings["scene_memory_timestamp_bbox_out"])

        self.bindings["memory_timestamp_map_in"].copy_(self.bindings["memory_timestamp_map_out"])
        self.bindings["sample_time_map_in"].copy_(self.bindings["sample_time_map_out"])
        self.bindings["memory_egopose_map_in"].copy_(self.bindings["memory_egopose_map_out"])
        self.bindings["memory_embedding_map_in"].copy_(self.bindings["memory_embedding_map_out"])
        self.bindings["memory_reference_point_map_in"].copy_(self.bindings["memory_reference_point_map_out"])

        # inference
        self.cuda_ctx.push()
        for i in range(len(self.names)):
            self.context.set_tensor_address(self.names[i], self.bindings[str(self.names[i])].data_ptr())
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        self.cuda_ctx.pop()

        # ---- 从 vision engine 输出提取视觉特征 ----
        vision_embeded = self.bindings["vision_embeded"].clone()  # [1, 529, 4096]

        # ---- can_bus_embed 修复：替换 vision_embeded[272]（用提前算好的结果）----
        if _cb_embed_ready is not None and vision_embeded.shape[-2] >= 273:
            vision_embeded[0, 272, :] = _cb_embed_ready[0].to(vision_embeded.device)
        elif self.canbus_fixer is not None and vision_embeded.shape[-2] >= 273:
            # 兜底：提前计算失败时回退到 Vision 后计算
            try:
                scene_same = torch.tensor([1.0]) if is_first_frame.item() == 0 else torch.tensor([0.0])
                if getattr(self, '_canbus_scene', None) != self.curr_scene_token:
                    self.canbus_fixer.reset()
                    self._canbus_scene = self.curr_scene_token
                with torch.no_grad():
                    _fixer_dev = next(self.canbus_fixer.can_bus_embed.parameters()).device
                    _to_dev = lambda t: t.to(_fixer_dev) if _fixer_dev.type == 'cuda' else t.cpu()
                    cb_embed = self.canbus_fixer.step(
                        _to_dev(is_first_frame.float()), _to_dev(scene_same),
                        _to_dev(ego_pose_inv.float()), _to_dev(ego_pose.float()),
                        _to_dev(command.float()), _to_dev(can_bus.float()))
                vision_embeded[0, 272, :] = cb_embed[0].to(vision_embeded.device)
            except Exception as e:
                print(f"[canbus] fix failed: {e}")

        # ---- 使用 InferTrtLLM 合并文本与视觉特征 ----
        # 当前 vision engine 是 _wolm 导出（不含 LLM），不产生 inputs_embeds/new_input_ids
        # 需要通过 InferTrtLLM.prepare_inputs_labels_for_multimodal_() 生成
        inputs_embeds, new_input_ids = self.LLM_engine.prepare_inputs_labels_for_multimodal_(
            input_ids, vision_embeded
        )

        # ---- 调用 LLM engine 推理 ----
        output_ids, ego_feature, _per_layer = self.LLM_engine.generate_test(inputs_embeds, new_input_ids)

        # 新增：使用ego_fut_pred_sim.engine进行推理（可选）
        if self.ego_loaded:
            self.cuda_ctx.push()
            # 设置输入特征 (假设输入名称为"ego_feature")
            self.ego_bindings["ego_feature"].copy_(ego_feature.contiguous())
            # 设置所有绑定地址
            for i in range(len(self.ego_names)):
                self.ego_context.set_tensor_address(self.ego_names[i], self.ego_bindings[self.ego_names[i]].data_ptr())
            # 执行推理
            self.ego_context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
            self.cuda_ctx.pop()
            # 获取输出结果 (假设输出名称为"ego_fut_pred")
            ego_fut_preds = self.ego_bindings["ego_fut_pred"].clone()
        else:
            # PyTorch fallback: 使用原始模型权重计算 ego_fut_pred
            if self.torch_ego_model is not None:
                with torch.no_grad():
                    m = self.torch_ego_model
                    _ego_dev = next(m.parameters()).device
                    if _ego_dev.type == 'cuda':
                        # [Orion-Latency] GPU 模式 + 批处理 decoder（延迟 ~49ms -> ~2ms）
                        ego_feature_g = ego_feature.to(torch.float32).to(_ego_dev)
                        current_states = ego_feature_g.unsqueeze(1)
                        sample, _ = m.distribution_forward(current_states)
                        hidden_states = ego_feature_g.unsqueeze(1)
                        states_hs, _ = m.future_states_predict(1, sample, hidden_states, current_states)
                        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                        # 批处理: [fut_ts, 1, 8192] 一次过 decoder
                        stacked = torch.cat([ego_query_hs[ti] for ti in range(self.fut_ts)], dim=0)
                        out = m.ego_fut_decoder(stacked)  # [fut_ts, 1, 12]
                        ego_fut_preds = out.reshape(self.fut_ts, 1, m.ego_fut_mode, 2).permute(1, 2, 0, 3)
                    else:
                        # CPU 模式（原逻辑）
                        ego_feature_cpu = ego_feature.to(torch.float32).cpu()
                        current_states = ego_feature_cpu.unsqueeze(1)
                        sample, _ = m.distribution_forward(current_states)
                        hidden_states = ego_feature_cpu.unsqueeze(1)
                        states_hs, _ = m.future_states_predict(1, sample, hidden_states, current_states)
                        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                        ego_fut_trajs_list = []
                        for ti in range(self.fut_ts):
                            out = m.ego_fut_decoder(ego_query_hs[ti]).reshape(1, m.ego_fut_mode, 2)
                            ego_fut_trajs_list.append(out)
                        ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2).cuda()
            else:
                print("[WARNING] ego engine not loaded and no torch model, using zeros.")
                ego_fut_mode = 6
                ego_fut_preds = torch.zeros(1, self.fut_ts, ego_fut_mode, 2, device=f"cuda:{self.device_id}")

        bbox_results = self.get_bbox(self.bindings["all_cls_scores"].clone(), 
                                     self.bindings["all_bbox_preds"].clone(),
                                     self.bindings["all_traj_preds"].clone(),
                                     self.bindings["all_traj_cls_scores"].clone(),
                                     img_metas)
        
        lane_results = self.get_lane(self.bindings["all_lane_cls_one2one"].clone(), 
                                     self.bindings["all_lane_preds_one2one"].clone(), 
                                     img_metas)

        # metric_dict
        metric_dict = {}
        # lane_results
        ego_fut_trajs = ego_fut_trajs[0, 0]
        ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)
        # ego_fut_preds = self.bindings["ego_fut_pred"].clone()
        mask_active_cmd = ego_fut_cmd[:,0,0] == 1 # [1, 6]
        ego_fut_preds = ego_fut_preds[mask_active_cmd].flatten(0,1).to('cpu') # torch.Size([6, 2])
        ego_fut_preds = ego_fut_preds.to(torch.float32)
        ego_fut_pred = ego_fut_preds.cumsum(dim=-2)

        metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None].to('cpu'),
                gt_ego_fut_trajs = ego_fut_trajs[None].to('cpu'),
                gt_agent_boxes = gt_bboxes_3d,
                gt_agent_feats = gt_attr_labels.unsqueeze(0),
                fut_valid_flag = fut_valid_flag # 当前帧是否涵盖6个轨迹
            )
        metric_dict.update(metric_dict_planner_stp3)
        lane_results[0]['fut_valid_flag'] = fut_valid_flag
        lane_results[0]['ego_fut_preds'] = torch.nan_to_num(ego_fut_pred)
        lane_results[0]['ego_fut_cmd'] = ego_fut_cmd

        # generate text
        generated_text = []

        result_list = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(result_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        result_list[0]['text_out'] = generated_text
        result_list[0]['pts_bbox'].update(lane_results[0])

        return result_list

    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag or pred_ego_fut_trajs.size(1)==6 :
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i+1)] = np.nan_to_num(traj_L2)
                metric_dict['plan_obj_col_{}s'.format(i+1)] = np.nan_to_num(obj_coll.mean().item())
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = np.nan_to_num(obj_box_coll.mean().item())
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict
    
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) an engine')
    parser.add_argument('--config',help='test config file path')
    parser.add_argument('--engine_pth', help='engine file path')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
    parser.add_argument('--precision', choices=['fp16', 'mixed'], default='mixed', help='Precision mode for inference')
    parser.add_argument('--ego_engine_pth', type=str, default=None)
    parser.add_argument('--llm_engine_pth', type=str, default=None)
    parser.add_argument('--tokenizer_pth', type=str, default=None)
    parser.add_argument('--qa_save_path', type=str, default=None)
    parser.add_argument('--embedding_weights_pth', type=str, default=None)
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    # if hasattr(cfg, 'plugin'):
    #     if cfg.plugin:
    #         import importlib
    #         if hasattr(cfg, 'plugin_dir'):
    #             plugin_dir = cfg.plugin_dir
    #             _module_dir = os.path.dirname(plugin_dir)
    #             _module_dir = _module_dir.split('/')
    #             _module_path = _module_dir[0]

    #             for m in _module_dir[1:]:
    #                 _module_path = _module_path + '.' + m
    #             print(_module_path)
    #             plg_lib = importlib.import_module(_module_path)
    #         else:
    #             # import dir is the dirpath for the config file
    #             _module_dir = os.path.dirname(args.config)
    #             _module_dir = _module_dir.split('/')
    #             _module_path = _module_dir[0]
    #             for m in _module_dir[1:]:
    #                 _module_path = _module_path + '.' + m
    #             print(_module_path)
    #             plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,  # 避免 DataLoader fork 导致内存翻倍
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    # cfg.model.train_cfg = None
    # model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)
    # # old versions did not save class info in checkpoints, this walkaround is
    # # for backward compatibility
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES
    # # palette for visualization in segmentation tasks
    # if 'PALETTE' in checkpoint.get('meta', {}):
    #     model.PALETTE = checkpoint['meta']['PALETTE']
    # elif hasattr(dataset, 'PALETTE'):
    #     # segmentation dataset has `PALETTE` attribute
    #     model.PALETTE = dataset.PALETTE
    
    # if not os.path.exists(args.qa_save_path):
    #     os.makedirs(args.qa_save_path)
    # build the engine
    logger = trt.Logger(trt.Logger.VERBOSE)
    engine = InferTrt(logger, ego_engine_pth=args.ego_engine_pth)
    engine.read(args.engine_pth)
    engine.LLM_engine = InferTrtLLM(llm_engine_pth=args.llm_engine_pth, tokenizer_pth=args.tokenizer_pth)

    # 如果 ego engine 不存在，加载 PyTorch 模型作为 fallback（用于 planning 轨迹预测）
    # 用轻量 ego 模块（仅提取 Orion.pth 中的 ego 预测头 ~1.6GB），
    # 避免 build_model 加载完整模型（含 7B vision tower / LLM，38GB checkpoint 导致 OOM）。
    if not engine.ego_loaded and args.checkpoint:
        print("Loading lightweight ego module for ego prediction fallback...")
        import importlib.util
        _ego_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lightweight_ego.py')
        _spec = importlib.util.spec_from_file_location('lightweight_ego', _ego_path)
        _ego_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ego_mod)
        # [Orion-Latency] 支持 ego 头跑 GPU（EGO_ON_GPU=1），延迟 ~49ms -> ~2ms
        _ego_dev = 'cuda:0' if os.environ.get('EGO_ON_GPU', '0') == '1' else 'cpu'
        torch_ego_model = _ego_mod.load_lightweight_ego(args.checkpoint, device=_ego_dev)
        torch_ego_model.eval()
        engine.torch_ego_model = torch_ego_model
        print("Lightweight ego module loaded.")

    if not distributed:
        # model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test_trt(engine, data_loader)
    else:
        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False)
        # engine.torch_ref_model = model
        outputs = custom_multi_gpu_test(engine, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs['bbox_results'], **eval_kwargs))


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('fork')
    cuda.init()
    # torch.cuda.init()
    main()
