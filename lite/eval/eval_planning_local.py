#!/usr/bin/env python
"""Local real-pipeline planning eval: per-frame forward passes for plan_L2
(real images + real cross-frame memory evolution), matching the upstream
test.py convention (model.simple_test -> metric_results.plan_L2_*), but:
- per-frame plan_L2 extraction (avoids the 24L@640 bbox_results memory peak)
- configurable resolution / backbone depth
- reports all (173) / valid_no_pollute (163) / fut_valid_only (133) subsets of
  the 173-frame eval route - see data/README.md for how that set is defined

Usage:
  ORION_LITE_PROJECT_ROOT=<clone of the Orion-Lite training repo> \
  python <this-repo>/lite/eval/eval_planning_local.py --ckpt <ckpt> [--depth 24|12] [--img-size 512] [--out xxx.json]

The model config is vendored at configs/orion_eval_tiny173.py (eval-frozen copy of the
training repo's orion_distill_24l640_tiny.py). Eval data location comes from
ORION_B2D_DATA_ROOT (default <ORION_LITE_PROJECT_ROOT>/data); see data/README.md.
"""
import argparse, os, sys, json, copy
import numpy as np
import torch

_release_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('ORION_QFORMER_PATH', os.path.join(_release_root, 'ckpts', 'hf_models', 'pretrain_qformer'))
# Default eval-data root: this repo's data/ layout (see data/README.md). The
# vendored config reads it; an explicit ORION_B2D_DATA_ROOT wins.
os.environ.setdefault('ORION_B2D_DATA_ROOT', os.path.join(_release_root, 'data'))
_training_repo = os.environ.get('ORION_LITE_PROJECT_ROOT')
if not _training_repo:
    raise SystemExit('error: ORION_LITE_PROJECT_ROOT is not set - point it at a clone of the '
                     'Orion-Lite training repo (the model code imports from there)')
sys.path.insert(0, _training_repo)
sys.path.insert(0, os.path.join(_training_repo, 'adzoo'))

# flash_attn is hard-imported at module level by the training repo's vendored mmcv
# fork, but it is a compiled wheel pinned to a torch ABI (a torch upgrade routinely
# breaks it). The config overrides below force flash_attn=False on every attention
# consumer, so the CUDA functions are never called during eval. If flash_attn is
# not importable, install a raise-if-called stub instead of failing; same pattern as
# lite/step2_onnx_export/export_vision_onnx.py.
try:
    import flash_attn  # noqa: F401
    from flash_attn.bert_padding import index_first_axis  # noqa: F401
except Exception:
    def _flash_stub(*_a, **_kw):
        raise RuntimeError(
            'flash_attn was not importable, so this script stubbed it; the config '
            'overrides below force flash_attn=False and should never call it. This '
            'call means the flash_attn=False overrides do not cover your model config.')
    import importlib.util

    def _mk(name):
        spec = importlib.util.spec_from_loader(name, None)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        return mod

    _fa = _mk('flash_attn')
    _fi = _mk('flash_attn.flash_attn_interface')
    _fi.flash_attn_varlen_kvpacked_func = _flash_stub
    _fi.flash_attn_unpadded_kvpacked_func = _flash_stub
    _bp = _mk('flash_attn.bert_padding')
    _bp.unpad_input = _flash_stub
    _bp.pad_input = _flash_stub
    _bp.index_first_axis = _flash_stub
    print('note: flash_attn not importable - stubbed (eval runs flash_attn=False)')

# nuscenes-devkit's map_api imports mark_inset from mpl_toolkits.axes_grid1.inset_locator.
# On hosts mixing an older system matplotlib with a newer user-site one, that import
# breaks (matplotlib dropped its `docstring` module, and the system mpl_toolkits
# package shadows the user-site one regardless of sys.path order). map_api uses
# mark_inset only in plotting helpers that eval never calls, so pre-insert a working
# module if the real one fails to import.
try:
    import mpl_toolkits.axes_grid1.inset_locator  # noqa: F401
except Exception:
    _spec = importlib.util.spec_from_loader('mpl_toolkits.axes_grid1.inset_locator', None)
    _inset = importlib.util.module_from_spec(_spec)
    sys.modules['mpl_toolkits.axes_grid1.inset_locator'] = _inset

    def _inset_stub(*_a, **_kw):
        raise RuntimeError('inset_locator was stubbed (plotting helper, never used in eval)')

    _inset.mark_inset = _inset_stub
    print('note: mpl_toolkits.axes_grid1.inset_locator not importable - stubbed')

# matplotlib >= 3.8 removed the 'seaborn-whitegrid' style that nuscenes-devkit's map_api
# calls at import time. It is cosmetic plotting state; eval never plots. Swallow any
# removed-style error instead of failing the import.
try:
    import matplotlib.style as _mstyle
    _orig_style_use = _mstyle.use

    def _style_use(name):
        try:
            _orig_style_use(name)
        except OSError:
            print(f"note: matplotlib style '{name}' not available - skipped (cosmetic)")

    _mstyle.use = _style_use
except Exception:
    pass

from mmcv.utils import Config, load_checkpoint
from mmcv.models import build_model
from mmcv.datasets import build_dataset, build_dataloader

# Some torch builds reject tensor-vs-numpy-scalar rich comparisons
# (`tensor > np.float32` TypeError; torch 2.4.1 raises it, torch 2.11 does not).
# The training repo's LiDAR box filter compares against config-supplied numpy
# scalars. Probe once; only if the running torch rejects the comparison, patch
# the one affected method to coerce the range to a tensor first (identical
# semantics), so the script runs on both kinds of torch.
try:
    torch.zeros(2) > np.float32(1.0)
except TypeError:
    from mmcv.core.bbox.structures import lidar_box3d as _l3d_mod

    def _in_range_bev(self, box_range):
        br = torch.as_tensor(np.asarray(box_range), device=self.tensor.device)
        t = self.tensor
        return ((t[:, 0] > br[0]) & (t[:, 1] > br[1])
                & (t[:, 0] < br[2]) & (t[:, 1] < br[3]))

    _l3d_mod.LiDARInstance3DBoxes.in_range_bev = _in_range_bev
    print('note: torch rejects tensor-vs-numpy comparisons - patched '
          'LiDARInstance3DBoxes.in_range_bev')
from torch.nn import DataParallel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--config', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'configs', 'orion_eval_tiny173.py'))
    p.add_argument('--depth', type=int, default=24, help='backbone 层数（24L 用 24；12L ckpt 用 12）')
    p.add_argument('--img-size', type=int, default=512)
    p.add_argument('--out', default='work_dirs/eval_planning_local.json')
    p.add_argument('--max-frames', type=int, default=0, help='0=全部 173')
    p.add_argument('--smoke', action='store_true', help='仅 3 帧冒烟')
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    def _flash_off(node):
        for v in node.values():
            if isinstance(v, dict):
                if 'flash_attn' in v:
                    v['flash_attn'] = False
                if 'with_cp' in v:
                    v['with_cp'] = False
                _flash_off(v)

    _flash_off(cfg.model)
    os.environ.pop('ORION_STUDENT_MODEL_PATH', None)
    cfg.model['student_model_path'] = None
    cfg.model['student_load_strict'] = False
    cfg.model['img_backbone']['img_size'] = args.img_size
    if args.depth != 24:
        cfg.model['img_backbone']['depth'] = args.depth
        cfg.model['img_backbone']['window_block_indexes'] = tuple(range(0, args.depth))
        cfg.model['img_backbone']['residual_block_indexes'] = ()
    # 同步 test pipeline 的 resize 分辨率（否则输入尺寸与 backbone rope 不匹配）
    for pipe_cfg in [cfg.data.test.pipeline, cfg.data.test.get('inference_only_pipeline', [])]:
        if not isinstance(pipe_cfg, list):
            continue
        for step in pipe_cfg:
            if step.get('type') == 'ResizeMultiview3D':
                step['img_scale'] = (args.img_size, args.img_size)

    # dataset: real images + real annotations (the config's data_root points at the local Bench2Drive copy)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                                   dist=False, shuffle=False)
    print(f'dataset: {len(dataset)} frames, img_size={args.img_size}, depth={args.depth}')

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 加载 ckpt；若 ckpt 分辨率 rope 与 config 不匹配，跳过 rope（rope 按 img_size 生成）
    ckpt_sd = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt_sd.get('state_dict', ckpt_sd)
    rope_keys = [k for k in state_dict if 'rope' in k or 'freqs' in k]
    for k in rope_keys:
        state_dict.pop(k, None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'load {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)} skipped_rope={len(rope_keys)}')
    model.cuda().eval()
    model = DataParallel(model, device_ids=[0])
    print('model loaded on cuda')

    results = {k: [] for k in ['plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s']}
    fut_valid = []
    frame_ids = []
    n = len(dataset) if args.max_frames == 0 else min(args.max_frames, len(dataset))
    if args.smoke:
        n = min(3, n)
    # 逐帧落盘（崩溃可恢复）
    out_jsonl = args.out.replace('.json', '.jsonl')
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)
    print(f'evaluating {n} frames...', flush=True)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= n:
                break
            try:
                result = model(data, return_loss=False)
                mr = result[0].get('metric_results', {})
                row = {'frame': i}
                for k in results:
                    val = float(mr.get(k, 0.0))
                    results[k].append(val)
                    row[k] = val
                row['fut_valid'] = bool(mr.get('fut_valid_flag', False))
                fut_valid.append(row['fut_valid'])
                frame_ids.append(i)
                with open(out_jsonl, 'a') as f:
                    f.write(json.dumps(row) + '\n')
            except Exception as e:
                print(f'  frame {i} ERROR: {e}', flush=True)
                continue
            torch.cuda.empty_cache()
            if (i + 1) % 20 == 0:
                mem = torch.cuda.memory_allocated() / 1e9
                print(f'  frame {i+1}/{n} (gpu {mem:.1f}G)', flush=True)

    # 汇总：全量 + 排除污染帧 10-19 + 仅 fut_valid
    valid = np.ones(len(frame_ids), bool)
    for idx, fid in enumerate(frame_ids):
        if 10 <= fid <= 19:
            valid[idx] = False

    def summ(vals, mask=None):
        v = np.array(vals)
        if mask is not None:
            v = v[mask]
        if v.size == 0:
            return {'mean': float('nan'), 'min': float('nan'), 'max': float('nan'), 'n': 0}
        return {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()), 'n': int(len(v))}

    out = {
        'config': {'ckpt': args.ckpt, 'img_size': args.img_size, 'depth': args.depth, 'n': n},
        'all': {k: summ(v) for k, v in results.items()},
        'valid_no_pollute': {k: summ(v, valid) for k, v in results.items()},
        'fut_valid_only': {k: summ([v for v, f in zip(results[k], fut_valid) if f]) for k in results},
        'frame_ids': frame_ids,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n=== plan_L2（全部帧）===')
    for k, v in out['all'].items():
        print(f'  {k}: mean={v["mean"]:.4f}')
    print('=== plan_L2（排除污染 10-19）===')
    for k, v in out['valid_no_pollute'].items():
        print(f'  {k}: mean={v["mean"]:.4f} (n={v["n"]})')
    print('=== plan_L2（仅 fut_valid 帧）===')
    for k, v in out['fut_valid_only'].items():
        print(f'  {k}: mean={v["mean"]:.4f} (n={v["n"]})')
    print(f'\nsaved to {args.out}')


if __name__ == '__main__':
    main()
