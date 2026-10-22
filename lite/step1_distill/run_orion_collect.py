#!/usr/bin/env python
"""Standalone runner for Orion teacher distillation collection."""

import argparse
import glob
import os
import shutil
import sys

import torch
from torch.nn import DataParallel


REPO_ROOT = os.environ.get("ORION_LITE_PROJECT_ROOT")
if not REPO_ROOT:
    raise SystemExit("error: ORION_LITE_PROJECT_ROOT is not set - point it at a clone of the "
                     "upstream Orion training repo (Lite variant, see ckpts/README.md)")


def ensure_repo_on_path():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def _find_sibling_orion_mmcv():
    # the compiled mmcv extensions can be shared from the 7B training repo
    sibling = os.environ.get("ORION_PROJECT_ROOT")
    if sibling and os.path.isdir(os.path.join(sibling, 'mmcv')):
        return os.path.join(sibling, 'mmcv')
    return os.path.join(REPO_ROOT, 'mmcv')


def ensure_local_mmcv_extensions():
    """Reuse compiled mmcv extensions from sibling Orion when local ones are absent."""
    local_mmcv_dir = os.path.join(REPO_ROOT, 'mmcv')
    sibling_orion_mmcv_dir = _find_sibling_orion_mmcv()
    extension_patterns = (
        ('_ext*.so', ''),
        ('iou3d_cuda*.so', os.path.join('ops', 'iou3d_det')),
        ('roiaware_pool3d_ext*.so', os.path.join('ops', 'roiaware_pool3d')),
    )

    for pattern, subdir in extension_patterns:
        local_dir = os.path.join(local_mmcv_dir, subdir)
        if glob.glob(os.path.join(local_dir, pattern)):
            continue

        source_dir = os.path.join(sibling_orion_mmcv_dir, subdir)
        source_matches = glob.glob(os.path.join(source_dir, pattern))
        if not source_matches:
            continue

        os.makedirs(local_dir, exist_ok=True)
        for source_path in source_matches:
            dest_path = os.path.join(local_dir, os.path.basename(source_path))
            if os.path.exists(dest_path):
                continue
            try:
                os.symlink(source_path, dest_path)
                print(f"[collect] Linked extension: {dest_path} -> {source_path}")
            except OSError:
                shutil.copy2(source_path, dest_path)
                print(f"[collect] Copied extension: {dest_path} <- {source_path}")


def resolve_repo_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Orion distillation data (saves .npz per frame)"
    )
    parser.add_argument('--config', required=True,
                        help='Teacher Orion config (for example adzoo/orion/configs/orion_stage3_infer.py)')
    parser.add_argument('--checkpoint', required=True,
                        help='Orion checkpoint path')
    parser.add_argument('--save-dir', required=True,
                        help='Output directory for .npz files')
    parser.add_argument('--split', default='val', choices=['train', 'val'],
                        help='Dataset split to collect')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--data-root', default=None,
                        help='Override data root (bench2drive directory)')
    parser.add_argument('--info-root', default=None,
                        help='Override info root (directory with b2d_infos_*.pkl)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of new .npz files to save for this run')
    parser.add_argument('--fp16', action='store_true',
                        help='Load model in FP16 to reduce GPU memory (recommended for 24GB GPUs)')
    parser.add_argument('--multi-gpu', type=int, default=None,
                        help='Use multi-GPU model parallelism. Specify the second GPU id '
                             '(e.g. --gpu 2 --multi-gpu 3 uses GPU 2+3). '
                             'Requires accelerate library.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Samples per GPU batch (default: 1). Increase for faster collection '
                             'when GPU memory allows.')
    parser.add_argument('--ann-file', default=None,
                        help='Override ann_file path (e.g. for parallel collection with split infos)')
    return parser.parse_args()


def main():
    ensure_repo_on_path()
    ensure_local_mmcv_extensions()

    from mmcv.utils import Config
    from mmcv.models import build_detector
    from mmcv.utils.checkpoint import load_checkpoint
    from mmcv.datasets import build_dataset, build_dataloader

    import adzoo.orion  # noqa: F401

    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.model.type = 'OrionCollect'
    cfg.model.save_dir = args.save_dir
    for key in (
        'student_model_path',
        'student_model_conf',
        'student_load_strict',
        'load_checkpoint_verbose',
    ):
        cfg.model.pop(key, None)

    if cfg.model.get('lm_head', None) is None:
        raise ValueError(
            'Collection requires a teacher Orion config with `lm_head` enabled. '
            f'Got config without lm_head: {args.config}'
        )

    data_root = resolve_repo_path(args.data_root or cfg.data.test.data_root)
    info_root = resolve_repo_path(args.info_root or cfg.get('info_root', 'data/infos'))

    ann_file = os.path.join(info_root, f'b2d_infos_{args.split}.pkl')
    if args.ann_file is not None:
        ann_file = resolve_repo_path(args.ann_file)
    cfg.data.test.ann_file = ann_file
    cfg.data.test.data_root = data_root
    if 'map_root' in cfg.data.test:
        cfg.data.test.map_root = os.path.join(
            data_root, os.path.basename(os.path.normpath(cfg.data.test.map_root))
        )
    if 'map_file' in cfg.data.test:
        cfg.data.test.map_file = os.path.join(
            info_root, os.path.basename(cfg.data.test.map_file)
        )

    cfg.data.workers_per_gpu = args.num_workers

    print(f"[collect] config     : {args.config}")
    print(f"[collect] checkpoint : {args.checkpoint}")
    print(f"[collect] save_dir   : {args.save_dir}")
    print(f"[collect] split      : {args.split}  ({ann_file})")
    print(f"[collect] data_root  : {cfg.data.test.data_root}")
    if 'map_root' in cfg.data.test:
        print(f"[collect] map_root   : {cfg.data.test.map_root}")
    if 'map_file' in cfg.data.test:
        print(f"[collect] map_file   : {cfg.data.test.map_file}")
    if args.max_samples is not None:
        print(f"[collect] max_samples: {args.max_samples}")

    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fp16:
        print("[collect] FP16 mode: using autocast(bfloat16) for stable inference")

    if args.multi_gpu is not None:
        n_gpus = torch.cuda.device_count()
        vision_device = torch.device(f'cuda:{args.gpu}')
        llm_device = torch.device(f'cuda:{args.multi_gpu}')
        print(f"[collect] Multi-GPU mode: "
              f"vision on cuda:{args.gpu}, LLM on cuda:{args.multi_gpu}")

        for name, module in model.named_children():
            if name == 'lm_head':
                module.to(torch.bfloat16).to(llm_device)
                print(f"  {name} -> {llm_device} (BF16)")
            else:
                module.to(vision_device)
                print(f"  {name} -> {vision_device}")

        for name, param in model.named_parameters():
            prefix = name.split('.')[0]
            if prefix not in dict(model.named_children()):
                param.data = param.data.to(vision_device)

        for name, buf in model.named_buffers():
            prefix = name.split('.')[0]
            if prefix not in dict(model.named_children()):
                buf.data = buf.data.to(vision_device)

        original_inference_ego = model.lm_head.inference_ego
        def _cross_gpu_inference_ego(inputs, images=None, **kw):
            if images is not None:
                images = images.to(llm_device)
            inputs_on_llm = {k: v.to(llm_device) if isinstance(v, torch.Tensor) else v
                             for k, v in inputs.items()} if isinstance(inputs, dict) else inputs
            result = original_inference_ego(inputs=inputs_on_llm, images=images, **kw)
            if isinstance(result, torch.Tensor):
                result = result.to(vision_device)
            return result
        model.lm_head.inference_ego = _cross_gpu_inference_ego
        print("[collect] Cross-GPU bridge installed for lm_head.inference_ego")
    else:
        model = DataParallel(model, device_ids=[args.gpu])
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)
    initial_count = len([f for f in os.listdir(args.save_dir) if f.endswith('.npz')])
    total = len(dataloader)

    target_device = vision_device if args.multi_gpu is not None else torch.device(f'cuda:{args.gpu}')

    def _move_data(data, device):
        from mmcv.parallel import DataContainer
        if isinstance(data, DataContainer):
            return DataContainer(
                _move_data(data.data, device),
                stack=data.stack,
                padding_value=data.padding_value,
                cpu_only=data.cpu_only,
            )
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: _move_data(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            moved = [_move_data(v, device) for v in data]
            return type(data)(moved)
        return data

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if args.max_samples is not None:
                current_count = len([f for f in os.listdir(args.save_dir) if f.endswith('.npz')])
                if (current_count - initial_count) >= args.max_samples:
                    print(f"[collect] Reached max_samples={args.max_samples}. Stopping early.")
                    break

            data = _move_data(data, target_device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(data, return_loss=False)

            if (i + 1) % 100 == 0:
                count = len([f for f in os.listdir(args.save_dir) if f.endswith('.npz')])
                print(f"[collect] {i + 1}/{total}  saved so far: {count}")

    count = len([f for f in os.listdir(args.save_dir) if f.endswith('.npz')])
    print(f"[collect] Done. {count} .npz files written to {args.save_dir}")


if __name__ == '__main__':
    main()
