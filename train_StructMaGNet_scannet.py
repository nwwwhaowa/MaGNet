#!/usr/bin/env python3
"""ScanNet depth-supervised retraining: fixed D/F, train G/upsampling and one gate.

No oracle targets, noise labels, curriculum, or additional loss. Use gate_mode=off
for the same-budget MaGNet ablation. See docs/SECOND_TRAINING.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.retraining import (
    configure_training, depth_loss, load_backbone, perturb_rotations,
    restore_checkpoint, save_checkpoint, set_train_mode,
)


def build_model_args(cli):
    return SimpleNamespace(
        output_dim=2,
        output_type="G",
        downsample_ratio=4,
        DNET_architecture="DenseDepth_BN",
        DNET_fix_encoder_weights="None",
        DNET_ckpt=cli.dnet_ckpt,
        FNET_architecture="PSM-Net",
        FNET_feature_dim=64,
        FNET_ckpt=cli.fnet_ckpt,
        MAGNET_sampling_range=3,
        MAGNET_num_samples=5,
        MAGNET_mvs_weighting="CW5",
        MAGNET_num_train_iter=cli.num_train_iter,
        MAGNET_num_test_iter=cli.num_test_iter,
        MAGNET_window_radius=cli.window_radius,
        MAGNET_num_source_views=cli.num_source_views,
        MAGNET_ckpt=cli.magnet_ckpt,
        loss_fn="gaussian",
        loss_gamma=cli.loss_gamma,
        dataset_name="scannet",
        dataset_path=os.path.expanduser(cli.dataset_root),
        scannet_train_split=os.path.expanduser(cli.train_split),
        scannet_val_split=os.path.expanduser(cli.val_split),
        scannet_scans_dir=cli.scans_dir,
        scannet_raw_wh=os.path.expanduser(cli.raw_wh_json),
        input_height=cli.input_height,
        input_width=cli.input_width,
        dpv_height=cli.dpv_height,
        dpv_width=cli.dpv_width,
        min_depth=cli.min_depth,
        max_depth=cli.max_depth,
        gate_mode=cli.gate_mode,
        persistent_workers=False,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        workers=cli.num_workers,
        num_threads=cli.num_workers,
        distributed=False,
        pin_memory=cli.pin_memory,
        data_augmentation_color=cli.color_aug,
        mode="train",
        do_kb_crop=False,
        eigen_crop=False,
        garg_crop=False,
    )


def _check_inputs(cli):
    required_files = {
        "D-Net checkpoint": cli.dnet_ckpt,
        "F-Net checkpoint": cli.fnet_ckpt,
        "MaGNet checkpoint": cli.magnet_ckpt,
        "ScanNet train split": cli.train_split,
        "ScanNet val split": cli.val_split,
        "ScanNet raw W/H metadata": cli.raw_wh_json,
    }
    for label, value in required_files.items():
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    scans_root = Path(cli.dataset_root).expanduser() / cli.scans_dir
    if not scans_root.is_dir():
        raise FileNotFoundError(f"ScanNet scans directory not found: {scans_root}")


def prepare_batch(batch, device):
    from utils.utils import data_preprocess
    data_array, intrinsics = batch
    ref, sources, poses, valid = data_preprocess(data_array, data_array[0]['img'].shape[0])
    return (ref['img'].to(device), torch.cat([d['img'] for d in sources]).to(device),
            poses.to(device), valid, intrinsics, ref['gt_dmap'].to(device))


def append_csv(path, row):
    new = not path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validate(model, loader, device, cli):
    """Same views/axes at every noise level and across all A/B/C comparisons."""
    model.eval()
    rows = []
    for degrees in cli.val_degrees:
        noise_rng = torch.Generator().manual_seed(cli.seed + 10000)
        totals = dict(squared=0., relative=0., correct=0., nll=0., pixels=0,
                      gate=0., update=0., gate_pixels=0, samples=0)
        for index, batch in enumerate(loader):
            if cli.val_max_samples and index >= cli.val_max_samples:
                break
            ref, sources, poses, valid_views, intrinsics, gt = prepare_batch(batch, device)
            poses, _ = perturb_rotations(poses, generator=noise_rng, fixed_degrees=degrees)
            with torch.cuda.amp.autocast(enabled=cli.amp):
                predictions, aux = model(ref, sources, poses, valid_views, intrinsics,
                                         mode='test', rot_unc=None, return_aux=True)
            valid = torch.isfinite(gt) & (gt > cli.min_depth) & (gt < cli.max_depth)
            if not valid.any():
                continue
            # Check the raw prediction; the metric depth clamp must not hide NaN.
            nll = depth_loss([predictions[-1]], gt, cli.min_depth, cli.max_depth)
            mu = predictions[-1][:, :1].float()[valid].clamp(cli.min_depth, cli.max_depth)
            target = gt.float()[valid]
            count = target.numel()
            totals['squared'] += (mu - target).square().sum().item()
            totals['relative'] += ((mu - target).abs() / target).sum().item()
            totals['correct'] += (torch.maximum(mu / target, target / mu) < 1.25).sum().item()
            totals['nll'] += nll.item() * count
            totals['pixels'] += count
            totals['samples'] += gt.shape[0]
            for gate, update in zip(aux['geometry_gate'], aux['depth_update']):
                totals['gate'] += gate.sum().item()
                totals['update'] += update.abs().sum().item()
                totals['gate_pixels'] += gate.numel()
        if not totals['pixels']:
            raise RuntimeError('Validation contains no valid depth pixels')
        n, ng = totals['pixels'], totals['gate_pixels']
        row = dict(noise_deg=degrees, samples=totals['samples'], valid_pixels=n,
                   rmse=(totals['squared'] / n) ** .5, abs_rel=totals['relative'] / n,
                   a1=totals['correct'] / n, nll=totals['nll'] / n,
                   gate_mean=totals['gate'] / ng, update_abs_mean=totals['update'] / ng)
        rows.append(row)
        print(json.dumps(row))
    return rows


def train(cli):
    import random
    import numpy as np
    from tqdm import tqdm
    from data.dataloader_scannet_train import ScannetTrainLoader, _read_split
    from models.STRUCTMAGNET import STRUCTMAGNET

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full ScanNet training')
    device = torch.device('cuda:0')
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    # Dedicated RNGs keep noise and sample order identical across model ablations.
    noise_rng = torch.Generator().manual_seed(cli.seed + 1)
    loader_rng = torch.Generator().manual_seed(cli.seed + 2)
    args = build_model_args(cli)
    args.loader_generator = loader_rng
    train_spaces = {s.split('_')[0] for s, _ in _read_split(args.scannet_train_split)}
    val_spaces = {s.split('_')[0] for s, _ in _read_split(args.scannet_val_split)}
    if train_spaces & val_spaces:
        raise ValueError('Train/val contain the same physical ScanNet spaces')
    train_loader = ScannetTrainLoader(args, 'train').data
    val_loader = ScannetTrainLoader(args, 'val').data
    if len(train_loader) == 0:
        raise ValueError('Training split is smaller than batch_size with drop_last=True')

    model = STRUCTMAGNET(args).to(device)
    load_backbone(cli.magnet_ckpt, model)
    parameters = configure_training(model)
    optimizer = torch.optim.AdamW(parameters, lr=cli.lr, weight_decay=cli.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cli.amp)
    epoch_start, step, best_score = 0, 0, float('inf')
    if cli.resume:
        state = restore_checkpoint(cli.resume, model,
                                   None if cli.eval_only else optimizer,
                                   None if cli.eval_only else scaler,
                                   noise_rng, loader_rng)
        epoch_start, step = state['epoch'] + 1, state['step']
        best_score = state['best_score']
        if not cli.eval_only:
            for key in ('clean_probability', 'rotation_max_deg', 'num_train_iter',
                        'num_test_iter', 'loss_gamma', 'val_degrees', 'val_max_samples',
                        'train_split', 'val_split', 'dataset_root', 'batch_size', 'seed',
                        'lr', 'weight_decay', 'color_aug', 'min_depth', 'max_depth',
                        'window_radius', 'num_source_views', 'input_height', 'input_width',
                        'dpv_height', 'dpv_width', 'dnet_ckpt', 'fnet_ckpt', 'magnet_ckpt'):
                if vars(cli)[key] != state['config'][key]:
                    raise ValueError(f'Resume configuration differs: {key}')

    output = Path(cli.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / ('eval_config.json' if cli.eval_only else 'config.json')
    if config_path.exists() and not cli.resume:
        raise FileExistsError('Use a new output_dir or --resume to preserve existing results')
    config_path.write_text(json.dumps(vars(cli), indent=2))
    print('Trainable parameters:', sum(p.numel() for p in parameters))
    print('Loss: depth Gaussian NLL; gate:', cli.gate_mode,
          '; clean probability:', cli.clean_probability,
          '; max rotation degrees:', cli.rotation_max_deg)
    print('Rotation labels are diagnostics only; rot_unc=None in training and evaluation.')
    if cli.eval_only:
        rows = validate(model, val_loader, device, cli)
        (output / 'evaluation.json').write_text(json.dumps(rows, indent=2))
        return

    for epoch in range(epoch_start, cli.epochs):
        # Seed nonpersistent loader workers from loader_rng on every epoch.
        # Also seed the main-process augmentations when num_workers=0.
        random.seed(cli.seed + epoch)
        np.random.seed(cli.seed + epoch)
        set_train_mode(model)
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cli.epochs}')
        for index, batch in enumerate(progress):
            if cli.max_train_steps and index >= cli.max_train_steps:
                break
            ref, sources, poses, valid_views, intrinsics, gt = prepare_batch(batch, device)
            poses, angles = perturb_rotations(poses, cli.clean_probability,
                                             cli.rotation_max_deg, noise_rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cli.amp):
                predictions, aux = model(ref, sources, poses, valid_views, intrinsics,
                                         mode='train', rot_unc=None, return_aux=True)
            # Existing Gaussian objective matches MaGNet's sigma parameterization.
            loss = depth_loss(predictions, gt, cli.min_depth, cli.max_depth, cli.loss_gamma)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training depth loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, cli.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            row = dict(epoch=epoch + 1, step=step, depth_nll=loss.item(),
                       gate_mean=torch.stack([g.detach().mean() for g in aux['geometry_gate']]).mean().item(),
                       update_abs_mean=torch.stack([u.abs().mean() for u in aux['depth_update']]).mean().item(),
                       noise_mean_deg=angles.mean().item(),
                       clean_fraction=(angles == 0).all(dim=1).float().mean().item(),
                       lr=optimizer.param_groups[0]['lr'])
            append_csv(output / 'train.csv', row)
            progress.set_postfix(loss=f'{loss.item():.4f}', gate=f'{row["gate_mean"]:.3f}')
        results = validate(model, val_loader, device, cli)
        score = sum(r['rmse'] for r in results) / len(results)
        for row in results:
            append_csv(output / 'val.csv', dict(epoch=epoch + 1, **row, score=score))
        improved = score < best_score
        best_score = min(best_score, score)
        for filename in (['last.pt', 'best.pt'] if improved else ['last.pt']):
            save_checkpoint(output / filename, model, optimizer, scaler, epoch,
                            step, best_score, vars(cli), noise_rng, loader_rng)
    print('Finished. Checkpoints and metrics:', output)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--scans_dir', default='scans')
    parser.add_argument('--train_split', default='./data_split/scannet_train_stride5.txt')
    parser.add_argument('--val_split', default='./data_split/scannet_val_stride5.txt')
    parser.add_argument('--raw_wh_json', default='./data/scannet_raw_WH.json')
    for name, value in [('window_radius', 20), ('num_source_views', 4),
                        ('input_height', 480), ('input_width', 640),
                        ('dpv_height', 120), ('dpv_width', 160),
                        ('batch_size', 1), ('num_workers', 1), ('epochs', 1),
                        ('num_train_iter', 3), ('num_test_iter', 3),
                        ('max_train_steps', 0), ('val_max_samples', 32), ('seed', 1234)]:
        parser.add_argument('--' + name, type=int, default=value)
    for name, value in [('lr', 1e-4), ('weight_decay', 1e-5), ('grad_clip', 1.),
                        ('loss_gamma', .8), ('min_depth', 1e-3), ('max_depth', 10.),
                        ('clean_probability', .5), ('rotation_max_deg', 5.)]:
        parser.add_argument('--' + name, type=float, default=value)
    parser.add_argument('--dnet_ckpt', default='./ckpts/DNET_scannet.pt')
    parser.add_argument('--fnet_ckpt', default='./ckpts/FNET_scannet.pt')
    parser.add_argument('--magnet_ckpt', default='./ckpts/MAGNET_scannet.pt')
    parser.add_argument('--gate_mode', choices=['learned', 'off'], default='learned')
    parser.add_argument('--val_degrees', type=float, nargs='+', default=[0., 5., 8.])
    for flag in ('amp', 'pin_memory', 'color_aug', 'eval_only'):
        parser.add_argument('--' + flag, action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume')
    parser.add_argument('--output_dir', default='./exp/STRUCTMAGNET/scannet/second_training')
    return parser


def main():
    parser = build_parser()
    cli = parser.parse_args()
    import math
    if not 0 <= cli.clean_probability <= 1 or not math.isfinite(cli.rotation_max_deg) or cli.rotation_max_deg < 0:
        parser.error('Invalid clean_probability or rotation_max_deg')
    if any(not math.isfinite(x) or x < 0 for x in cli.val_degrees):
        parser.error('val_degrees must be finite and nonnegative')
    if not 0 < cli.loss_gamma <= 1 or not 0 < cli.min_depth < cli.max_depth:
        parser.error('Invalid loss_gamma or depth bounds')
    if not all(math.isfinite(v) for v in (cli.lr, cli.weight_decay, cli.grad_clip, cli.max_depth)):
        parser.error('Optimizer parameters and max_depth must be finite')
    if cli.lr <= 0 or cli.grad_clip <= 0 or cli.weight_decay < 0:
        parser.error('Invalid optimizer parameters')
    if min(cli.batch_size, cli.epochs, cli.num_train_iter, cli.num_test_iter) < 1:
        parser.error('Batch size, epochs and iteration counts must be positive')
    if min(cli.num_workers, cli.max_train_steps, cli.val_max_samples) < 0:
        parser.error('Worker and step/sample limits must be nonnegative')
    if (cli.input_height, cli.input_width) != (4 * cli.dpv_height, 4 * cli.dpv_width):
        parser.error('Input height/width must be four times DPV height/width')
    if cli.eval_only and not cli.resume:
        parser.error('--eval_only requires --resume')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cli.gpu
    for name in ('dataset_root', 'train_split', 'val_split', 'raw_wh_json',
                 'dnet_ckpt', 'fnet_ckpt', 'magnet_ckpt', 'output_dir', 'resume'):
        if getattr(cli, name) is not None:
            setattr(cli, name, os.path.expanduser(getattr(cli, name)))
    _check_inputs(cli)
    train(cli)


if __name__ == '__main__':
    main()
