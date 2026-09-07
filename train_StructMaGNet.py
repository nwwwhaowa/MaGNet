#!/usr/bin/env python3
"""Phase-A GeometryGate training for StructMaGNet on 7-Scenes Chess + Office.

Stage goal:
    - Keep D-Net, F-Net, G-Net and learned upsampling frozen.
    - Train GeometryGate only.
    - Inject rotation-only perturbations online.
    - Preserve translation exactly.
    - Supervise the gate by multi-view utility:

        y_g = sigmoid((e_mono - e_mv) / tau)

      where e_mv is computed from the frozen ungated MaGNet proposal.
    - Validate at fixed 0 deg, 5 deg and 8 deg rotation perturbations.
- Report Gate iteration diagnostics and oracle-target alignment.
- Support --eval_only to diagnose an existing gate checkpoint without training.

Run this file from the MaGNet repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Locate repository root independent of current working directory.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = None
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    if (
        (_parent / 'models').is_dir()
        and (_parent / 'data').is_dir()
        and (_parent / 'utils').is_dir()
    ):
        _REPO_ROOT = _parent
        break

if _REPO_ROOT is None:
    raise RuntimeError(
        f'Cannot locate MaGNet repository root from {_THIS_FILE}'
    )

sys.path.insert(0, str(_REPO_ROOT))

import utils.utils as utils
from data.dataloader_7scenes_train import SevenScenesTrainLoader
from models.STRUCTMAGNET import STRUCTMAGNET
from utils.losses import MagnetLoss


def build_model_args(cli):
    return SimpleNamespace(
        # output
        output_dim=2,
        output_type='G',
        downsample_ratio=4,

        # D-Net
        DNET_architecture='DenseDepth_BN',
        DNET_fix_encoder_weights='None',
        DNET_ckpt=cli.dnet_ckpt,

        # F-Net
        FNET_architecture='PSM-Net',
        FNET_feature_dim=64,
        FNET_ckpt=cli.fnet_ckpt,

        # MaGNet
        MAGNET_sampling_range=3,
        MAGNET_num_samples=5,
        MAGNET_mvs_weighting='CW5',
        MAGNET_num_train_iter=cli.num_train_iter,
        MAGNET_num_test_iter=cli.num_test_iter,
        MAGNET_window_radius=cli.window_radius,
        MAGNET_num_source_views=cli.num_source_views,
        MAGNET_ckpt=cli.magnet_ckpt,

        # loss
        loss_fn='gaussian',
        loss_gamma=cli.loss_gamma,

        # dataset
        dataset_name='7scenes',
        dataset_path=os.path.expanduser(cli.dataset_root),
        seven_scenes=cli.scenes,
        seven_val_sequences=cli.val_sequences,
        seven_frame_stride=cli.frame_stride,
        input_height=480,
        input_width=640,
        dpv_height=120,
        dpv_width=160,
        min_depth=cli.min_depth,
        max_depth=cli.max_depth,

        # loader/shared attributes
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        workers=cli.num_workers,
        num_threads=cli.num_workers,
        mode='train',
        distributed=False,
        do_kb_crop=False,
        eigen_crop=False,
        garg_crop=False,
        data_augmentation_color=False,
        pin_memory=True,
    )


def strip_module_prefix(state_dict):
    out = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        out[key] = value
    return out


def load_compatible_backbone(checkpoint_path, model):
    """Load all matching original MaGNet tensors, keep GeometryGate init."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    ckpt = strip_module_prefix(ckpt)

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in ckpt.items()
        if key in current
        and current[key].shape == value.shape
    }

    result = model.load_state_dict(compatible, strict=False)
    print(f'loaded original MaGNet tensors: {len(compatible)}')
    print(f'missing tensors after partial load: {len(result.missing_keys)}')
    if result.unexpected_keys:
        print('unexpected tensors:', result.unexpected_keys[:10])
    return model


def freeze_phase_a(model):
    """Freeze everything, then enable GeometryGate only."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.geometry_gate.parameters():
        param.requires_grad = True

    trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    if not trainable:
        raise RuntimeError('No trainable GeometryGate parameters found.')

    bad = [
        name for name in trainable
        if not name.startswith('geometry_gate.')
    ]
    if bad:
        raise RuntimeError(
            f'Phase A unexpectedly has non-gate trainable tensors: {bad}'
        )

    print(f'trainable tensors: {len(trainable)}')
    print('trainable module : GeometryGate only')


def set_phase_a_train_mode(model):
    model.train()
    # model.train() recursively toggles frozen backbones, so restore them.
    model.d_net.eval()
    model.f_net.eval()
    model.g_net.eval()
    model.mask_head.eval()
    model.geometry_gate.train()


def _skew(rotvec):
    """rotvec: (..., 3) -> (..., 3, 3)."""
    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ],
        dim=-1,
    ).reshape(*rotvec.shape[:-1], 3, 3)


def so3_exp(rotvec):
    # torch.matrix_exp is stable and sufficient for the small perturbations here.
    return torch.matrix_exp(_skew(rotvec))


def sample_train_rotation_noise(poses):
    """Online Phase-A rotation augmentation.

    Per sample category:
      25% clean   : 0 deg
      25% mild    : 0.5-2 deg
      30% medium  : 2-5 deg
      20% severe  : 5-8 deg

    Each source view gets an independent random axis and angle inside the
    sample's selected severity interval. Translation is copied unchanged.
    """
    B, V = poses.shape[:2]
    device = poses.device
    dtype = poses.dtype

    severity = torch.rand(B, device=device)
    angle_deg = torch.zeros(B, V, device=device, dtype=dtype)

    for b in range(B):
        r = float(severity[b].item())
        if r < 0.25:
            lo, hi = 0.0, 0.0
        elif r < 0.50:
            lo, hi = 0.5, 2.0
        elif r < 0.80:
            lo, hi = 2.0, 5.0
        else:
            lo, hi = 5.0, 8.0

        if hi > 0.0:
            angle_deg[b] = (
                lo
                + (hi - lo)
                * torch.rand(V, device=device, dtype=dtype)
            )

    angle_rad = angle_deg * (math.pi / 180.0)

    axis = torch.randn(B, V, 3, device=device, dtype=dtype)
    axis = axis / torch.linalg.norm(
        axis,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)

    rotvec = axis * angle_rad.unsqueeze(-1)
    dR = so3_exp(rotvec)

    noisy = poses.clone()
    R = poses[:, :, :3, :3]

    # Follow the controlled-noise convention R_noisy = R * Exp(epsilon^).
    noisy[:, :, :3, :3] = torch.matmul(R, dR)

    # IMPORTANT: translation is not modified.
    noisy[:, :, :3, 3] = poses[:, :, :3, 3]

    return noisy, angle_rad, axis


def fixed_validation_rotation_noise(poses, angle_deg):
    """Deterministic fixed-axis validation perturbation."""
    B, V = poses.shape[:2]
    device = poses.device
    dtype = poses.dtype

    angle_rad = torch.full(
        (B, V),
        float(angle_deg) * math.pi / 180.0,
        device=device,
        dtype=dtype,
    )

    base_axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    base_axes = base_axes / torch.linalg.norm(
        base_axes,
        dim=-1,
        keepdim=True,
    )
    if V > base_axes.shape[0]:
        repeats = math.ceil(V / base_axes.shape[0])
        base_axes = base_axes.repeat(repeats, 1)
    axis = base_axes[:V].unsqueeze(0).repeat(B, 1, 1)

    rotvec = axis * angle_rad.unsqueeze(-1)
    dR = so3_exp(rotvec)

    noisy = poses.clone()
    noisy[:, :, :3, :3] = torch.matmul(
        poses[:, :, :3, :3],
        dR,
    )
    noisy[:, :, :3, 3] = poses[:, :, :3, 3]
    return noisy, angle_rad, axis


def last_aux_tensor(aux, key):
    value = aux.get(key)
    if value is None:
        raise KeyError(
            f'STRUCTMAGNET return_aux is missing required key: {key}'
        )
    if isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError(f'aux[{key!r}] is empty')
        return value[-1]
    return value


def compute_gate_oracle_diagnostics(aux, gt_depth, args):
    """Compute Phase-A oracle diagnostics for the supervised first gate.

    The Phase-A loss supervises geometry_gate[0] against the optimal
    interpolation coefficient between the frozen monocular prediction and
    the first frozen ungated multi-view proposal. This helper evaluates the
    same target during validation and reports how well gate[0] follows it.
    """
    gates = aux.get('geometry_gate')
    ungated = aux.get('ungated_gmm')
    mono_gmm = aux.get('mono_gmm')

    if not isinstance(gates, (list, tuple)) or not gates:
        raise RuntimeError('aux[geometry_gate] must be a non-empty list')
    if not isinstance(ungated, (list, tuple)) or not ungated:
        raise RuntimeError('aux[ungated_gmm] must be a non-empty list')
    if mono_gmm is None:
        raise RuntimeError('aux[mono_gmm] is required')

    gate1 = gates[0]
    raw_mv = ungated[0]

    mono_mu = mono_gmm[:, :1].detach()
    raw_mv_mu = raw_mv[:, :1].detach()

    gt_low = F.interpolate(
        gt_depth,
        size=gate1.shape[-2:],
        mode='nearest',
    )

    delta = raw_mv_mu - mono_mu
    valid = (
        (gt_low > args.min_depth)
        & (gt_low < args.max_depth)
        & torch.isfinite(gt_low)
        & torch.isfinite(mono_mu)
        & torch.isfinite(raw_mv_mu)
        & torch.isfinite(gate1)
        & (torch.abs(delta) > args.gate_min_delta)
    )

    if not torch.any(valid):
        return {
            'target_mean': float('nan'),
            'target_std': float('nan'),
            'mae': float('nan'),
            'corr': float('nan'),
            'valid_pixels': 0,
        }

    numerator = (gt_low - mono_mu) * delta
    denominator = delta.square() + 1e-6
    target = (numerator / denominator).clamp(0.0, 1.0)
    target = torch.nan_to_num(
        target,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )

    gate_v = torch.nan_to_num(
        gate1[valid].float(),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    target_v = target[valid].float()

    mae = torch.mean(torch.abs(gate_v - target_v))

    gate_centered = gate_v - gate_v.mean()
    target_centered = target_v - target_v.mean()
    denom = torch.sqrt(
        torch.sum(gate_centered.square())
        * torch.sum(target_centered.square())
    )
    if float(denom.item()) > 1e-12:
        corr = torch.sum(gate_centered * target_centered) / denom
        corr_value = float(corr.item())
    else:
        corr_value = float('nan')

    return {
        'target_mean': float(target_v.mean().item()),
        'target_std': float(target_v.std(unbiased=False).item()),
        'mae': float(mae.item()),
        'corr': corr_value,
        'valid_pixels': int(target_v.numel()),
    }


def _gate_iteration_means(aux, max_iters=3):
    gates = aux.get('geometry_gate')
    if not isinstance(gates, (list, tuple)) or not gates:
        raise RuntimeError('aux[geometry_gate] must be a non-empty list')

    means = []
    for i in range(max_iters):
        if i < len(gates):
            means.append(float(gates[i].detach().mean().item()))
        else:
            means.append(float('nan'))
    return means


def compute_gate_utility_loss(aux, gt_depth, args, tau):
    """Oracle interpolation supervision for Phase-A GeometryGate.

    For the first MaGNet refinement step:
        mu_gate = mu_mono + g * (mu_mv - mu_mono)

    The least-squares optimal per-pixel interpolation coefficient is:
        g* = ((gt - mu_mono) * delta) / (delta^2 + eps)
    clipped to [0, 1], where delta = mu_mv - mu_mono.

    This directly matches the semantics of GeometryGate. Pixels for which
    the frozen MV proposal is nearly identical to the monocular prior are
    ignored because the gate is unidentifiable there.
    """
    gates = aux.get('geometry_gate')
    ungated = aux.get('ungated_gmm')
    mono_gmm = aux.get('mono_gmm')

    if not isinstance(gates, (list, tuple)) or not gates:
        raise RuntimeError('aux[geometry_gate] must be a non-empty list')
    if not isinstance(ungated, (list, tuple)) or not ungated:
        raise RuntimeError('aux[ungated_gmm] must be a non-empty list')
    if mono_gmm is None:
        raise RuntimeError('aux[mono_gmm] is required')

    # Phase-A warm-up supervises the first refinement only. At this step
    # the base prediction is exactly the frozen monocular Gaussian, which
    # avoids target feedback from already-gated later iterations.
    gate = gates[0]
    raw_mv = ungated[0]

    mono_mu = mono_gmm[:, :1].detach()
    raw_mv_mu = raw_mv[:, :1].detach()

    gt_low = F.interpolate(
        gt_depth,
        size=gate.shape[-2:],
        mode='nearest',
    )

    delta = raw_mv_mu - mono_mu

    valid = (
        (gt_low > args.min_depth)
        & (gt_low < args.max_depth)
        & torch.isfinite(gt_low)
        & torch.isfinite(mono_mu)
        & torch.isfinite(raw_mv_mu)
        # If the MV proposal barely moves relative to mono, the gate has
        # almost no observable effect and should not be supervised.
        & (torch.abs(delta) > args.gate_min_delta)
    )

    if not torch.any(valid):
        raise RuntimeError('No valid pixels for oracle gate supervision')

    with torch.no_grad():
        numerator = (gt_low - mono_mu) * delta
        denominator = delta.square() + 1e-6

        target = (numerator / denominator).clamp(0.0, 1.0)
        target = torch.nan_to_num(
            target,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )

    # The target is an interpolation coefficient rather than a Bernoulli
    # label, so Smooth-L1 is a better fit than BCE for Phase-A warm-up.
    with torch.cuda.amp.autocast(enabled=False):
        gate_fp32 = torch.nan_to_num(
            gate[valid].float(),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        target_fp32 = target[valid].float()

        loss_gate = F.smooth_l1_loss(
            gate_fp32,
            target_fp32,
            beta=0.1,
        )

    target_mean = target_fp32.mean()
    target_std = target_fp32.std(unbiased=False)

    return loss_gate, target_mean, target_std



def stable_magnet_loss(pred_list, gt_depth, min_depth, max_depth, gamma):
    """Numerically stable version of MaGNet's Gaussian NLL.

    The mathematical form is unchanged:
        (mu - gt)^2 / (2 sigma^2) + 0.5 log(sigma^2)

    Differences from the original training helper:
      1) NLL is evaluated in FP32 even when AMP is enabled.
      2) Non-finite prediction/GT pixels are excluded.
      3) sigma is clamped away from zero before forming the variance.
    """
    with torch.cuda.amp.autocast(enabled=False):
        gt = gt_depth.float()

        base_valid = (
            (gt > min_depth)
            & (gt < max_depth)
            & torch.isfinite(gt)
        )

        total_loss = gt.new_tensor(0.0)
        valid_predictions = 0

        n_predictions = len(pred_list)

        for i, pred in enumerate(pred_list):
            pred_fp32 = pred.float()
            mu, sigma = torch.split(pred_fp32, 1, dim=1)

            valid = (
                base_valid
                & torch.isfinite(mu)
                & torch.isfinite(sigma)
            )

            if not torch.any(valid):
                continue

            mu_v = mu[valid]
            sigma_v = sigma[valid]
            gt_v = gt[valid]

            # Sigma is theoretically positive. Clamp only for numerical
            # stability; the upper bound prevents extreme random-pose
            # outliers from overflowing the NLL.
            sigma_v = sigma_v.abs().clamp(
                min=1e-4,
                max=max_depth,
            )

            # Keep extreme but finite corrupted-pose proposals from
            # overflowing the squared term in FP32.
            mu_v = mu_v.clamp(
                min=-10.0 * max_depth,
                max=10.0 * max_depth,
            )

            var_v = sigma_v.square().clamp_min(1e-8)

            nll = (
                (mu_v - gt_v).square() / (2.0 * var_v)
                + 0.5 * torch.log(var_v)
            )

            finite_nll = torch.isfinite(nll)
            if not torch.any(finite_nll):
                continue

            i_weight = gamma ** (n_predictions - i - 1)
            total_loss = (
                total_loss
                + i_weight * nll[finite_nll].mean()
            )
            valid_predictions += 1

        if valid_predictions == 0:
            raise RuntimeError(
                "Stable MaGNet loss found no finite prediction pixels."
            )

        return total_loss



def depth_metrics(pred, gt, min_depth, max_depth):
    pred_mu, pred_sigma = torch.split(pred, 1, dim=1)
    valid = (
        (gt > min_depth)
        & (gt < max_depth)
        & torch.isfinite(gt)
    )

    if not torch.any(valid):
        return None

    p = pred_mu[valid].clamp(min=min_depth, max=max_depth)
    g = gt[valid]
    sigma = pred_sigma[valid].clamp_min(1e-6)

    rmse = torch.sqrt(torch.mean((p - g) ** 2))
    abs_rel = torch.mean(torch.abs(p - g) / g.clamp_min(1e-6))
    ratio = torch.maximum(
        p / g.clamp_min(1e-6),
        g / p.clamp_min(1e-6),
    )
    a1 = torch.mean((ratio < 1.25).float())

    var = sigma ** 2
    nll = torch.mean(
        ((p - g) ** 2) / (2.0 * var)
        + 0.5 * torch.log(var)
    )

    return {
        'rmse': float(rmse.item()),
        'abs_rel': float(abs_rel.item()),
        'a1': float(a1.item()),
        'nll': float(nll.item()),
    }


def append_csv(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_gate_checkpoint(path, model, optimizer, epoch, step, metrics, args):
    """Small checkpoint: only the trainable GeometryGate + optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_state = {
        f'geometry_gate.{key}': value.detach().cpu()
        for key, value in model.geometry_gate.state_dict().items()
    }
    torch.save(
        {
            'model': gate_state,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'args': vars(args),
        },
        path,
    )


def load_gate_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = strip_module_prefix(state)

    gate_state = {}
    for key, value in state.items():
        if key.startswith('geometry_gate.'):
            gate_state[key[len('geometry_gate.'):]] = value
        else:
            gate_state[key] = value

    model.geometry_gate.load_state_dict(gate_state, strict=True)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt


def validate(model, loader, device, args, fixed_deg, max_samples):
    model.eval()

    totals = {
        'rmse': 0.0,
        'abs_rel': 0.0,
        'a1': 0.0,
        'nll': 0.0,
        # Keep gate_mean as the LAST iteration for backward compatibility
        # with the earlier val.csv format.
        'gate_mean': 0.0,
        'gate1_mean': 0.0,
        'gate2_mean': 0.0,
        'gate3_mean': 0.0,
        'gate_target_mean': 0.0,
        'gate_target_std': 0.0,
        'gate_target_mae': 0.0,
        'forward_ms': 0.0,
    }
    corr_sum = 0.0
    corr_count = 0
    count = 0

    with torch.no_grad():
        for batch_idx, (data_array, cam_intrins) in enumerate(loader):
            if max_samples > 0 and count >= max_samples:
                break

            cur_batch_size = data_array[0]['img'].shape[0]
            ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(
                data_array,
                cur_batch_size,
            )

            ref_img = ref_dat['img'].to(device, non_blocking=True)
            gt = ref_dat['gt_dmap'].to(device, non_blocking=True)
            nghbr_imgs = torch.cat(
                [d['img'].to(device, non_blocking=True) for d in nghbr_dats],
                dim=0,
            )
            poses = nghbr_poses.to(device, non_blocking=True)

            noisy_poses, rot_unc, _ = fixed_validation_rotation_noise(
                poses,
                fixed_deg,
            )

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred_list, aux = model(
                ref_img,
                nghbr_imgs,
                noisy_poses,
                is_valid,
                cam_intrins,
                mode='test',
                rot_unc=rot_unc,
                return_aux=True,
            )

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            metrics = depth_metrics(
                pred_list[-1],
                gt,
                args.min_depth,
                args.max_depth,
            )
            if metrics is None:
                continue

            gates = aux.get('geometry_gate')
            if not isinstance(gates, (list, tuple)) or not gates:
                raise RuntimeError('aux[geometry_gate] must be a non-empty list')

            gate_last = gates[-1]
            gate_means = _gate_iteration_means(aux, max_iters=3)
            oracle = compute_gate_oracle_diagnostics(aux, gt, args)

            for key in ('rmse', 'abs_rel', 'a1', 'nll'):
                totals[key] += metrics[key]

            totals['gate_mean'] += float(gate_last.mean().item())
            totals['gate1_mean'] += gate_means[0]
            totals['gate2_mean'] += gate_means[1]
            totals['gate3_mean'] += gate_means[2]
            totals['gate_target_mean'] += oracle['target_mean']
            totals['gate_target_std'] += oracle['target_std']
            totals['gate_target_mae'] += oracle['mae']
            totals['forward_ms'] += elapsed_ms

            if math.isfinite(oracle['corr']):
                corr_sum += oracle['corr']
                corr_count += 1

            count += 1

    if count == 0:
        raise RuntimeError('Validation produced zero valid samples')

    result = {
        key: value / count
        for key, value in totals.items()
    }
    result['gate_target_corr'] = (
        corr_sum / corr_count if corr_count > 0 else float('nan')
    )
    result['samples'] = count
    result['noise_deg'] = float(fixed_deg)
    return result


# === StructMaGNet CLI propagation fix ===
def _merge_cli_into_args(model_args, cli):
    # Copy all CLI options first, then let derived/model-specific attributes
    # from build_model_args() take precedence.
    merged = dict(vars(cli))
    merged.update(vars(model_args))
    return SimpleNamespace(**merged)


def _validate_phase_a_args(args):
    # Fail before entering the training loop if an expected Phase-A option
    # was not propagated.
    required = (
        "dataset_root",
        "scenes",
        "frame_stride",
        "window_radius",
        "num_source_views",
        "batch_size",
        "epochs",
        "lr",
        "lambda_gate",
        "gate_min_delta",
        "max_train_steps",
        "val_max_samples",
    )

    missing = [name for name in required if not hasattr(args, name)]
    if missing:
        raise ValueError(
            "Missing required Phase-A training arguments: "
            + ", ".join(missing)
            + ". Check argparse definitions and CLI propagation."
        )


def _print_phase_a_args(args):
    print(f"learning rate     : {args.lr}")
    print(f"lambda gate       : {args.lambda_gate}")
    print(f"gate min delta    : {args.gate_min_delta}")
    print(f"max train steps   : {args.max_train_steps}")
    print(f"val max samples   : {args.val_max_samples}")
# === end StructMaGNet CLI propagation fix ===


def train(cli):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cli.gpu

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this StructMaGNet training script')

    device = torch.device('cuda:0')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    args = build_model_args(cli)
    args = _merge_cli_into_args(args, cli)
    _validate_phase_a_args(args)
    _print_phase_a_args(args)
    out_dir = Path(cli.output_dir)
    ckpt_dir = out_dir / 'checkpoints'
    log_dir = out_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / 'config.json').open('w') as f:
        json.dump(vars(cli), f, indent=2)

    print('=== Phase A1: GeometryGate oracle warm-up ===')
    print('dataset          :', args.dataset_path)
    print('scenes           :', args.seven_scenes)
    print('frame stride     :', args.seven_frame_stride)
    print('window radius    :', args.MAGNET_window_radius)
    print('source views     :', args.MAGNET_num_source_views)
    print('batch size       :', args.batch_size)
    print('gate lr          :', cli.lr)
    print('lambda gate      :', cli.lambda_gate)
    print('gate target tau  :', cli.gate_tau, '(unused by current oracle coefficient)')
    print('device           :', torch.cuda.get_device_name(0))

    train_loader = SevenScenesTrainLoader(args, 'train').data
    val_loader = SevenScenesTrainLoader(args, 'val').data

    print('train samples    :', len(train_loader.dataset))
    print('val samples      :', len(val_loader.dataset))

    model = STRUCTMAGNET(args).to(device)
    model = load_compatible_backbone(cli.magnet_ckpt, model)
    freeze_phase_a(model)

    optimizer = torch.optim.AdamW(
        model.geometry_gate.parameters(),
        lr=cli.lr,
        weight_decay=cli.weight_decay,
    )

    # Keep the original Gaussian NLL formulation, but evaluate it through
    # a Phase-A stable FP32 wrapper because random rotation perturbations can
    # produce isolated extreme sigma/mu values under AMP.
    scaler = torch.cuda.amp.GradScaler(enabled=cli.amp)

    start_epoch = 0
    global_step = 0
    resumed_best_score = None
    if cli.resume:
        ckpt = load_gate_checkpoint(cli.resume, model, optimizer)
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        global_step = int(ckpt.get('step', 0))
        metrics = ckpt.get('metrics', {})
        if isinstance(metrics, dict):
            resumed_best_score = metrics.get('score')
        print('resumed gate checkpoint:', cli.resume)
        print('resume epoch      :', start_epoch)

    train_csv = log_dir / 'train.csv'
    val_csv = log_dir / 'val.csv'
    val_diag_csv = log_dir / 'val_diagnostics.csv'

    train_fields = [
        'epoch', 'step', 'loss_total', 'loss_depth', 'loss_gate',
        'gate_mean', 'gate_std', 'gate_target_mean', 'gate_target_std',
        'noise_mean_deg', 'lr',
    ]
    val_fields = [
        'epoch', 'noise_deg', 'samples', 'rmse', 'abs_rel', 'a1',
        'nll', 'gate_mean', 'forward_ms', 'score',
    ]

    best_score = (
        float(resumed_best_score)
        if resumed_best_score is not None
        else float('inf')
    )

    if cli.eval_only:
        if not cli.resume:
            raise ValueError('--eval_only requires --resume CHECKPOINT')

        print('\n=== Gate diagnostic evaluation only ===')
        eval_results = []
        for fixed_deg in (0.0, 5.0, 8.0):
            result = validate(
                model, val_loader, device, args,
                fixed_deg=fixed_deg,
                max_samples=cli.val_max_samples,
            )
            eval_results.append(result)
            print(
                ('%g deg : RMSE=%.4f AbsRel=%.4f a1=%.4f | '
                 'g1=%.4f g2=%.4f g3=%.4f | '
                 'target=%.4f±%.4f MAE=%.4f corr=%.4f')
                % (
                    fixed_deg,
                    result['rmse'],
                    result['abs_rel'],
                    result['a1'],
                    result['gate1_mean'],
                    result['gate2_mean'],
                    result['gate3_mean'],
                    result['gate_target_mean'],
                    result['gate_target_std'],
                    result['gate_target_mae'],
                    result['gate_target_corr'],
                )
            )

        diag_fields = [
            'noise_deg', 'samples', 'rmse', 'abs_rel', 'a1', 'nll',
            'gate_mean', 'gate1_mean', 'gate2_mean', 'gate3_mean',
            'gate_target_mean', 'gate_target_std',
            'gate_target_mae', 'gate_target_corr', 'forward_ms',
        ]
        # Rewrite this small diagnostic file on every eval-only run.
        if val_diag_csv.exists():
            val_diag_csv.unlink()
        for result in eval_results:
            append_csv(
                val_diag_csv,
                diag_fields,
                {key: result[key] for key in diag_fields},
            )
        print('diagnostic CSV    :', val_diag_csv)
        print('[PASS] Gate diagnostic evaluation completed.')
        return

    for epoch in range(start_epoch, cli.epochs):
        set_phase_a_train_mode(model)

        running = {
            'loss_total': 0.0,
            'loss_depth': 0.0,
            'loss_gate': 0.0,
            'gate_mean': 0.0,
            'gate_std': 0.0,
            'target_mean': 0.0,
            'target_std': 0.0,
            'noise_deg': 0.0,
        }
        running_count = 0

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{cli.epochs} Train',
        )

        for batch_idx, (data_array, cam_intrins) in enumerate(pbar):
            if cli.max_train_steps > 0 and batch_idx >= cli.max_train_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            cur_batch_size = data_array[0]['img'].shape[0]
            ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(
                data_array,
                cur_batch_size,
            )

            ref_img = ref_dat['img'].to(device, non_blocking=True)
            gt = ref_dat['gt_dmap'].to(device, non_blocking=True)
            gt = gt.clone()
            gt[~torch.isfinite(gt)] = 0.0
            gt[gt > args.max_depth] = 0.0
            gt_mask = (
                (gt > args.min_depth)
                & (gt < args.max_depth)
            )

            nghbr_imgs = torch.cat(
                [d['img'].to(device, non_blocking=True) for d in nghbr_dats],
                dim=0,
            )
            poses = nghbr_poses.to(device, non_blocking=True)

            noisy_poses, rot_unc, _ = sample_train_rotation_noise(poses)

            with torch.cuda.amp.autocast(enabled=cli.amp):
                pred_list, aux = model(
                    ref_img,
                    nghbr_imgs,
                    noisy_poses,
                    is_valid,
                    cam_intrins,
                    mode='train',
                    rot_unc=rot_unc,
                    return_aux=True,
                )

                loss_depth = stable_magnet_loss(
                    pred_list,
                    gt,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    gamma=args.loss_gamma,
                )

                loss_gate, target_mean, target_std = compute_gate_utility_loss(
                    aux,
                    gt,
                    args,
                    cli.gate_tau,
                )

                # Phase-A warm-up: optimize GeometryGate supervision only.
                # Depth NLL is still computed/logged as a diagnostic and will
                # return to the objective in the later gate-depth adaptation phase.
                loss_total = cli.lambda_gate * loss_gate

            if global_step == 0:
                pred_dbg = pred_list[-1].detach()
                mu_dbg, sigma_dbg = torch.split(
                    pred_dbg,
                    1,
                    dim=1,
                )
                print('\n--- first-step depth-loss check ---')
                print(
                    'pred finite      :',
                    bool(torch.isfinite(pred_dbg).all().item()),
                )
                print(
                    'mu finite        :',
                    bool(torch.isfinite(mu_dbg).all().item()),
                )
                print(
                    'sigma finite     :',
                    bool(torch.isfinite(sigma_dbg).all().item()),
                )

                finite_sigma = sigma_dbg[
                    torch.isfinite(sigma_dbg)
                ]
                if finite_sigma.numel() > 0:
                    print(
                        'sigma min/max   : %.6e / %.6e'
                        % (
                            float(finite_sigma.min().item()),
                            float(finite_sigma.max().item()),
                        )
                    )

                print(
                    'loss_depth finite:',
                    bool(torch.isfinite(loss_depth).item()),
                )
                print(
                    'loss_gate finite :',
                    bool(torch.isfinite(loss_gate).item()),
                )
                print(
                    'loss_total finite:',
                    bool(torch.isfinite(loss_total).item()),
                )

            if not torch.isfinite(loss_total):
                raise RuntimeError(
                    'Non-finite total loss after stable FP32 filtering. '
                    'See the first-step diagnostics above.'
                )

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.geometry_gate.parameters(),
                cli.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            gates_diag = aux.get('geometry_gate')
            if not isinstance(gates_diag, (list, tuple)) or not gates_diag:
                raise RuntimeError('aux[geometry_gate] must be a non-empty list')
            # Phase-A supervises the FIRST gate. Use it for training logs.
            gate = gates_diag[0]
            gate_last = gates_diag[-1]

            # First-step numerical diagnostics. These should all be finite.
            if global_step == 1:
                entropy_dbg = last_aux_tensor(aux, 'cost_entropy')
                peak_dbg = last_aux_tensor(aux, 'cost_peak')
                print('\n--- first-step finite check ---')
                print('gate1 finite     :', bool(torch.isfinite(gate).all().item()))
                print('gateLast finite  :', bool(torch.isfinite(gate_last).all().item()))
                print('entropy finite   :', bool(torch.isfinite(entropy_dbg).all().item()))
                print('peak finite      :', bool(torch.isfinite(peak_dbg).all().item()))
                print(
                    'gate1 min/max    : %.6f / %.6f'
                    % (
                        float(gate.detach().min().item()),
                        float(gate.detach().max().item()),
                    )
                )

            gate_mean = float(gate.detach().mean().item())
            gate_std = float(gate.detach().std(unbiased=False).item())
            noise_mean_deg = float(
                rot_unc.detach().mean().item()
                * 180.0 / math.pi
            )

            row = {
                'epoch': epoch + 1,
                'step': global_step,
                'loss_total': float(loss_total.detach().item()),
                'loss_depth': float(loss_depth.detach().item()),
                'loss_gate': float(loss_gate.detach().item()),
                'gate_mean': gate_mean,
                'gate_std': gate_std,
                'gate_target_mean': float(target_mean.detach().item()),
                'gate_target_std': float(target_std.detach().item()),
                'noise_mean_deg': noise_mean_deg,
                'lr': optimizer.param_groups[0]['lr'],
            }
            append_csv(train_csv, train_fields, row)

            for key in ('loss_total', 'loss_depth', 'loss_gate'):
                running[key] += row[key]
            running['gate_mean'] += gate_mean
            running['gate_std'] += gate_std
            running['target_mean'] += row['gate_target_mean']
            running['target_std'] += row['gate_target_std']
            running['noise_deg'] += noise_mean_deg
            running_count += 1

            pbar.set_postfix(
                loss=f'{row["loss_total"]:.4f}',
                gate=f'{gate_mean:.3f}',
                gstd=f'{gate_std:.3f}',
                tgt=f'{row["gate_target_mean"]:.3f}',
                rot=f'{noise_mean_deg:.2f}deg',
            )

        if running_count == 0:
            raise RuntimeError('No training steps were executed')

        print('\n--- epoch train summary ---')
        print('loss_total       : %.6f' % (running['loss_total'] / running_count))
        print('loss_depth       : %.6f' % (running['loss_depth'] / running_count))
        print('loss_gate        : %.6f' % (running['loss_gate'] / running_count))
        print('gate_mean        : %.4f' % (running['gate_mean'] / running_count))
        print('gate_std         : %.4f' % (running['gate_std'] / running_count))
        print('gate_target_mean : %.4f' % (running['target_mean'] / running_count))
        print('gate_target_std  : %.4f' % (running['target_std'] / running_count))
        print('noise_mean_deg   : %.3f' % (running['noise_deg'] / running_count))

        val_clean = validate(
            model,
            val_loader,
            device,
            args,
            fixed_deg=0.0,
            max_samples=cli.val_max_samples,
        )
        val_rot5 = validate(
            model,
            val_loader,
            device,
            args,
            fixed_deg=5.0,
            max_samples=cli.val_max_samples,
        )
        val_rot8 = validate(
            model,
            val_loader,
            device,
            args,
            fixed_deg=8.0,
            max_samples=cli.val_max_samples,
        )

        score = (
            val_clean['rmse']
            + val_rot5['rmse']
            + val_rot8['rmse']
        )

        print('\n--- validation ---')
        print(
            ('0 deg : RMSE=%.4f AbsRel=%.4f a1=%.4f | '
             'g1=%.4f g2=%.4f g3=%.4f | target=%.4f MAE=%.4f corr=%.4f')
            % (
                val_clean['rmse'],
                val_clean['abs_rel'],
                val_clean['a1'],
                val_clean['gate1_mean'],
                val_clean['gate2_mean'],
                val_clean['gate3_mean'],
                val_clean['gate_target_mean'],
                val_clean['gate_target_mae'],
                val_clean['gate_target_corr'],
            )
        )
        print(
            ('5 deg : RMSE=%.4f AbsRel=%.4f a1=%.4f | '
             'g1=%.4f g2=%.4f g3=%.4f | target=%.4f MAE=%.4f corr=%.4f')
            % (
                val_rot5['rmse'],
                val_rot5['abs_rel'],
                val_rot5['a1'],
                val_rot5['gate1_mean'],
                val_rot5['gate2_mean'],
                val_rot5['gate3_mean'],
                val_rot5['gate_target_mean'],
                val_rot5['gate_target_mae'],
                val_rot5['gate_target_corr'],
            )
        )
        print(
            ('8 deg : RMSE=%.4f AbsRel=%.4f a1=%.4f | '
             'g1=%.4f g2=%.4f g3=%.4f | target=%.4f MAE=%.4f corr=%.4f')
            % (
                val_rot8['rmse'],
                val_rot8['abs_rel'],
                val_rot8['a1'],
                val_rot8['gate1_mean'],
                val_rot8['gate2_mean'],
                val_rot8['gate3_mean'],
                val_rot8['gate_target_mean'],
                val_rot8['gate_target_mae'],
                val_rot8['gate_target_corr'],
            )
        )
        print('selection score  : %.4f' % score)

        diag_fields = [
            'epoch', 'noise_deg', 'samples', 'rmse', 'abs_rel', 'a1', 'nll',
            'gate_mean', 'gate1_mean', 'gate2_mean', 'gate3_mean',
            'gate_target_mean', 'gate_target_std',
            'gate_target_mae', 'gate_target_corr', 'forward_ms', 'score',
        ]

        for val_result in (val_clean, val_rot5, val_rot8):
            base_row = {
                'epoch': epoch + 1,
                **val_result,
                'score': score,
            }
            append_csv(
                val_csv,
                val_fields,
                {key: base_row[key] for key in val_fields},
            )
            append_csv(
                val_diag_csv,
                diag_fields,
                {key: base_row[key] for key in diag_fields},
            )

        save_gate_checkpoint(
            ckpt_dir / 'last_gate.pt',
            model,
            optimizer,
            epoch,
            global_step,
            {
                'clean': val_clean,
                'rot5': val_rot5,
                'rot8': val_rot8,
                'score': score,
            },
            cli,
        )

        if score < best_score:
            best_score = score
            save_gate_checkpoint(
                ckpt_dir / 'best_gate.pt',
                model,
                optimizer,
                epoch,
                global_step,
                {
                    'clean': val_clean,
                    'rot5': val_rot5,
                    'rot8': val_rot8,
                    'score': score,
                },
                cli,
            )
            print('saved new best gate checkpoint')

    print('\n[PASS] Phase-A training run completed.')
    print('best gate checkpoint:', ckpt_dir / 'best_gate.pt')
    print('train log           :', train_csv)
    print('validation log      :', val_csv)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--scenes', default='chess,office')
    parser.add_argument('--val_sequences', default='')
    parser.add_argument('--frame_stride', type=int, default=5)
    parser.add_argument('--window_radius', type=int, default=20)
    parser.add_argument('--num_source_views', type=int, default=4)

    parser.add_argument('--dnet_ckpt', default='./ckpts/DNET_scannet.pt')
    parser.add_argument('--fnet_ckpt', default='./ckpts/FNET_scannet.pt')
    parser.add_argument('--magnet_ckpt', default='./ckpts/MAGNET_scannet.pt')

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--loss_gamma', type=float, default=0.8)
    parser.add_argument('--lambda_gate', type=float, default=0.1)
    parser.add_argument('--gate_tau', type=float, default=0.10)
    parser.add_argument(
        '--gate_min_delta',
        type=float,
        default=0.01,
        help='Ignore pixels where |mu_mv-mu_mono| is below this depth change (m).',
    )

    parser.add_argument('--num_train_iter', type=int, default=3)
    parser.add_argument('--num_test_iter', type=int, default=3)
    parser.add_argument('--min_depth', type=float, default=1e-3)
    parser.add_argument('--max_depth', type=float, default=10.0)

    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=0,
        help='0 = full epoch; use e.g. 50 for the first smoke training.',
    )
    parser.add_argument(
        '--val_max_samples',
        type=int,
        default=32,
        help='0 = full validation set.',
    )
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--resume', default=None)
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Load --resume checkpoint and run 0/5/8-deg Gate diagnostics only.',
    )

    parser.add_argument(
        '--output_dir',
        default='./exp/STRUCTMAGNET/7scenes_chess_office/phaseA_gate',
    )

    cli = parser.parse_args()

    for name, path in (
        ('D-Net', cli.dnet_ckpt),
        ('F-Net', cli.fnet_ckpt),
        ('MaGNet', cli.magnet_ckpt),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{name} checkpoint not found: {path}')

    train(cli)


if __name__ == '__main__':
    main()
