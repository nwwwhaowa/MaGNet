#!/usr/bin/env python3
"""Phase-A GeometryGate training for StructMaGNet on 7-Scenes Chess + Office.

Stage goal:
    - Keep D-Net, F-Net, G-Net and learned upsampling frozen.
    - Train GeometryGate only.
    - Inject rotation-only perturbations online.
    - Preserve translation exactly.
    - Supervise the gate by multi-view utility:

        g_star = clamp((gt - mu_mono) / (mu_mv - mu_mono), 0, 1)

      on identifiable pixels, using only the first frozen ungated proposal.
      Optimize the selected first-step gate loss (default Smooth-L1);
      depth NLL is diagnostic only.
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
from utils.gate_oracle import first_gate_oracle, oracle_loss, OracleAccumulator


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
        and not key.startswith('geometry_gate.')
        and current[key].shape == value.shape
    }

    required = [key for key in current if key.startswith(('g_net.', 'mask_head.'))]
    missing = [key for key in required if key not in compatible]
    if missing:
        raise RuntimeError('Incomplete MaGNet backbone checkpoint: missing or '
                           'shape-mismatched G-Net/upsampling tensors: ' + ', '.join(missing))
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
    oracle = first_gate_oracle(aux, gt_depth, args.min_depth, args.max_depth,
                              args.gate_min_delta)
    stats = OracleAccumulator()
    stats.update(oracle)
    return stats.compute()


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


def compute_gate_utility_loss(aux, gt_depth, args, tau=None):
    """Compatibility wrapper; tau is unused by the interpolation oracle."""
    return oracle_loss(first_gate_oracle(
        aux, gt_depth, args.min_depth, args.max_depth, args.gate_min_delta))


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

    if not torch.isfinite(pred_mu[valid]).all() or not torch.isfinite(pred_sigma[valid]).all():
        raise RuntimeError('Non-finite depth prediction on valid GT during validation')
    if (pred_sigma[valid] <= 0).any():
        raise RuntimeError('Non-positive Gaussian sigma during validation')
    p = pred_mu[valid].float().clamp(min=min_depth, max=max_depth)
    g = gt[valid].float()
    sigma = pred_sigma[valid].float().clamp_min(1e-6)

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
    if exists and path.stat().st_size:
        with path.open(newline='') as f:
            if next(csv.reader(f)) != list(fieldnames):
                raise ValueError(f'CSV schema mismatch: {path}. Use a new output directory.')
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


VALIDATION_PROTOCOL = 'scene-balanced-pixel-oracle-v1'


def resume_best_score(ckpt, args):
    """Scores from the former Chess-prefix validation are not comparable."""
    if ckpt.get('validation_protocol') != VALIDATION_PROTOCOL:
        return None
    previous = ckpt.get('args', {})
    fields = ('dataset_root', 'scenes', 'val_sequences', 'frame_stride',
              'window_radius', 'num_source_views', 'num_test_iter',
              'val_max_samples', 'min_depth', 'max_depth', 'gate_min_delta')
    if any(previous.get(key) != getattr(args, key, None) for key in fields):
        return None
    return ckpt.get('best_score')


def save_gate_checkpoint(path, model, optimizer, epoch, step, metrics, args,
                         best_score=None, scaler=None):
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
            'best_score': best_score,
            'scaler': scaler.state_dict() if scaler is not None else None,
            'stage': 'stage1a-first-gate-oracle',
            'validation_protocol': VALIDATION_PROTOCOL,
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


ORACLE_DIAG_FIELDS = [
    'oracle_valid_pixels', 'oracle_valid_fraction', 'oracle_empty_samples',
    'mono_rmse_low', 'mv_rmse_low', 'fused_rmse_low', 'oracle_rmse_low',
    'target_closed_fraction', 'target_open_fraction',
]


def validate(model, loader, device, args, fixed_deg, max_samples, oracle_callback=None):
    model.eval()
    totals = dict.fromkeys(('rmse', 'abs_rel', 'a1', 'nll', 'gate_mean',
                            'gate1_mean', 'gate2_mean', 'gate3_mean', 'forward_ms'), 0.0)
    stats = OracleAccumulator()
    count = 0
    with torch.no_grad():
        for data_array, cam_intrins in loader:
            if max_samples > 0 and count >= max_samples:
                break
            batch_size = data_array[0]['img'].shape[0]
            ref_dat, neighbors, poses, is_valid = utils.data_preprocess(data_array, batch_size)
            ref = ref_dat['img'].to(device, non_blocking=True)
            gt = ref_dat['gt_dmap'].to(device, non_blocking=True)
            src = torch.cat([d['img'].to(device, non_blocking=True) for d in neighbors], dim=0)
            poses = poses.to(device, non_blocking=True)
            noisy, rot_unc, _ = fixed_validation_rotation_noise(poses, fixed_deg)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            preds, aux = model(ref, src, noisy, is_valid, cam_intrins,
                               mode='test', rot_unc=rot_unc, return_aux=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            per_sample_ms = (time.perf_counter() - start) * 1000.0 / batch_size
            # Metrics remain mean per-image metrics, independent of loader batch size.
            for i in range(batch_size):
                if max_samples > 0 and count >= max_samples:
                    break
                metrics = depth_metrics(preds[-1][i:i+1], gt[i:i+1],
                                        args.min_depth, args.max_depth)
                if metrics is None:
                    continue
                sample_aux = {}
                for key in ('geometry_gate', 'ungated_gmm', 'geometry_valid', 'gate_input'):
                    if key in aux:
                        sample_aux[key] = [v[i:i+1] for v in aux[key]]
                sample_aux['mono_gmm'] = aux['mono_gmm'][i:i+1]
                sample_oracle = first_gate_oracle(sample_aux, gt[i:i+1], args.min_depth,
                                                  args.max_depth, args.gate_min_delta)
                stats.update(sample_oracle)
                if oracle_callback is not None:
                    oracle_callback(sample_oracle, sample_aux, count)
                means = _gate_iteration_means(sample_aux)
                for key, value in metrics.items():
                    totals[key] += value
                totals['gate_mean'] += float(sample_aux['geometry_gate'][-1].mean().item())
                for j in range(3):
                    totals[f'gate{j+1}_mean'] += means[j]
                totals['forward_ms'] += per_sample_ms
                count += 1
    if not count:
        raise RuntimeError('Validation produced zero valid samples')
    result = {key: value/count for key, value in totals.items()}
    oracle = stats.compute()
    for key in ('mean', 'std', 'mae', 'corr'):
        source = 'target_'+key if key in ('mean', 'std') else key
        result['gate_target_'+key] = oracle[source]
    result.update(oracle_valid_pixels=oracle['valid_pixels'],
                  oracle_valid_fraction=oracle['valid_fraction'],
                  oracle_empty_samples=oracle['empty_samples'])
    for key in ORACLE_DIAG_FIELDS[3:]:
        result[key] = oracle[key]
    result.update(samples=count, noise_deg=float(fixed_deg))
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

    positive = ('lr', 'lambda_gate', 'gate_min_delta', 'batch_size', 'epochs',
                'num_train_iter', 'num_test_iter', 'grad_clip')
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not 0 < args.min_depth < args.max_depth or not math.isfinite(args.max_depth):
        raise ValueError('Expected finite 0 < min_depth < max_depth')
    if args.num_source_views <= 0 or args.num_source_views % 2:
        raise ValueError('num_source_views must be a positive even number')
    if args.val_max_samples < 0 or args.max_train_steps < 0:
        raise ValueError('sample and step limits must be non-negative')
    if not math.isfinite(getattr(args, 'gate_init_bias', 4.0)):
        raise ValueError('gate_init_bias must be finite')


def _print_phase_a_args(args):
    print(f"learning rate     : {args.lr}")
    print(f"lambda gate       : {args.lambda_gate}")
    print(f"gate loss         : {getattr(args, 'gate_loss', 'smooth_l1')}")
    print(f"fresh gate bias   : {getattr(args, 'gate_init_bias', 4.0)} (overridden by resume weights)")
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
    print('rotation input   : oracle injected angle (radians), not estimated covariance')
    print('device           :', torch.cuda.get_device_name(0))

    train_loader = None if cli.eval_only else SevenScenesTrainLoader(args, 'train').data
    val_args = SimpleNamespace(**vars(args))
    val_args.batch_size = 1
    val_loader = SevenScenesTrainLoader(val_args, 'val').data

    print('train samples    :', len(train_loader.dataset) if train_loader is not None else 'not loaded (eval-only)')
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
        if not cli.eval_only:
            old_args = ckpt.get('args', {})
            for name, default in (('gate_init_bias', 4.0), ('gate_loss', 'smooth_l1')):
                if old_args.get(name, default) != getattr(cli, name, default):
                    raise ValueError(f'{name} differs from checkpoint; start a fresh ablation without --resume')
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        global_step = int(ckpt.get('step', 0))
        resumed_best_score = resume_best_score(ckpt, cli)
        if resumed_best_score is None:
            print('validation protocol/settings changed or historical best unavailable; '
                  'resetting selection score while restoring gate and optimizer')
        if ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        print('resumed gate checkpoint:', cli.resume)
        print('resume epoch      :', start_epoch)

    train_csv = log_dir / 'train.csv'
    val_csv = log_dir / 'val.csv'
    val_diag_csv = log_dir / 'val_diagnostics.csv'

    train_fields = [
        'epoch', 'step', 'loss_total', 'loss_depth', 'loss_gate',
        'gate_mean', 'gate_std', 'gate_target_mean', 'gate_target_std',
        'noise_mean_deg', 'lr', 'oracle_valid_pixels', 'oracle_valid_fraction',
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

    if getattr(cli, 'gate_audit', False):
        if not cli.eval_only or not cli.resume:
            raise ValueError('--gate_audit requires --eval_only and --resume')
        from utils.gate_audit import run_gate_audit
        run_gate_audit(model, val_loader, device, args, cli, validate,
                       checkpoint_args=ckpt.get('args', {}))
        return

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
        ] + ORACLE_DIAG_FIELDS
        val_diag_csv = log_dir / 'eval_diagnostics.csv'
        # This is separate from epoch diagnostics.
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
        skipped_oracle_batches = 0

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

            oracle = first_gate_oracle(aux, gt, args.min_depth, args.max_depth,
                                      args.gate_min_delta)
            oracle_pixels = int(oracle['valid'].sum().item())
            if not oracle_pixels:
                skipped_oracle_batches += 1
                continue  # Do not step AdamW on zero signal (including weight decay).
            loss_gate, target_mean, target_std = oracle_loss(
                oracle, getattr(cli, 'gate_loss', 'smooth_l1'))
            with torch.no_grad():
                loss_depth = stable_magnet_loss(
                    pred_list, gt, args.min_depth, args.max_depth, args.loss_gamma)
            # Only gate[0] contributes gradients in Stage 1A.
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
                'oracle_valid_pixels': oracle_pixels,
                'oracle_valid_fraction': oracle_pixels / oracle['valid'].numel(),
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
            raise RuntimeError('No optimizer steps: all batches lacked identifiable oracle pixels; '
                               'check valid views, depth masks, and gate_min_delta')

        print('\n--- epoch train summary ---')
        print('skipped empty oracle batches:', skipped_oracle_batches)
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
        ] + ORACLE_DIAG_FIELDS

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

        improved = score < best_score
        best_score = min(best_score, score)
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
            cli, best_score=best_score, scaler=scaler,
        )

        if improved:
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
                cli, best_score=best_score, scaler=scaler,
            )
            print('saved new best gate checkpoint')

    print('\n[PASS] Phase-A training run completed.')
    best_path = ckpt_dir / 'best_gate.pt'
    print('best gate checkpoint:', best_path if best_path.exists() else cli.resume)
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
    parser.add_argument('--gate_init_bias', type=float, default=4.0,
                        help='Fresh-run final gate bias: 4 preserves legacy; 0 starts at 0.5.')
    parser.add_argument('--gate_loss', choices=['smooth_l1', 'mse', 'depth_mse'],
                        default='smooth_l1')
    parser.add_argument('--gate_audit', action='store_true',
                        help='Evaluate first-step fixed/spatial/input controls; requires eval_only.')
    parser.add_argument('--gate_audit_full', action='store_true',
                        help='Also rerun all refinement iterations for each fixed/spatial policy.')
    parser.add_argument('--gate_tau', type=float, default=0.10,
                        help='Deprecated compatibility option; unused by the interpolation oracle.')
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
    if cli.gate_audit_full and not cli.gate_audit:
        parser.error('--gate_audit_full requires --gate_audit')
    if cli.gate_audit and (not cli.eval_only or not cli.resume):
        parser.error('--gate_audit requires --eval_only and --resume')

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
