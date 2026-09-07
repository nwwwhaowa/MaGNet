#!/usr/bin/env python3
"""Phase-A training for StructMaGNet on Chess + Office from 7-Scenes.

Goal:
  - load the official MaGNet checkpoint into STRUCTMAGNET;
  - freeze D-Net, F-Net, G-Net and upsampling head;
  - train ONLY GeometryGate;
  - inject online rotational pose perturbations during training;
  - validate at fixed 0 deg and 5 deg;
  - save CSV logs, checkpoints, and paper-friendly raw/PNG visualizations.

Drop this file into the MaGNet repository root.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils.utils as utils
from utils.losses import MagnetLoss
from models.STRUCTMAGNET import STRUCTMAGNET
from data.dataloader_7scenes_train import SevenScenesTrainLoader


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_magnet_backbone(path, model):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_state = model.state_dict()
    compatible = {}
    for k, v in state.items():
        k2 = k[7:] if k.startswith("module.") else k
        if k2 in model_state and model_state[k2].shape == v.shape:
            compatible[k2] = v
    msg = model.load_state_dict(compatible, strict=False)
    print("loaded original MaGNet tensors:", len(compatible))
    print("StructMaGNet-only/unloaded tensors:", len(msg.missing_keys))
    for k in msg.missing_keys:
        if k.startswith("geometry_gate"):
            print("  new:", k)
    return model


def set_phase_a_trainability(model):
    for p in model.parameters():
        p.requires_grad = False
    if not hasattr(model, "geometry_gate"):
        raise AttributeError("STRUCTMAGNET has no geometry_gate")
    for p in model.geometry_gate.parameters():
        p.requires_grad = True


def set_phase_a_mode(model, training: bool):
    # Avoid BN/statistics drift in frozen modules when model.train() is called.
    model.eval()
    if training:
        model.geometry_gate.train()
    else:
        model.geometry_gate.eval()


def _skew(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack(
        [o, -z, y, z, o, -x, -y, x, o], dim=-1
    ).reshape(v.shape[:-1] + (3, 3))


def so3_exp(rotvec):
    """Differentiable Rodrigues exponential for (...,3) rotation vectors."""
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / theta.clamp_min(1e-12)
    K = _skew(axis)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    eye = eye.view(*([1] * (rotvec.ndim - 1)), 3, 3).expand(K.shape)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)
    # Exactly zero rotvec should map exactly to identity.
    zero = (theta[..., 0] < 1e-10)[..., None, None]
    return torch.where(zero, eye, R)


def sample_train_angles_deg(batch_size, device, dtype):
    """Sample one severity per sample: 25% clean, 25% mild, 30% medium, 20% severe."""
    u = torch.rand(batch_size, device=device, dtype=dtype)
    angle = torch.zeros_like(u)
    m = (u >= 0.25) & (u < 0.50)
    angle[m] = 0.5 + 1.5 * torch.rand(int(m.sum()), device=device, dtype=dtype)
    m = (u >= 0.50) & (u < 0.80)
    angle[m] = 2.0 + 3.0 * torch.rand(int(m.sum()), device=device, dtype=dtype)
    m = u >= 0.80
    angle[m] = 5.0 + 3.0 * torch.rand(int(m.sum()), device=device, dtype=dtype)
    return angle


def inject_rotation_noise(poses, angle_deg=None):
    """Perturb only R in BxVx4x4 relative poses. Translation stays unchanged.

    If angle_deg is None, training distribution is sampled. If it is a scalar,
    every view gets that fixed magnitude (axis remains random).
    """
    B, V = poses.shape[:2]
    dtype, device = poses.dtype, poses.device
    if angle_deg is None:
        per_sample_deg = sample_train_angles_deg(B, device, dtype)
    else:
        per_sample_deg = torch.full((B,), float(angle_deg), device=device, dtype=dtype)

    per_view_rad = torch.deg2rad(per_sample_deg)[:, None].expand(B, V)
    axis = torch.randn(B, V, 3, device=device, dtype=dtype)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    rotvec = axis * per_view_rad[..., None]
    dR = so3_exp(rotvec)

    noisy = poses.clone()
    R = noisy[:, :, :3, :3]
    noisy[:, :, :3, :3] = R @ dR
    # Translation is intentionally untouched.
    return noisy, per_view_rad, rotvec


def model_forward(model, ref_img, src_imgs, poses, is_valid, cam_intrins, rot_unc, mode):
    """Call both V1 and V2 StructMaGNet signatures safely.

    In Phase A, if the V2 forward supports rot_hyp_vec, zeros are passed so
    pose marginalization is disabled while GeometryGate learns first.
    """
    kwargs = {"mode": mode, "return_aux": True}
    params = inspect.signature(model.forward).parameters
    if "rot_unc" in params:
        kwargs["rot_unc"] = rot_unc
    if "rot_hyp_vec" in params:
        B, V = poses.shape[:2]
        kwargs["rot_hyp_vec"] = torch.zeros(
            B, V, 3, device=poses.device, dtype=poses.dtype
        )
    out = model(ref_img, src_imgs, poses, is_valid, cam_intrins, **kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    # Fallback for an old forward without aux; training can run but gate plots cannot.
    return out, {}


def last_aux(aux, key):
    x = aux.get(key)
    if isinstance(x, (list, tuple)):
        x = x[-1] if x else None
    return x


def prepare_batch(data_array, cam_intrins, device, max_depth):
    B = data_array[0]["img"].shape[0]
    ref_dat, nghbr_dats, poses, is_valid = utils.data_preprocess(data_array, B)
    ref_img = ref_dat["img"].to(device, non_blocking=True)
    gt = ref_dat["gt_dmap"].to(device, non_blocking=True)
    gt = gt.clone()
    gt[gt > max_depth] = 0.0
    src_imgs = torch.cat(
        [d["img"].to(device, non_blocking=True) for d in nghbr_dats], dim=0
    )
    poses = poses.to(device, non_blocking=True)
    return B, ref_dat, ref_img, gt, src_imgs, poses, is_valid, cam_intrins


def append_csv(path, row, fieldnames):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def to_2d(x):
    if x is None:
        return None
    x = x.detach().float().cpu()
    while x.ndim > 2:
        x = x[0]
    return x.numpy()


def save_visual_bundle(out_dir, tag, ref_img, gt, pred, stdev, aux, metadata):
    d = Path(out_dir) / tag
    d.mkdir(parents=True, exist_ok=True)

    rgb = ref_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    rgb = utils.unnormalize(rgb)
    gt_np = gt.detach().cpu().numpy()[0, 0]
    pred_np = pred.detach().cpu().numpy()[0, 0]
    std_np = stdev.detach().cpu().numpy()[0, 0]
    err_np = np.abs(pred_np - gt_np)
    valid = np.logical_and(gt_np > 1e-3, gt_np < 10.0)
    err_np[~valid] = 0.0

    gate = to_2d(last_aux(aux, "geometry_gate"))
    entropy = to_2d(last_aux(aux, "cost_entropy"))
    peak = to_2d(last_aux(aux, "cost_peak"))

    mono = aux.get("mono_gmm")
    mono_np = None
    if torch.is_tensor(mono):
        mono_mu = mono[:, :1]
        mono_mu = F.interpolate(mono_mu, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        mono_np = mono_mu.detach().cpu().numpy()[0, 0]

    arrays = {
        "gt_depth": gt_np,
        "pred_depth": pred_np,
        "abs_error": err_np,
        "stdev": std_np,
    }
    if mono_np is not None:
        arrays["mono_depth"] = mono_np
    if gate is not None:
        arrays["geometry_gate"] = gate
    if entropy is not None:
        arrays["cost_entropy"] = entropy
    if peak is not None:
        arrays["cost_peak"] = peak
    for name, arr in arrays.items():
        np.save(d / f"{name}.npy", arr)

    plt.imsave(d / "rgb.png", rgb)
    plt.imsave(d / "gt_depth.png", gt_np, vmin=0, vmax=5)
    plt.imsave(d / "pred_depth.png", pred_np, vmin=0, vmax=5)
    plt.imsave(d / "abs_error.png", err_np, vmin=0, vmax=1)
    plt.imsave(d / "stdev.png", std_np, vmin=0, vmax=1)
    if mono_np is not None:
        plt.imsave(d / "mono_depth.png", mono_np, vmin=0, vmax=5)
    if gate is not None:
        plt.imsave(d / "geometry_gate.png", gate, vmin=0, vmax=1)
    if entropy is not None:
        plt.imsave(d / "cost_entropy.png", entropy)
    if peak is not None:
        plt.imsave(d / "cost_peak.png", peak, vmin=0, vmax=1)
    with open(d / "meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


def validate(model, loader, args, device, fixed_noise_deg, save_dir=None, epoch=None):
    set_phase_a_mode(model, training=False)
    metrics = utils.RunningAverageDict()
    gate_values = []
    times = []
    first_saved = False

    with torch.inference_mode():
        for data_array, cam_intrins in tqdm(loader, desc=f"Val {fixed_noise_deg:g}deg", leave=False):
            B, ref_dat, ref_img, gt, src, poses, is_valid, cam_intrins = prepare_batch(
                data_array, cam_intrins, device, args.max_depth
            )
            noisy_poses, rot_unc, _ = inject_rotation_noise(poses, fixed_noise_deg)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_list, aux = model_forward(
                model, ref_img, src, noisy_poses, is_valid, cam_intrins, rot_unc, mode="test"
            )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            pred, stdev = torch.split(pred_list[-1], 1, dim=1)
            gt_np = gt.detach().cpu().numpy()[0, 0]
            pred_np = pred.detach().cpu().numpy()[0, 0]
            var_np = np.square(stdev.detach().cpu().numpy()[0, 0])
            valid = np.logical_and(gt_np > args.min_depth, gt_np < args.max_depth)
            pred_np = np.nan_to_num(pred_np, nan=args.min_depth, posinf=args.max_depth, neginf=args.min_depth)
            pred_np = np.clip(pred_np, args.min_depth, args.max_depth)
            if valid.any():
                metrics.update(utils.compute_depth_errors(gt_np[valid], pred_np[valid], var_np[valid]))

            gate = last_aux(aux, "geometry_gate")
            if torch.is_tensor(gate):
                gate_values.append(float(gate.detach().mean().cpu()))

            if save_dir is not None and not first_saved:
                m = metrics.get_value() if metrics._dict is not None else {}
                save_visual_bundle(
                    save_dir,
                    f"epoch_{epoch:03d}_noise_{fixed_noise_deg:g}",
                    ref_img, gt, pred, stdev, aux,
                    {
                        "epoch": epoch,
                        "noise_deg": fixed_noise_deg,
                        "scene_name": str(ref_dat["scene_name"][0]),
                        "ref_frame": int(ref_dat["img_idx"][0]),
                        "rmse_running": float(m.get("rmse", float("nan"))),
                    },
                )
                first_saved = True

    out = metrics.get_value()
    out["gate_mean"] = float(np.mean(gate_values)) if gate_values else float("nan")
    out["forward_ms"] = 1000.0 * float(np.mean(times)) if times else float("nan")
    return out


def train(args):
    seed_everything(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")
    device = torch.device("cuda:0")
    print("device:", torch.cuda.get_device_name(0))

    exp = Path(args.exp_dir)
    ckpt_dir = exp / "checkpoints"
    log_dir = exp / "logs"
    vis_dir = exp / "visuals"
    for d in [exp, ckpt_dir, log_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)
    with open(exp / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader = SevenScenesTrainLoader(args, "train").data
    val_loader = SevenScenesTrainLoader(args, "val").data
    print("train samples:", len(train_loader.dataset))
    print("val samples  :", len(val_loader.dataset))

    model = STRUCTMAGNET(args).to(device)
    model = load_magnet_backbone(args.MAGNET_ckpt, model)
    set_phase_a_trainability(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("trainable parameters:", sum(p.numel() for p in trainable))
    print("phase A: ONLY GeometryGate is trainable")

    loss_fn = MagnetLoss(args)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    train_csv = str(log_dir / "train_log.csv")
    val_csv = str(log_dir / "val_summary.csv")
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        set_phase_a_mode(model, training=True)
        sum_loss = 0.0
        sum_gate = 0.0
        sum_noise = 0.0
        n = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for data_array, cam_intrins in bar:
            B, ref_dat, ref_img, gt, src, poses, is_valid, cam_intrins = prepare_batch(
                data_array, cam_intrins, device, args.max_depth
            )
            gt_mask = torch.logical_and(gt > args.min_depth, gt < args.max_depth)
            if not gt_mask.any():
                continue

            noisy_poses, rot_unc, _ = inject_rotation_noise(poses, angle_deg=None)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_list, aux = model_forward(
                    model, ref_img, src, noisy_poses, is_valid, cam_intrins, rot_unc, mode="train"
                )
                loss = loss_fn(pred_list, gt, gt_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            gate = last_aux(aux, "geometry_gate")
            gate_mean = float(gate.detach().mean().cpu()) if torch.is_tensor(gate) else float("nan")
            noise_mean_deg = float(torch.rad2deg(rot_unc).mean().detach().cpu())
            sum_loss += float(loss.detach().cpu())
            if math.isfinite(gate_mean):
                sum_gate += gate_mean
            sum_noise += noise_mean_deg
            n += 1
            bar.set_postfix(
                loss=f"{sum_loss/n:.4f}", gate=f"{sum_gate/n:.3f}", noise=f"{sum_noise/n:.2f}deg"
            )

            if args.max_train_steps > 0 and n >= args.max_train_steps:
                break

        if n == 0:
            raise RuntimeError("No valid training batches")

        train_row = {
            "epoch": epoch,
            "loss": sum_loss / n,
            "gate_mean": sum_gate / n,
            "noise_deg_mean": sum_noise / n,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_csv(train_csv, train_row, list(train_row.keys()))

        val0 = validate(model, val_loader, args, device, 0.0, vis_dir, epoch)
        val5 = validate(model, val_loader, args, device, 5.0, vis_dir, epoch)
        score = val0["rmse"] + val5["rmse"]
        val_row = {
            "epoch": epoch,
            "rmse_0": val0["rmse"],
            "absrel_0": val0["abs_rel"],
            "a1_0": val0["a1"],
            "nll_0": val0["nll"],
            "gate_0": val0["gate_mean"],
            "ms_0": val0["forward_ms"],
            "rmse_5": val5["rmse"],
            "absrel_5": val5["abs_rel"],
            "a1_5": val5["a1"],
            "nll_5": val5["nll"],
            "gate_5": val5["gate_mean"],
            "ms_5": val5["forward_ms"],
            "score": score,
        }
        append_csv(val_csv, val_row, list(val_row.keys()))

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "score": score,
            "args": vars(args),
        }
        torch.save(state, ckpt_dir / "last.pt")
        torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pt")
        if score < best_score:
            best_score = score
            torch.save(state, ckpt_dir / "best.pt")

        print(
            f"epoch {epoch}: train_loss={train_row['loss']:.4f} | "
            f"val0 RMSE={val0['rmse']:.4f}, gate={val0['gate_mean']:.3f} | "
            f"val5 RMSE={val5['rmse']:.4f}, gate={val5['gate_mean']:.3f} | "
            f"score={score:.4f}"
        )

    print("\n[DONE] Phase-A GeometryGate training completed")
    print("best checkpoint:", ckpt_dir / "best.pt")
    print("logs           :", log_dir)
    print("visuals        :", vis_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # experiment
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--exp_dir", default="./exp/STRUCTMAGNET/7scenes_chess_office/phaseA_gate")
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=1234)

    # subset / split
    p.add_argument("--seven_scenes", default="chess,office")
    p.add_argument("--seven_val_sequences", default="", help="e.g. chess:4,office:7; empty = last TrainSplit sequence")
    p.add_argument("--seven_frame_stride", type=int, default=5)

    # checkpoints
    p.add_argument("--DNET_ckpt", default="./ckpts/DNET_scannet.pt")
    p.add_argument("--FNET_ckpt", default="./ckpts/FNET_scannet.pt")
    p.add_argument("--MAGNET_ckpt", default="./ckpts/MAGNET_scannet.pt")

    # model args expected by StructMaGNet
    p.add_argument("--output_dim", type=int, default=2)
    p.add_argument("--output_type", default="G")
    p.add_argument("--downsample_ratio", type=int, default=4)
    p.add_argument("--DNET_architecture", default="DenseDepth_BN")
    p.add_argument("--DNET_fix_encoder_weights", default="None")
    p.add_argument("--FNET_architecture", default="PSM-Net")
    p.add_argument("--FNET_feature_dim", type=int, default=64)
    p.add_argument("--MAGNET_sampling_range", type=int, default=3)
    p.add_argument("--MAGNET_num_samples", type=int, default=5)
    p.add_argument("--MAGNET_mvs_weighting", default="CW5")
    p.add_argument("--MAGNET_num_train_iter", type=int, default=3)
    p.add_argument("--MAGNET_num_test_iter", type=int, default=3)
    p.add_argument("--MAGNET_window_radius", type=int, default=20)
    p.add_argument("--MAGNET_num_source_views", type=int, default=4)

    # loss
    p.add_argument("--loss_fn", default="gaussian")
    p.add_argument("--loss_gamma", type=float, default=0.8)

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--max_train_steps", type=int, default=0, help="0=all; use 20 for smoke training")

    # dataset geometry
    p.add_argument("--input_height", type=int, default=480)
    p.add_argument("--input_width", type=int, default=640)
    p.add_argument("--dpv_height", type=int, default=120)
    p.add_argument("--dpv_width", type=int, default=160)
    p.add_argument("--min_depth", type=float, default=1e-3)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--do_kb_crop", action="store_true", default=False)
    p.add_argument("--eigen_crop", action="store_true", default=False)
    p.add_argument("--garg_crop", action="store_true", default=False)
    p.add_argument("--data_augmentation_color", action="store_true", default=False)

    # compatibility attributes used by shared code
    p.add_argument("--dataset_name", default="7scenes")
    p.add_argument("--distributed", action="store_true", default=False)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    args.mode = "train"
    args.num_threads = args.num_workers

    for name in ["DNET_ckpt", "FNET_ckpt", "MAGNET_ckpt"]:
        path = getattr(args, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    train(args)
