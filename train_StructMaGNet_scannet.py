#!/usr/bin/env python3
"""Phase-A GeometryGate training for StructMaGNet on ScanNet.

This is the ScanNet counterpart of train_StructMaGNet.py. It intentionally
keeps the validated Phase-A1 optimization unchanged:
  - D-Net/F-Net/G-Net/upsampling stay frozen.
  - GeometryGate is the only trainable module.
  - rotation-only perturbations are generated online.
  - gate supervision uses the same oracle interpolation target.
  - validation is performed at 0/5/8 degree rotation perturbations.

The only substantive difference is the dataset layer: this script uses strict
frame-level ScanNet train/val splits produced by
tools/scannet/build_scannet_frame_splits.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent
sys.path.insert(0, str(_REPO_ROOT))

import utils.utils as utils
import train_StructMaGNet as phase_a
from data.dataloader_scannet_train import ScannetTrainLoader
from models.STRUCTMAGNET import STRUCTMAGNET


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
        scannet_raw_wh=cli.raw_wh_json,
        input_height=cli.input_height,
        input_width=cli.input_width,
        dpv_height=cli.dpv_height,
        dpv_width=cli.dpv_width,
        min_depth=cli.min_depth,
        max_depth=cli.max_depth,
        gate_min_delta=cli.gate_min_delta,
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


def _print_setup(args, cli, train_loader, val_loader):
    print("=== StructMaGNet ScanNet Phase A1: GeometryGate oracle warm-up ===")
    print("dataset root      :", args.dataset_path)
    print("scans dir         :", args.scannet_scans_dir)
    print("train split       :", args.scannet_train_split)
    print("val split         :", args.scannet_val_split)
    print("window radius     :", args.MAGNET_window_radius)
    print("source views      :", args.MAGNET_num_source_views)
    print("batch size        :", args.batch_size)
    print("workers           :", args.num_workers)
    print("pin memory        :", args.pin_memory)
    print("AMP               :", cli.amp)
    print("gate lr           :", cli.lr)
    print("lambda gate       :", cli.lambda_gate)
    print("train samples     :", len(train_loader.dataset))
    print("val samples       :", len(val_loader.dataset))
    print("device            :", torch.cuda.get_device_name(0))


def _validate_all(model, val_loader, device, args, cli):
    results = {}
    for deg in (0.0, 5.0, 8.0):
        result = phase_a.validate(
            model,
            val_loader,
            device,
            args,
            fixed_deg=deg,
            max_samples=cli.val_max_samples,
        )
        results[deg] = result
        print(
            ("%g deg : RMSE=%.4f AbsRel=%.4f a1=%.4f | "
             "g1=%.4f g2=%.4f g3=%.4f | "
             "target=%.4f MAE=%.4f corr=%.4f")
            % (
                deg,
                result["rmse"],
                result["abs_rel"],
                result["a1"],
                result["gate1_mean"],
                result["gate2_mean"],
                result["gate3_mean"],
                result["gate_target_mean"],
                result["gate_target_mae"],
                result["gate_target_corr"],
            )
        )
    return results


def train(cli):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StructMaGNet ScanNet training")

    device = torch.device("cuda:0")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    args = build_model_args(cli)

    out_dir = Path(cli.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(cli), f, indent=2)

    train_loader = ScannetTrainLoader(args, "train").data
    val_loader = ScannetTrainLoader(args, "val").data
    _print_setup(args, cli, train_loader, val_loader)

    model = STRUCTMAGNET(args).to(device)
    model = phase_a.load_compatible_backbone(cli.magnet_ckpt, model)
    phase_a.freeze_phase_a(model)

    optimizer = torch.optim.AdamW(
        model.geometry_gate.parameters(),
        lr=cli.lr,
        weight_decay=cli.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cli.amp)

    start_epoch = 0
    global_step = 0
    resumed_best_score = None
    if cli.resume:
        ckpt = phase_a.load_gate_checkpoint(cli.resume, model, optimizer)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("step", 0))
        metrics = ckpt.get("metrics", {})
        if isinstance(metrics, dict):
            resumed_best_score = metrics.get("score")
        print("resumed gate checkpoint:", cli.resume)
        print("resume epoch          :", start_epoch)

    train_csv = log_dir / "train.csv"
    val_csv = log_dir / "val.csv"
    diag_csv = log_dir / "val_diagnostics.csv"

    train_fields = [
        "epoch", "step", "loss_total", "loss_depth", "loss_gate",
        "gate_mean", "gate_std", "gate_target_mean", "gate_target_std",
        "noise_mean_deg", "lr",
    ]
    val_fields = [
        "epoch", "noise_deg", "samples", "rmse", "abs_rel", "a1", "nll",
        "gate_mean", "gate1_mean", "gate2_mean", "gate3_mean",
        "gate_target_mean", "gate_target_std", "gate_target_mae",
        "gate_target_corr", "forward_ms", "score",
    ]

    best_score = (
        float(resumed_best_score)
        if resumed_best_score is not None
        else float("inf")
    )

    if cli.eval_only:
        if not cli.resume:
            raise ValueError("--eval_only requires --resume CHECKPOINT")
        print("\n=== ScanNet gate diagnostic evaluation only ===")
        results = _validate_all(model, val_loader, device, args, cli)
        score = sum(results[d]["rmse"] for d in (0.0, 5.0, 8.0))
        if diag_csv.exists():
            diag_csv.unlink()
        for _, result in results.items():
            row = {"epoch": start_epoch, **result, "score": score}
            phase_a.append_csv(
                diag_csv,
                val_fields,
                {key: row[key] for key in val_fields},
            )
        print("diagnostic CSV:", diag_csv)
        return

    for epoch in range(start_epoch, cli.epochs):
        phase_a.set_phase_a_train_mode(model)

        running = {
            "loss_total": 0.0,
            "loss_depth": 0.0,
            "loss_gate": 0.0,
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "target_mean": 0.0,
            "target_std": 0.0,
            "noise_deg": 0.0,
        }
        running_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cli.epochs} ScanNet Train",
        )

        for batch_idx, (data_array, cam_intrins) in enumerate(pbar):
            if cli.max_train_steps > 0 and batch_idx >= cli.max_train_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            cur_batch_size = data_array[0]["img"].shape[0]
            ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(
                data_array,
                cur_batch_size,
            )

            ref_img = ref_dat["img"].to(device, non_blocking=True)
            gt = ref_dat["gt_dmap"].to(device, non_blocking=True).clone()
            gt[~torch.isfinite(gt)] = 0.0
            gt[gt > args.max_depth] = 0.0

            nghbr_imgs = torch.cat(
                [d["img"].to(device, non_blocking=True) for d in nghbr_dats],
                dim=0,
            )
            poses = nghbr_poses.to(device, non_blocking=True)

            noisy_poses, rot_unc, _ = phase_a.sample_train_rotation_noise(poses)

            with torch.cuda.amp.autocast(enabled=cli.amp):
                pred_list, aux = model(
                    ref_img,
                    nghbr_imgs,
                    noisy_poses,
                    is_valid,
                    cam_intrins,
                    mode="train",
                    rot_unc=rot_unc,
                    return_aux=True,
                )

                loss_depth = phase_a.stable_magnet_loss(
                    pred_list,
                    gt,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    gamma=args.loss_gamma,
                )
                loss_gate, target_mean, target_std = (
                    phase_a.compute_gate_utility_loss(
                        aux,
                        gt,
                        args,
                        cli.gate_tau,
                    )
                )
                loss_total = cli.lambda_gate * loss_gate

            if not torch.isfinite(loss_total):
                raise RuntimeError(
                    "Non-finite ScanNet Phase-A total loss. "
                    "Run a short smoke test without AMP to diagnose the sample."
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

            gates = aux.get("geometry_gate")
            if not isinstance(gates, (list, tuple)) or not gates:
                raise RuntimeError("aux[geometry_gate] must be a non-empty list")
            gate = gates[0]
            gate_mean = float(gate.detach().mean().item())
            gate_std = float(gate.detach().std(unbiased=False).item())
            noise_mean_deg = float(
                rot_unc.detach().mean().item() * 180.0 / math.pi
            )

            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss_total": float(loss_total.detach().item()),
                "loss_depth": float(loss_depth.detach().item()),
                "loss_gate": float(loss_gate.detach().item()),
                "gate_mean": gate_mean,
                "gate_std": gate_std,
                "gate_target_mean": float(target_mean.detach().item()),
                "gate_target_std": float(target_std.detach().item()),
                "noise_mean_deg": noise_mean_deg,
                "lr": optimizer.param_groups[0]["lr"],
            }
            phase_a.append_csv(train_csv, train_fields, row)

            running["loss_total"] += row["loss_total"]
            running["loss_depth"] += row["loss_depth"]
            running["loss_gate"] += row["loss_gate"]
            running["gate_mean"] += row["gate_mean"]
            running["gate_std"] += row["gate_std"]
            running["target_mean"] += row["gate_target_mean"]
            running["target_std"] += row["gate_target_std"]
            running["noise_deg"] += row["noise_mean_deg"]
            running_count += 1

            pbar.set_postfix(
                loss=f'{row["loss_total"]:.4f}',
                gate=f"{gate_mean:.3f}",
                tgt=f'{row["gate_target_mean"]:.3f}',
                rot=f"{noise_mean_deg:.2f}deg",
            )

        if running_count == 0:
            raise RuntimeError("No ScanNet training steps were executed")

        print("\n--- epoch train summary ---")
        for name in ("loss_total", "loss_depth", "loss_gate"):
            print(f"{name:17s}: {running[name] / running_count:.6f}")
        print("gate_mean        : %.4f" % (running["gate_mean"] / running_count))
        print("gate_std         : %.4f" % (running["gate_std"] / running_count))
        print("gate_target_mean : %.4f" % (running["target_mean"] / running_count))
        print("noise_mean_deg   : %.3f" % (running["noise_deg"] / running_count))

        print("\n--- validation ---")
        results = _validate_all(model, val_loader, device, args, cli)
        score = sum(results[d]["rmse"] for d in (0.0, 5.0, 8.0))
        print("selection score  : %.4f" % score)

        for _, result in results.items():
            row = {"epoch": epoch + 1, **result, "score": score}
            phase_a.append_csv(
                val_csv,
                val_fields,
                {key: row[key] for key in val_fields},
            )
            phase_a.append_csv(
                diag_csv,
                val_fields,
                {key: row[key] for key in val_fields},
            )

        metrics = {
            "clean": results[0.0],
            "rot5": results[5.0],
            "rot8": results[8.0],
            "score": score,
        }
        phase_a.save_gate_checkpoint(
            ckpt_dir / "last_gate.pt",
            model,
            optimizer,
            epoch,
            global_step,
            metrics,
            cli,
        )

        if score < best_score:
            best_score = score
            phase_a.save_gate_checkpoint(
                ckpt_dir / "best_gate.pt",
                model,
                optimizer,
                epoch,
                global_step,
                metrics,
                cli,
            )
            print("saved new best ScanNet gate checkpoint")

    print("\n[PASS] ScanNet Phase-A training completed.")
    print("best gate checkpoint:", ckpt_dir / "best_gate.pt")
    print("train log           :", train_csv)
    print("validation log      :", val_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Train StructMaGNet GeometryGate on compact ScanNet data"
    )

    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--scans_dir", default="scans")
    parser.add_argument(
        "--train_split",
        default="./data_split/scannet_train_stride5.txt",
    )
    parser.add_argument(
        "--val_split",
        default="./data_split/scannet_val_stride5.txt",
    )
    parser.add_argument(
        "--raw_wh_json",
        default="./data/scannet_raw_WH.json",
    )

    parser.add_argument("--window_radius", type=int, default=20)
    parser.add_argument("--num_source_views", type=int, default=4)
    parser.add_argument("--input_height", type=int, default=480)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--dpv_height", type=int, default=120)
    parser.add_argument("--dpv_width", type=int, default=160)

    parser.add_argument("--dnet_ckpt", default="./ckpts/DNET_scannet.pt")
    parser.add_argument("--fnet_ckpt", default="./ckpts/FNET_scannet.pt")
    parser.add_argument("--magnet_ckpt", default="./ckpts/MAGNET_scannet.pt")

    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Orin-safe default. Increase only after monitoring RAM with tegrastats.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Disabled by default on Jetson unified-memory systems.",
    )
    parser.add_argument("--color_aug", action="store_true")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_gamma", type=float, default=0.8)
    parser.add_argument("--lambda_gate", type=float, default=0.1)
    parser.add_argument("--gate_tau", type=float, default=0.10)
    parser.add_argument("--gate_min_delta", type=float, default=0.01)

    parser.add_argument("--num_train_iter", type=int, default=3)
    parser.add_argument("--num_test_iter", type=int, default=3)
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=10.0)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="0 = full epoch; use 10/50 first on Orin.",
    )
    parser.add_argument(
        "--val_max_samples",
        type=int,
        default=32,
        help="0 = full validation set.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument(
        "--output_dir",
        default="./exp/STRUCTMAGNET/scannet/phaseA_gate",
    )

    cli = parser.parse_args()
    _check_inputs(cli)
    train(cli)


if __name__ == "__main__":
    main()
