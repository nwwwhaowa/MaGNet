#!/usr/bin/env python3
"""Run the original MaGNet on the 7-Scenes heads subset.

This script intentionally does NOT modify MaGNet geometry or network modules.
It is a smoke-test/baseline runner for the original model using the official
7-Scenes preprocessing and the heads-only evaluation subset.

Example (from MaGNet repo root):
  python test_MaGNet_heads.py \
      --dataset_root ~/26workspace/datasets/SevenScenes/heads \
      --dnet_ckpt ./ckpts/DNET_scannet.pt \
      --fnet_ckpt ./ckpts/FNET_scannet.pt \
      --magnet_ckpt ./ckpts/MAGNET_scannet.pt \
      --max_samples 1
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Locate the MaGNet repository root so this script is not sensitive to cwd.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = None
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    if ((_parent / "utils").is_dir() and (_parent / "data").is_dir()
            and (_parent / "models").is_dir()):
        _REPO_ROOT = _parent
        break
if _REPO_ROOT is None:
    raise RuntimeError(
        f"Cannot locate MaGNet repository root from {_THIS_FILE}. "
        "Expected a directory containing utils/, data/, and models/."
    )
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.utils as utils
from data.dataloader_7scenes_heads import SevenScenesHeadsLoader
from models.MAGNET import MAGNET


def build_model_args(cli):
    # Match the official MaGNet 7-Scenes config unless explicitly overridden.
    return SimpleNamespace(
        # output
        output_dim=2,
        output_type="G",
        downsample_ratio=4,
        # D-Net
        DNET_architecture="DenseDepth_BN",
        DNET_fix_encoder_weights="None",
        DNET_ckpt=cli.dnet_ckpt,
        # F-Net
        FNET_architecture="PSM-Net",
        FNET_feature_dim=64,
        FNET_ckpt=cli.fnet_ckpt,
        # MaGNet
        MAGNET_sampling_range=3,
        MAGNET_num_samples=5,
        MAGNET_mvs_weighting="CW5",
        MAGNET_num_train_iter=3,
        MAGNET_num_test_iter=cli.num_test_iter,
        MAGNET_window_radius=cli.window_radius,
        MAGNET_num_source_views=cli.num_source_views,
        MAGNET_ckpt=cli.magnet_ckpt,
        # dataset
        dataset_name="7scenes",
        dataset_path=os.path.expanduser(cli.dataset_root),
        input_height=480,
        input_width=640,
        dpv_height=120,
        dpv_width=160,
        min_depth=1e-3,
        max_depth=10.0,
        # misc attributes used by shared code
        do_kb_crop=False,
        eigen_crop=False,
        garg_crop=False,
        data_augmentation_color=False,
        num_threads=1,
        mode="online_eval",
        distributed=False,
        num_workers=cli.num_workers,
        pin_memory=True,
    )


def save_visuals(save_dir, idx, ref_img, gt, pred, stdev):
    os.makedirs(save_dir, exist_ok=True)

    img = ref_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    img = utils.unnormalize(img)

    gt_np = gt.detach().cpu().numpy()[0, 0]
    pred_np = pred.detach().cpu().numpy()[0, 0]
    std_np = stdev.detach().cpu().numpy()[0, 0]

    err_np = np.abs(pred_np - gt_np)
    valid = np.logical_and(gt_np > 1e-3, gt_np < 10.0)
    err_np[~valid] = 0.0

    stem = f"{idx:04d}"
    plt.imsave(os.path.join(save_dir, f"{stem}_rgb.png"), img)
    plt.imsave(os.path.join(save_dir, f"{stem}_gt_depth.png"), gt_np, vmin=0.0, vmax=5.0)
    plt.imsave(os.path.join(save_dir, f"{stem}_pred_depth.png"), pred_np, vmin=0.0, vmax=5.0)
    plt.imsave(os.path.join(save_dir, f"{stem}_abs_error.png"), err_np, vmin=0.0, vmax=1.0)
    plt.imsave(os.path.join(save_dir, f"{stem}_stdev.png"), std_np, vmin=0.0, vmax=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--dnet_ckpt", default="./ckpts/DNET_scannet.pt")
    parser.add_argument("--fnet_ckpt", default="./ckpts/FNET_scannet.pt")
    parser.add_argument("--magnet_ckpt", default="./ckpts/MAGNET_scannet.pt")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max_samples", type=int, default=1,
                        help="1 for smoke test; 0 means all heads samples")
    parser.add_argument("--num_source_views", type=int, default=4)
    parser.add_argument("--window_radius", type=int, default=20)
    parser.add_argument("--num_test_iter", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", default="./exp/MAGNET/heads_smoke")
    cli = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. MaGNet smoke test expects a CUDA GPU.")
    device = torch.device("cuda:0")

    args = build_model_args(cli)

    for name, path in [
        ("D-Net", args.DNET_ckpt),
        ("F-Net", args.FNET_ckpt),
        ("MaGNet", args.MAGNET_ckpt),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    print("=== configuration ===")
    print("dataset_root      :", args.dataset_path)
    print("source views      :", args.MAGNET_num_source_views)
    print("window radius     :", args.MAGNET_window_radius)
    print("test iterations   :", args.MAGNET_num_test_iter)
    print("device            :", torch.cuda.get_device_name(0))

    print("\n[1/3] building model...")
    model = MAGNET(args).to(device)
    print("[2/3] loading MaGNet checkpoint:", args.MAGNET_ckpt)
    model = utils.load_checkpoint(args.MAGNET_ckpt, model)
    model.eval()

    print("[3/3] loading heads data...")
    test_loader = SevenScenesHeadsLoader(args, "test").data
    print("heads samples:", len(test_loader.dataset))

    metrics = utils.RunningAverageDict()
    forward_times = []

    limit = len(test_loader.dataset) if cli.max_samples == 0 else min(
        cli.max_samples, len(test_loader.dataset)
    )

    with torch.inference_mode():
        for batch_idx, (data_array, cam_intrins) in enumerate(test_loader):
            if batch_idx >= limit:
                break

            batch_size = data_array[0]["img"].shape[0]
            ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(
                data_array, batch_size
            )

            ref_img = ref_dat["img"].to(device, non_blocking=True)
            gt = ref_dat["gt_dmap"].to(device, non_blocking=True)
            gt[gt > args.max_depth] = 0.0

            src_imgs = torch.cat(
                [d["img"].to(device, non_blocking=True) for d in nghbr_dats],
                dim=0,
            )
            nghbr_poses = nghbr_poses.to(device, non_blocking=True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_list = model(
                ref_img,
                src_imgs,
                nghbr_poses,
                is_valid,
                cam_intrins,
                mode="test",
            )
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - t0)

            pred, stdev = torch.split(pred_list[-1], 1, dim=1)

            gt_np = gt.detach().cpu().numpy()[0, 0]
            pred_np = pred.detach().cpu().numpy()[0, 0]
            var_np = np.square(stdev.detach().cpu().numpy()[0, 0])

            valid = np.logical_and(gt_np > args.min_depth, gt_np < args.max_depth)
            pred_np = np.nan_to_num(
                pred_np,
                nan=args.min_depth,
                posinf=args.max_depth,
                neginf=args.min_depth,
            )
            pred_np = np.clip(pred_np, args.min_depth, args.max_depth)

            if not valid.any():
                print(f"[WARN] sample {batch_idx}: no valid GT pixels; skipped")
                continue

            cur_metrics = utils.compute_depth_errors(
                gt_np[valid], pred_np[valid], var_np[valid]
            )
            metrics.update(cur_metrics)

            print(
                f"sample={batch_idx:03d} "
                f"ref={int(ref_dat['img_idx'][0]):06d} "
                f"RMSE={cur_metrics['rmse']:.4f} "
                f"AbsRel={cur_metrics['abs_rel']:.4f} "
                f"a1={cur_metrics['a1']:.4f} "
                f"forward={forward_times[-1] * 1000:.1f} ms"
            )

            if batch_idx == 0:
                save_visuals(cli.save_dir, batch_idx, ref_img, gt, pred, stdev)

    result = metrics.get_value()
    if result:
        print("\n=== aggregate ===")
        print("samples :", limit)
        print("RMSE    : %.4f" % result["rmse"])
        print("AbsRel  : %.4f" % result["abs_rel"])
        print("a1      : %.4f" % result["a1"])
        print("NLL     : %.4f" % result["nll"])
        print("forward : %.1f ms (mean)" % (1000.0 * np.mean(forward_times)))
        print("visuals :", cli.save_dir)
        print("\n[PASS] Original MaGNet forward completed on 7Scenes/heads.")


if __name__ == "__main__":
    main()
