#!/usr/bin/env python3
"""Sanity-check Chess+Office 7-Scenes training windows before training.

Drop this file into: MaGNet/tools/check_7scenes_train.py
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_THIS = Path(__file__).resolve()
_REPO = None
for p in (_THIS.parent, *_THIS.parents):
    if (p / "data").is_dir() and (p / "utils").is_dir() and (p / "models").is_dir():
        _REPO = p
        break
if _REPO is None:
    raise RuntimeError("Cannot locate MaGNet repo root")
sys.path.insert(0, str(_REPO))

import utils.utils as utils
from data.dataloader_7scenes_train import SevenScenesTrainLoader


def build_args(cli):
    return SimpleNamespace(
        dataset_path=os.path.expanduser(cli.dataset_root),
        seven_scenes=cli.scenes,
        seven_val_sequences=cli.val_sequences,
        seven_frame_stride=cli.frame_stride,
        MAGNET_window_radius=cli.window_radius,
        MAGNET_num_source_views=cli.num_source_views,
        input_height=480,
        input_width=640,
        dpv_height=120,
        dpv_width=160,
        batch_size=1,
        num_workers=cli.num_workers,
    )


def save_preview(out_dir, ref_dat):
    os.makedirs(out_dir, exist_ok=True)
    img = ref_dat["img"].detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    img = utils.unnormalize(img)
    depth = ref_dat["gt_dmap"].detach().cpu().numpy()[0, 0]
    valid = np.logical_and(depth > 1e-3, depth < 10.0)
    np.save(os.path.join(out_dir, "gt_depth.npy"), depth)
    plt.imsave(os.path.join(out_dir, "rgb.png"), img)
    plt.imsave(os.path.join(out_dir, "gt_depth.png"), depth, vmin=0.0, vmax=5.0)
    plt.imsave(os.path.join(out_dir, "valid_mask.png"), valid.astype(np.float32), vmin=0, vmax=1)


def inspect_mode(args, mode, out_dir):
    loader = SevenScenesTrainLoader(args, mode).data
    ds = loader.dataset
    print(f"\n=== {mode.upper()} ===")
    print("samples:", len(ds))
    print("first sample:", ds.samples[0])
    print("last sample :", ds.samples[-1])

    data_array, cam_intrins = next(iter(loader))
    cur_bs = data_array[0]["img"].shape[0]
    ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(data_array, cur_bs)

    print("ref img       :", tuple(ref_dat["img"].shape))
    print("ref gt        :", tuple(ref_dat["gt_dmap"].shape))
    print("neighbor views:", len(nghbr_dats))
    print("relative poses:", tuple(nghbr_poses.shape))
    print("is_valid      :", is_valid.tolist())
    print("intM          :", tuple(cam_intrins["intM"].shape))
    print("rays          :", tuple(cam_intrins["unit_ray_array_2D"].shape))

    gt = ref_dat["gt_dmap"].float()
    valid_depth = torch.logical_and(gt > 1e-3, gt < 10.0)
    print("valid depth %% : %.2f" % (100.0 * valid_depth.float().mean().item()))
    
    for v in range(nghbr_poses.shape[1]):
        R = nghbr_poses[0, v, :3, :3].double().numpy()
        det = np.linalg.det(R)
        ortho = np.linalg.norm(R.T @ R - np.eye(3))
        tnorm = torch.linalg.norm(nghbr_poses[0, v, :3, 3]).item()
        print(f"view {v}: det(R)={det:.6f}, ||R^TR-I||={ortho:.3e}, ||t||={tnorm:.4f} m")

    if mode == "train":
        save_preview(out_dir, ref_dat)
        print("preview saved:", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--scenes", default="chess,office")
    p.add_argument("--val_sequences", default="")
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--window_radius", type=int, default=20)
    p.add_argument("--num_source_views", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--out_dir", default="./exp/STRUCTMAGNET/7scenes_chess_office/dataset_check")
    cli = p.parse_args()
    args = build_args(cli)
    for mode in ["train", "val", "test"]:
        inspect_mode(args, mode, cli.out_dir)
    print("\n[PASS] 7-Scenes training dataset sanity check completed.")
