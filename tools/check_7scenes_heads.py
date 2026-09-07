#!/usr/bin/env python3
"""Sanity-check a heads-only 7-Scenes installation for MaGNet.

Run from the MaGNet repository root, for example:
  python tools/check_7scenes_heads.py \
      --dataset_root ~/26workspace/datasets/SevenScenes/heads
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Make the script runnable from tools/, tools/tools/, or another working directory.
# We search upward for the MaGNet repository root instead of relying on PYTHONPATH.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = None
for _parent in _THIS_FILE.parents:
    if (_parent / "utils").is_dir() and (_parent / "data").is_dir():
        _REPO_ROOT = _parent
        break
if _REPO_ROOT is None:
    raise RuntimeError(
        f"Cannot locate MaGNet repository root from {_THIS_FILE}. "
        "Expected a parent directory containing both utils/ and data/."
    )
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

import utils.utils as utils
from data.dataloader_7scenes_heads import SevenScenesHeadsLoadPreprocess


def build_loader_args(dataset_root: str):
    return SimpleNamespace(
        dataset_path=os.path.expanduser(dataset_root),
        MAGNET_window_radius=20,
        MAGNET_num_source_views=4,
        input_height=480,
        input_width=640,
        dpv_height=120,
        dpv_width=160,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    loader_args = build_loader_args(args.dataset_root)
    dataset = SevenScenesHeadsLoadPreprocess(loader_args, mode="test")

    print("[OK] dataset root:", loader_args.dataset_path)
    print("[OK] heads evaluation samples:", len(dataset))
    print("[OK] first split entry:", dataset.filenames[0].strip())
    print("[OK] last  split entry:", dataset.filenames[-1].strip())

    sample_idx = min(max(args.sample, 0), len(dataset) - 1)
    data_array, cam_intrins = dataset[sample_idx]

    print("\n=== frame window ===")
    for i, dat in enumerate(data_array):
        gt = dat["gt_dmap"]
        gt_shape = tuple(gt.shape) if torch.is_tensor(gt) else None
        print(
            f"view[{i}] scene={dat['scene_name']} frame={dat['img_idx']:06d} "
            f"img={tuple(dat['img'].shape)} gt={gt_shape}"
        )

    # Reproduce the official batch dimension expected by utils.data_preprocess.
    batched = []
    for dat in data_array:
        item = dict(dat)
        item["img"] = item["img"].unsqueeze(0)
        if torch.is_tensor(item["gt_dmap"]):
            item["gt_dmap"] = item["gt_dmap"].unsqueeze(0)
        item["extM"] = torch.as_tensor(item["extM"]).unsqueeze(0)
        item["img_idx"] = torch.tensor([item["img_idx"]])
        batched.append(item)

    ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(
        batched, cur_batch_size=1
    )

    print("\n=== reference ===")
    print("ref frame:", int(ref_dat["img_idx"][0]))
    gt = ref_dat["gt_dmap"][0, 0].numpy()
    valid = np.logical_and(gt > 1e-3, gt < 10.0)
    print("depth valid ratio: %.4f" % valid.mean())
    if valid.any():
        print(
            "depth [m] min/median/max: %.3f / %.3f / %.3f"
            % (gt[valid].min(), np.median(gt[valid]), gt[valid].max())
        )

    print("\n=== intrinsics at DPV resolution ===")
    print("K =\n", cam_intrins["intM"].numpy())
    print("unit_ray_array_2D:", tuple(cam_intrins["unit_ray_array_2D"].shape))

    print("\n=== relative poses (neighbor <- reference) ===")
    for j in range(nghbr_poses.shape[1]):
        T = nghbr_poses[0, j].numpy()
        R = T[:3, :3]
        t = T[:3, 3]
        print(
            f"src[{j}] valid={int(is_valid[0, j])} "
            f"det(R)={np.linalg.det(R):.6f} "
            f"||t||={np.linalg.norm(t):.4f} m"
        )
        if not np.isfinite(T).all():
            raise RuntimeError(f"Non-finite relative pose at source {j}")
        if abs(np.linalg.det(R) - 1.0) > 5e-3:
            raise RuntimeError(f"Invalid rotation determinant at source {j}")

    print("\n[PASS] 7Scenes heads data/pose/intrinsics sanity check passed.")


if __name__ == "__main__":
    main()
