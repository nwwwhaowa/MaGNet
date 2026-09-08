#!/usr/bin/env python3
"""Build strict frame-level ScanNet splits for MaGNet/StructMaGNet.

The official ScanNet benchmark split is scene-level. MaGNet needs a reference
frame plus symmetric temporal neighbours. This tool converts scene lists into
'<scene> <reference_frame>' lists and only keeps references whose complete
RGB/depth/pose window exists.

Typical compact-Orin setup:
    storage stride: 5
    window radius : 20
    source views  : 4
which yields offsets [-20, -10, 0, +10, +20].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_scene_list(path):
    scenes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                scenes.append(line.split()[0])
    if not scenes:
        raise RuntimeError(f"No scenes found in {path}")
    return scenes


def numeric_stems(directory, suffix):
    ids = set()
    if not directory.is_dir():
        return ids
    for path in directory.glob(f"*{suffix}"):
        try:
            ids.add(int(path.stem))
        except ValueError:
            continue
    return ids


def pose_is_finite(path):
    try:
        pose = np.loadtxt(path, dtype=np.float64)
    except Exception:
        return False
    return pose.shape == (4, 4) and np.isfinite(pose).all()


def valid_references(
    scans_root,
    scenes,
    window_radius,
    num_source_views,
    reference_stride,
    check_pose_finite,
):
    if num_source_views <= 0 or num_source_views % 2 != 0:
        raise ValueError("--num_source_views must be a positive even number")
    half = num_source_views // 2
    if window_radius % half != 0:
        raise ValueError(
            "--window_radius must be divisible by num_source_views/2"
        )
    frame_interval = window_radius // half
    offsets = [i * frame_interval for i in range(-half, half + 1)]

    rows = []
    scene_stats = []

    for scene in scenes:
        scene_dir = scans_root / scene
        color_ids = numeric_stems(scene_dir / "color", ".jpg")
        depth_ids = numeric_stems(scene_dir / "depth", ".png")
        pose_ids = numeric_stems(scene_dir / "pose", ".txt")
        common = color_ids & depth_ids & pose_ids

        if not common:
            scene_stats.append((scene, 0, 0))
            continue

        candidates = sorted(common)
        if reference_stride > 1:
            candidates = [idx for idx in candidates if idx % reference_stride == 0]

        kept = 0
        for ref in candidates:
            required = [ref + off for off in offsets]
            if not all(idx in common for idx in required):
                continue
            if check_pose_finite:
                if not all(
                    pose_is_finite(scene_dir / "pose" / f"{idx}.txt")
                    for idx in required
                ):
                    continue
            rows.append((scene, ref))
            kept += 1

        scene_stats.append((scene, len(common), kept))

    return rows, scene_stats, offsets


def write_rows(path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        if header:
            f.write(f"# {header}\n")
        for scene, frame in rows:
            f.write(f"{scene} {frame}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--scans_dir", default="scans")
    parser.add_argument("--train_scenes", required=True)
    parser.add_argument("--val_scenes", required=True)
    parser.add_argument(
        "--train_out",
        default="./data_split/scannet_train_stride5.txt",
    )
    parser.add_argument(
        "--val_out",
        default="./data_split/scannet_val_stride5.txt",
    )
    parser.add_argument("--window_radius", type=int, default=20)
    parser.add_argument("--num_source_views", type=int, default=4)
    parser.add_argument(
        "--reference_stride",
        type=int,
        default=5,
        help="Only allow reference frame IDs divisible by this value.",
    )
    parser.add_argument(
        "--skip_pose_finite_check",
        action="store_true",
        help="Faster, but invalid ScanNet poses may enter the split.",
    )
    args = parser.parse_args()

    scans_root = (
        Path(args.dataset_root).expanduser().resolve() / args.scans_dir
    )
    if not scans_root.is_dir():
        raise FileNotFoundError(scans_root)

    train_scenes = read_scene_list(args.train_scenes)
    val_scenes = read_scene_list(args.val_scenes)

    train_rows, train_stats, offsets = valid_references(
        scans_root,
        train_scenes,
        args.window_radius,
        args.num_source_views,
        args.reference_stride,
        not args.skip_pose_finite_check,
    )
    val_rows, val_stats, _ = valid_references(
        scans_root,
        val_scenes,
        args.window_radius,
        args.num_source_views,
        args.reference_stride,
        not args.skip_pose_finite_check,
    )

    header = (
        f"window_offsets={offsets}; reference_stride={args.reference_stride}; "
        f"generated from scene-level ScanNet split"
    )
    write_rows(Path(args.train_out), train_rows, header)
    write_rows(Path(args.val_out), val_rows, header)

    print("ScanNet frame split generation")
    print("  scans root      :", scans_root)
    print("  offsets         :", offsets)
    print("  train scenes    :", len(train_scenes))
    print("  train refs      :", len(train_rows))
    print("  val scenes      :", len(val_scenes))
    print("  val refs        :", len(val_rows))
    print("  train out       :", args.train_out)
    print("  val out         :", args.val_out)

    missing_train = sum(1 for _, common, kept in train_stats if common == 0)
    missing_val = sum(1 for _, common, kept in val_stats if common == 0)
    if missing_train or missing_val:
        print(
            "  scenes absent   :",
            f"train={missing_train}, val={missing_val}",
            "(expected when preparing a subset/pilot dataset)",
        )

    if not train_rows:
        raise RuntimeError("Generated training split is empty")
    if not val_rows:
        raise RuntimeError("Generated validation split is empty")


if __name__ == "__main__":
    main()
