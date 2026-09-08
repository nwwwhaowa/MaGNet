"""Training/validation loader for a small 7-Scenes subset (e.g. chess + office).

Drop this file into: MaGNet/data/dataloader_7scenes_train.py

The returned batch format intentionally matches the official MaGNet
``data/dataloader_7scenes.py`` loader so that ``utils.data_preprocess`` can be
used unchanged.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


def _read_extm_from_txt(path: str) -> np.ndarray:
    """Read the 7-Scenes camera pose and convert it to MaGNet's ExtM format.

    This follows the official MaGNet 7-Scenes loader: read the 4x4 matrix and
    invert it before returning.
    """
    mat = np.eye(4, dtype=np.float64)

    with open(path, "r") as f:
        rows = [line.strip() for line in f.readlines() if line.strip()]

    if len(rows) < 4:
        raise ValueError(f"Pose file has fewer than 4 rows: {path}")

    for r in range(4):
        values = [float(x) for x in rows[r].split()]
        if len(values) != 4:
            raise ValueError(f"Invalid pose row in {path}: {rows[r]}")
        mat[r, :] = np.asarray(values, dtype=np.float64)

    if not np.isfinite(mat).all():
        return np.full((4, 4), np.nan, dtype=np.float64)

    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.full((4, 4), np.nan, dtype=np.float64)


def _parse_scene_list(value) -> List[str]:
    if isinstance(value, str):
        return [x.strip().lower() for x in value.split(",") if x.strip()]
    return [str(x).strip().lower() for x in value]


def _read_split_sequences(scene_dir: Path, split_name: str) -> List[int]:
    split_path = scene_dir / split_name

    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    seqs: List[int] = []

    for raw in split_path.read_text().splitlines():
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        # Official files use strings such as 'sequence1'. Be permissive.
        match = re.search(r"(\d+)\s*$", line)

        if match is None:
            raise ValueError(
                f"Cannot parse sequence id from {split_path}: {raw!r}"
            )

        seqs.append(int(match.group(1)))

    if not seqs:
        raise RuntimeError(f"No sequences found in {split_path}")

    return sorted(set(seqs))


def _parse_val_sequences(spec: str) -> Dict[str, int]:
    """Parse 'chess:4,office:7'. Empty string means auto-select."""
    out: Dict[str, int] = {}

    if not spec:
        return out

    for token in spec.split(","):
        token = token.strip()

        if not token:
            continue

        if ":" not in token:
            raise ValueError(
                f"Invalid --seven_val_sequences item {token!r}; "
                "expected scene:id"
            )

        scene, seq = token.split(":", 1)
        out[scene.strip().lower()] = int(seq)

    return out


def _frame_ids(seq_dir: Path) -> List[int]:
    ids: List[int] = []

    for p in glob.glob(str(seq_dir / "frame-*.color.png")):
        m = re.search(r"frame-(\d+)\.color\.png$", p)

        if m:
            ids.append(int(m.group(1)))

    return sorted(set(ids))


class SevenScenesTrainDataset(Dataset):
    def __init__(self, args, mode: str):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be train/val/test, got {mode}")

        self.args = args
        self.mode = mode

        self.dataset_path = Path(os.path.expanduser(args.dataset_path))
        self.scenes = _parse_scene_list(
            getattr(args, "seven_scenes", "chess,office")
        )
        self.val_spec = _parse_val_sequences(
            getattr(args, "seven_val_sequences", "")
        )
        self.frame_stride = max(
            1,
            int(getattr(args, "seven_frame_stride", 5)),
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.window_radius = int(args.MAGNET_window_radius)
        self.n_views = int(args.MAGNET_num_source_views)

        if self.n_views % 2 != 0:
            raise ValueError(
                "MAGNET_num_source_views must be even for centered windows"
            )

        self.frame_interval = self.window_radius // (self.n_views // 2)

        if self.frame_interval <= 0:
            raise ValueError("MAGNET_window_radius is too small")

        self.img_idx_center = self.n_views // 2

        self.window_idx_list = [
            i * self.frame_interval
            for i in range(
                -self.n_views // 2,
                self.n_views // 2 + 1,
            )
        ]

        self.img_H = int(args.input_height)
        self.img_W = int(args.input_width)
        self.dpv_H = int(args.dpv_height)
        self.dpv_W = int(args.dpv_width)

        self.ray_array = self.get_ray_array()
        self.cam_intrins = self.get_cam_intrinsics()
        self.samples: List[Tuple[str, int, int]] = self._build_samples()

        if not self.samples:
            raise RuntimeError(
                f"No valid 7-Scenes samples found for mode={mode}, "
                f"scenes={self.scenes}. "
                "Check dataset path, split files, and window radius."
            )

    def _scene_sequences_for_mode(self, scene: str) -> List[int]:
        scene_dir = self.dataset_path / scene

        train_seqs = _read_split_sequences(
            scene_dir,
            "TrainSplit.txt",
        )
        test_seqs = _read_split_sequences(
            scene_dir,
            "TestSplit.txt",
        )

        if scene in self.val_spec:
            val_seq = self.val_spec[scene]

            if val_seq not in train_seqs:
                raise ValueError(
                    f"Validation seq {scene}:{val_seq} "
                    f"is not in TrainSplit.txt ({train_seqs})"
                )
        else:
            # Deterministic, leakage-free default:
            # reserve the last train sequence.
            val_seq = train_seqs[-1]

        if self.mode == "train":
            seqs = [s for s in train_seqs if s != val_seq]

            if not seqs:
                raise RuntimeError(
                    f"No training sequences remain for {scene} "
                    f"after reserving seq-{val_seq:02d}"
                )

            return seqs

        if self.mode == "val":
            return [val_seq]

        return test_seqs

    def _build_samples(self) -> List[Tuple[str, int, int]]:
        samples: List[Tuple[str, int, int]] = []

        for scene in self.scenes:
            scene_dir = self.dataset_path / scene

            if not scene_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing scene directory: {scene_dir}"
                )

            for seq_id in self._scene_sequences_for_mode(scene):
                seq_dir = scene_dir / f"seq-{seq_id:02d}"

                if not seq_dir.is_dir():
                    raise FileNotFoundError(
                        f"Missing sequence directory: {seq_dir}"
                    )

                ids = _frame_ids(seq_dir)
                id_set = set(ids)

                valid_centers = [
                    center
                    for center in ids
                    if all(
                        (center + off) in id_set
                        for off in self.window_idx_list
                    )
                ]

                # Temporal subsampling saves compute and reduces
                # near-duplicate frames.
                valid_centers = valid_centers[:: self.frame_stride]

                samples.extend(
                    (scene, seq_id, center)
                    for center in valid_centers
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_ray_array(self) -> np.ndarray:
        ray_array = np.ones(
            (self.dpv_H, self.dpv_W, 3),
            dtype=np.float64,
        )

        x_range = np.arange(self.dpv_W)
        y_range = np.arange(self.dpv_H)

        ray_array[:, :, 0] = (
            np.tile(x_range[None, :], (self.dpv_H, 1)) + 0.5
        )
        ray_array[:, :, 1] = (
            np.tile(y_range[:, None], (1, self.dpv_W)) + 0.5
        )

        return ray_array

    def get_cam_intrinsics(self):
        # Match official MaGNet's 7-Scenes loader.
        intm_raw = np.eye(3, dtype=np.float64)

        intm_raw[0, 0] = 585.0
        intm_raw[1, 1] = 585.0
        intm_raw[0, 2] = 320.0
        intm_raw[1, 2] = 240.0

        raw_W = self.img_W
        raw_H = self.img_H

        intm = np.zeros((3, 3), dtype=np.float64)
        intm[2, 2] = 1.0

        intm[0, 0] = intm_raw[0, 0] * (
            self.dpv_W / raw_W
        )
        intm[1, 1] = intm_raw[1, 1] * (
            self.dpv_H / raw_H
        )
        intm[0, 2] = intm_raw[0, 2] * (
            self.dpv_W / raw_W
        )
        intm[1, 2] = intm_raw[1, 2] * (
            self.dpv_H / raw_H
        )

        pixel_to_ray = np.copy(self.ray_array)

        pixel_to_ray[:, :, 0] = (
            (
                pixel_to_ray[:, :, 0]
                * (raw_W / self.dpv_W)
            )
            - intm_raw[0, 2]
        ) / intm_raw[0, 0]

        pixel_to_ray[:, :, 1] = (
            (
                pixel_to_ray[:, :, 1]
                * (raw_H / self.dpv_H)
            )
            - intm_raw[1, 2]
        ) / intm_raw[1, 1]

        pixel_to_ray_2d = np.reshape(
            np.transpose(
                pixel_to_ray,
                axes=[2, 0, 1],
            ),
            [3, -1],
        )

        return {
            "unit_ray_array_2D": torch.from_numpy(
                pixel_to_ray_2d.astype(np.float32)
            ),
            "intM": torch.from_numpy(
                intm.astype(np.float32)
            ),
        }

    def __getitem__(self, idx: int):
        scene, seq_id, center = self.samples[idx]

        seq_dir = (
            self.dataset_path
            / scene
            / f"seq-{seq_id:02d}"
        )

        frame_ids = [
            center + off
            for off in self.window_idx_list
        ]

        data_array = []

        for i, cur_idx in enumerate(frame_ids):
            img_path = (
                seq_dir
                / f"frame-{cur_idx:06d}.color.png"
            )
            depth_path = (
                seq_dir
                / f"frame-{cur_idx:06d}.depth.png"
            )
            pose_path = (
                seq_dir
                / f"frame-{cur_idx:06d}.pose.txt"
            )

            # RGB: H x W x 3 -> 3 x H x W
            img_pil = Image.open(img_path).convert("RGB")
            img_pil = img_pil.resize(
                size=(self.img_W, self.img_H),
                resample=Image.BILINEAR,
            )

            img = np.asarray(
                img_pil,
                dtype=np.float32,
            ) / 255.0

            img = torch.from_numpy(
                img.copy()
            ).permute(2, 0, 1)

            img = self.normalize(img)

            if i == self.img_idx_center:
                # Resize the depth once, then make a writable NumPy copy.
                depth_pil = Image.open(depth_path)
                depth_pil = depth_pil.resize(
                    size=(self.img_W, self.img_H),
                    resample=Image.NEAREST,
                )

                depth = np.asarray(depth_pil).copy()

                # 7-Scenes invalid depth marker.
                depth[depth == 65535] = 0

                # millimeters -> meters
                depth = depth.astype(np.float32) / 1000.0

                # H x W -> 1 x H x W
                gt_dmap = torch.from_numpy(
                    depth
                ).unsqueeze(0)
            else:
                gt_dmap = 0.0

            extM = _read_extm_from_txt(
                str(pose_path)
            )

            data_array.append(
                {
                    "img": img,
                    "gt_dmap": gt_dmap,
                    "extM": extM,
                    "scene_name": (
                        f"{scene}_seq-{seq_id:02d}"
                    ),
                    "img_idx": cur_idx,
                }
            )

        return data_array, self.cam_intrins


def balanced_validation_indices(samples, limit):
    """Deterministic scene-balanced subset, spread over each held-out sequence.

    Taking the first 32 entries of the scene-sorted list evaluates only Chess.
    Round-robin allocation covers every scene; evenly spaced indices reduce
    the near-duplicate temporal prefix in the previous smoke validation.
    """
    if limit <= 0 or limit >= len(samples):
        return list(range(len(samples)))
    groups = {}
    for i, (scene, _, _) in enumerate(samples):
        groups.setdefault(scene, []).append(i)
    if limit < len(groups):
        raise ValueError('val_max_samples must cover at least one frame per scene')
    counts = dict.fromkeys(groups, 0)
    for _ in range(limit):
        candidates = [key for key in groups if counts[key] < len(groups[key])]
        key = min(candidates, key=lambda key: counts[key])
        counts[key] += 1
    selected = {}
    for key, indices in groups.items():
        positions = np.linspace(0, len(indices)-1, counts[key], dtype=int)
        selected[key] = [indices[j] for j in positions]
    return [selected[key][j] for j in range(max(counts.values()))
            for key in groups if j < counts[key]]


class SevenScenesTrainLoader:
    def __init__(self, args, mode: str):
        dataset = SevenScenesTrainDataset(
            args,
            mode,
        )

        batch_size = int(
            getattr(args, "batch_size", 1)
        )

        workers = int(
            getattr(
                args,
                "num_workers",
                getattr(args, "workers", 1),
            )
        )

        self.t_samples = dataset

        limit = int(getattr(args, 'val_max_samples', 0))
        if mode == 'val' and limit > 0:
            dataset = Subset(dataset, balanced_validation_indices(dataset.samples, limit))

        self.data = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=workers,
            pin_memory=True,
            drop_last=(
                mode == "train"
                and batch_size > 1
            ),
        )
