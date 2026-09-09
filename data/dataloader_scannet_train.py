#!/usr/bin/env python3
"""Strict ScanNet loader for StructMaGNet training/validation.

Split format:
    scene0000_00 123
    scene0000_00 128
    ...

Unlike the original MaGNet ScanNet loader, this loader does not silently
substitute missing neighbour frames. Frame-level split files are expected to
contain only references for which the complete MaGNet temporal window exists.
Use tools/scannet/build_scannet_frame_splits.py to generate such splits.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _read_extm_from_txt(path):
    mat = np.loadtxt(path, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose matrix in {path}, got {mat.shape}")
    if not np.isfinite(mat).all():
        return np.full((4, 4), np.nan, dtype=np.float64)
    try:
        return np.linalg.inv(mat)  # ScanNet cam2world -> world2cam
    except np.linalg.LinAlgError:
        return np.full((4, 4), np.nan, dtype=np.float64)


def _read_intm_from_txt(path):
    mat = np.loadtxt(path, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 intrinsic matrix in {path}, got {mat.shape}")
    return mat


def _read_split(path):
    entries = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected '<scene> <frame_id>', got {line!r}"
                )
            scene, frame = parts
            entries.append((scene, int(frame)))
    if not entries:
        raise RuntimeError(f"Split is empty: {path}")
    return entries


class ScannetTrainLoader:
    def __init__(self, args, mode):
        self.t_samples = ScannetTrainDataset(args, mode)

        if mode == "train":
            if getattr(args, "distributed", False):
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.t_samples
                )
            else:
                self.train_sampler = None

            self.data = DataLoader(
                self.t_samples,
                batch_size=args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_workers,
                pin_memory=getattr(args, "pin_memory", False),
                drop_last=True,
                sampler=self.train_sampler,
                persistent_workers=(args.num_workers > 0 and
                                    getattr(args, 'persistent_workers', True)),
                generator=getattr(args, 'loader_generator', None),
                worker_init_fn=seed_worker,
            )
        elif mode == "val":
            self.train_sampler = None
            self.data = DataLoader(
                self.t_samples,
                batch_size=1,
                shuffle=False,
                num_workers=max(0, min(1, args.num_workers)),
                pin_memory=getattr(args, "pin_memory", False),
                drop_last=False,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class ScannetTrainDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.dataset_path = Path(args.dataset_path).expanduser().resolve()
        self.scans_dir = self.dataset_path / args.scannet_scans_dir

        if mode == "train":
            split_path = Path(args.scannet_train_split).expanduser()
        elif mode == "val":
            split_path = Path(args.scannet_val_split).expanduser()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.samples = _read_split(split_path)

        self.window_radius = int(args.MAGNET_window_radius)
        self.n_views = int(args.MAGNET_num_source_views)
        if self.n_views <= 0 or self.n_views % 2 != 0:
            raise ValueError("MAGNET_num_source_views must be a positive even number")
        if self.window_radius <= 0:
            raise ValueError("MAGNET_window_radius must be positive")

        half = self.n_views // 2
        if self.window_radius % half != 0:
            raise ValueError(
                "MAGNET_window_radius must be divisible by num_source_views/2 "
                "for symmetric ScanNet windows"
            )
        self.frame_interval = self.window_radius // half
        self.window_offsets = [
            i * self.frame_interval for i in range(-half, half + 1)
        ]
        self.ref_index = half

        self.img_H = int(args.input_height)
        self.img_W = int(args.input_width)
        self.dpv_H = int(args.dpv_height)
        self.dpv_W = int(args.dpv_width)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.ray_array = self._get_ray_array()

        raw_wh_path = Path(
            getattr(args, "scannet_raw_wh", "./data/scannet_raw_WH.json")
        )
        with raw_wh_path.open("r") as f:
            self.raw_WH_dict = json.load(f)

    def __len__(self):
        return len(self.samples)

    def _get_ray_array(self):
        ray_array = np.ones((self.dpv_H, self.dpv_W, 3), dtype=np.float32)
        xs = np.arange(self.dpv_W, dtype=np.float32)
        ys = np.arange(self.dpv_H, dtype=np.float32)
        ray_array[:, :, 0] = xs[None, :] + 0.5
        ray_array[:, :, 1] = ys[:, None] + 0.5
        return ray_array

    def _get_cam_intrinsics(self, scene_dir, scene_name):
        intrinsic_path = scene_dir / "intrinsic" / "intrinsic_color.txt"
        if not intrinsic_path.is_file():
            raise FileNotFoundError(f"Missing ScanNet intrinsic: {intrinsic_path}")

        if scene_name not in self.raw_WH_dict:
            raise KeyError(
                f"{scene_name} is missing from data/scannet_raw_WH.json. "
                "Use the original ScanNet image size metadata."
            )

        raw_W, raw_H = self.raw_WH_dict[scene_name]
        intm_raw = _read_intm_from_txt(intrinsic_path)

        intm = np.zeros((3, 3), dtype=np.float32)
        intm[2, 2] = 1.0
        intm[0, 0] = intm_raw[0, 0] * (self.dpv_W / raw_W)
        intm[1, 1] = intm_raw[1, 1] * (self.dpv_H / raw_H)
        intm[0, 2] = intm_raw[0, 2] * (self.dpv_W / raw_W)
        intm[1, 2] = intm_raw[1, 2] * (self.dpv_H / raw_H)

        rays = np.copy(self.ray_array)
        rays[:, :, 0] = (
            rays[:, :, 0] * (raw_W / self.dpv_W) - intm_raw[0, 2]
        ) / intm_raw[0, 0]
        rays[:, :, 1] = (
            rays[:, :, 1] * (raw_H / self.dpv_H) - intm_raw[1, 2]
        ) / intm_raw[1, 1]

        rays_2d = np.reshape(np.transpose(rays, (2, 0, 1)), (3, -1))
        return {
            "unit_ray_array_2D": torch.from_numpy(rays_2d.astype(np.float32)),
            "intM": torch.from_numpy(intm),
        }

    def _validate_window(self, scene_dir, frame_ids):
        missing = []
        for frame_id in frame_ids:
            if not (scene_dir / "color" / f"{frame_id}.jpg").is_file():
                missing.append(f"color/{frame_id}.jpg")
            if not (scene_dir / "pose" / f"{frame_id}.txt").is_file():
                missing.append(f"pose/{frame_id}.txt")
            if not (scene_dir / "depth" / f"{frame_id}.png").is_file():
                missing.append(f"depth/{frame_id}.png")
        if missing:
            raise FileNotFoundError(
                "Frame split references an incomplete MaGNet window in "
                f"{scene_dir}: {missing[:8]}. Rebuild the split with "
                "tools/scannet/build_scannet_frame_splits.py."
            )

    def __getitem__(self, idx):
        scene_name, ref_frame = self.samples[idx]
        scene_dir = self.scans_dir / scene_name
        frame_ids = [ref_frame + offset for offset in self.window_offsets]
        self._validate_window(scene_dir, frame_ids)

        cam_intrins = self._get_cam_intrinsics(scene_dir, scene_name)

        color_aug = False
        if self.mode == "train" and getattr(
            self.args, "data_augmentation_color", False
        ):
            if random.random() > 0.5:
                color_aug = True
                aug_gamma = random.uniform(0.9, 1.1)
                aug_brightness = random.uniform(0.75, 1.25)
                aug_colors = np.random.uniform(0.9, 1.1, size=3)

        data_array = []
        for i, frame_id in enumerate(frame_ids):
            image_path = scene_dir / "color" / f"{frame_id}.jpg"
            depth_path = scene_dir / "depth" / f"{frame_id}.png"
            pose_path = scene_dir / "pose" / f"{frame_id}.txt"

            image = (
                Image.open(image_path)
                .convert("RGB")
                .resize((self.img_W, self.img_H), resample=Image.BILINEAR)
            )
            image = np.asarray(image, dtype=np.float32) / 255.0
            if color_aug:
                image = self._augment_image(
                    image, aug_gamma, aug_brightness, aug_colors
                )
            image = self.normalize(torch.from_numpy(image).permute(2, 0, 1))

            if i == self.ref_index:
                depth = Image.open(depth_path).resize(
                    (self.img_W, self.img_H),
                    resample=Image.NEAREST,
                )
                depth = np.asarray(depth, dtype=np.float32)[:, :, None] / 1000.0
                gt_dmap = torch.from_numpy(depth).permute(2, 0, 1)
            else:
                gt_dmap = 0.0

            extm = _read_extm_from_txt(pose_path)

            data_array.append(
                {
                    "img": image,
                    "gt_dmap": gt_dmap,
                    "extM": extm,
                    "scene_name": scene_name,
                    "img_idx": str(frame_id),
                }
            )

        return data_array, cam_intrins

    @staticmethod
    def _augment_image(image, gamma, brightness, colors):
        image_aug = image ** gamma
        image_aug = image_aug * brightness
        white = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        color_image = np.stack(
            [white * colors[i] for i in range(3)],
            axis=2,
        )
        image_aug *= color_image
        return np.clip(image_aug, 0.0, 1.0)
