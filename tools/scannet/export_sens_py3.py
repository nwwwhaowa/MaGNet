#!/usr/bin/env python3
"""Python 3, low-storage ScanNet .sens exporter for MaGNet.

This reader follows ScanNet SensorData v4 and streams frames directly from the
.sens file. It never materializes the whole sequence in memory and exports
only every Nth frame.

For standard ScanNet v2 streams used by MaGNet:
  - color is JPEG-compressed inside .sens and written directly to .jpg;
  - depth is zlib-compressed uint16 and written as 16-bit PNG;
  - camera_to_world poses keep original frame IDs;
  - intrinsics/extrinsics are exported once per scene.
"""

from __future__ import annotations

import argparse
import os
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


COLOR_COMPRESSION = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def read_exact(handle, n):
    data = handle.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected EOF: requested {n} bytes, got {len(data)}")
    return data


def read_struct(handle, fmt):
    return struct.unpack(fmt, read_exact(handle, struct.calcsize(fmt)))


def read_matrix4f(handle):
    values = read_struct(handle, "<16f")
    return np.asarray(values, dtype=np.float32).reshape(4, 4)


def save_matrix(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.8f")


def save_depth_png(path, depth):
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    Image.fromarray(depth).save(path)


def export_sens(sens_path, dataset_root, scans_dir, stride, overwrite):
    sens_path = Path(sens_path).expanduser().resolve()
    if not sens_path.is_file():
        raise FileNotFoundError(sens_path)
    if stride <= 0:
        raise ValueError("--stride must be positive")

    scene_name = sens_path.stem
    scene_dir = Path(dataset_root).expanduser().resolve() / scans_dir / scene_name
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"

    for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with sens_path.open("rb") as f:
        version = read_struct(f, "<I")[0]
        if version != 4:
            raise ValueError(f"Unsupported ScanNet SensorData version {version}; expected 4")

        sensor_name_len = read_struct(f, "<Q")[0]
        sensor_name = read_exact(f, sensor_name_len).decode("utf-8", errors="replace")

        intrinsic_color = read_matrix4f(f)
        extrinsic_color = read_matrix4f(f)
        intrinsic_depth = read_matrix4f(f)
        extrinsic_depth = read_matrix4f(f)

        color_compression_id = read_struct(f, "<i")[0]
        depth_compression_id = read_struct(f, "<i")[0]
        color_width = read_struct(f, "<I")[0]
        color_height = read_struct(f, "<I")[0]
        depth_width = read_struct(f, "<I")[0]
        depth_height = read_struct(f, "<I")[0]
        depth_shift = read_struct(f, "<f")[0]
        num_frames = read_struct(f, "<Q")[0]

        color_compression = COLOR_COMPRESSION.get(color_compression_id, f"id={color_compression_id}")
        depth_compression = DEPTH_COMPRESSION.get(depth_compression_id, f"id={depth_compression_id}")

        if color_compression != "jpeg":
            raise NotImplementedError(
                "Low-storage exporter expects ScanNet JPEG color streams; "
                f"got {color_compression}."
            )
        if depth_compression != "zlib_ushort":
            raise NotImplementedError(
                "Low-storage exporter expects ScanNet zlib uint16 depth streams; "
                f"got {depth_compression}."
            )

        save_matrix(intrinsic_dir / "intrinsic_color.txt", intrinsic_color)
        save_matrix(intrinsic_dir / "extrinsic_color.txt", extrinsic_color)
        save_matrix(intrinsic_dir / "intrinsic_depth.txt", intrinsic_depth)
        save_matrix(intrinsic_dir / "extrinsic_depth.txt", extrinsic_depth)

        kept = 0
        for frame_idx in range(num_frames):
            camera_to_world = read_matrix4f(f)
            _timestamp_color = read_struct(f, "<Q")[0]
            _timestamp_depth = read_struct(f, "<Q")[0]
            color_size = read_struct(f, "<Q")[0]
            depth_size = read_struct(f, "<Q")[0]

            keep = frame_idx % stride == 0
            if keep:
                color_data = read_exact(f, color_size)
                depth_data = read_exact(f, depth_size)

                color_path = color_dir / f"{frame_idx}.jpg"
                depth_path = depth_dir / f"{frame_idx}.png"
                pose_path = pose_dir / f"{frame_idx}.txt"

                if overwrite or not color_path.exists():
                    with color_path.open("wb") as out:
                        out.write(color_data)

                if overwrite or not depth_path.exists():
                    depth_raw = zlib.decompress(depth_data)
                    depth = np.frombuffer(depth_raw, dtype=np.uint16)
                    expected = depth_width * depth_height
                    if depth.size != expected:
                        raise ValueError(
                            f"{scene_name} frame {frame_idx}: depth contains "
                            f"{depth.size} uint16 values, expected {expected}"
                        )
                    depth = depth.reshape(depth_height, depth_width)
                    save_depth_png(depth_path, depth)

                if overwrite or not pose_path.exists():
                    save_matrix(pose_path, camera_to_world)

                kept += 1
            else:
                f.seek(color_size + depth_size, os.SEEK_CUR)

    print("ScanNet .sens export complete")
    print("  scene            :", scene_name)
    print("  sensor           :", sensor_name)
    print("  color            :", f"{color_width}x{color_height} {color_compression}")
    print("  depth            :", f"{depth_width}x{depth_height} {depth_compression}")
    print("  depth shift      :", depth_shift)
    print("  total frames     :", num_frames)
    print("  stride           :", stride)
    print("  retained frames  :", kept)
    print("  output           :", scene_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sens", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--scans_dir", default="scans")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    export_sens(args.sens, args.dataset_root, args.scans_dir, args.stride, args.overwrite)


if __name__ == "__main__":
    main()
