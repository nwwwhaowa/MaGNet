# ScanNet training on Jetson Orin (1 TB storage)

This guide covers ScanNet preparation and Orin setup. On the `二次训练` branch,
the ScanNet trainer uses direct depth supervision. See
[SECOND_TRAINING.md](SECOND_TRAINING.md) for the current training and ablation protocol.

## Added files

- `train_StructMaGNet_scannet.py`: ScanNet depth-supervised retraining.
- `data/dataloader_scannet_train.py`: strict train/val loader using frame-level split files.
- `tools/scannet/export_sens_py3.py`: Python 3 streaming `.sens` exporter that can retain only every Nth frame.
- `tools/scannet/build_scannet_frame_splits.py`: builds valid `<scene> <reference_frame>` train/val lists.
- `requirements_orin.txt`: Orin-safe Python dependencies; intentionally excludes `torch` and `torchvision`.

## Environment setup on Orin

Do **not** install the repository's legacy `requirements.txt` on Jetson Orin because it pins old desktop PyTorch versions. Keep the already working Jetson-compatible PyTorch/CUDA environment and install only the missing Python packages:

```bash
python -m pip install -r requirements_orin.txt
```

If you only hit `ModuleNotFoundError: No module named 'scipy'`, the minimal fix is:

```bash
python -m pip install scipy
```

Then verify:

```bash
python - <<'PY'
import torch
import torchvision
import scipy

print('torch       :', torch.__version__)
print('torchvision :', torchvision.__version__)
print('scipy       :', scipy.__version__)
print('cuda        :', torch.version.cuda)
print('cuda avail  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU         :', torch.cuda.get_device_name(0))
print('environment : PASS')
PY
```

## 1 TB storage strategy

Do **not** download the full ScanNet release onto the Orin. StructMaGNet Phase-A only needs RGB, depth, camera pose and camera intrinsics from `.sens` sequences.

Process ScanNet scene-by-scene:

1. Download one `.sens` file.
2. Export only every 5th frame directly into the final `scans/` tree.
3. Verify that scene.
4. Delete the temporary `.sens` file.
5. Continue with the next scene.

This keeps peak storage close to `compact extracted dataset + one temporary .sens file`, rather than `all .sens + all extracted frames`.

With `--stride 5`, the default StructMaGNet window remains valid:

```text
window_radius=20, source_views=4
=> offsets [-20, -10, 0, +10, +20]
```

All offsets are multiples of 5, and the exporter preserves original frame IDs.

## Recommended layout

```text
~/26workspace/
├── StructMaGNet/MaGNet/
└── datasets/ScanNet/
    ├── scans/
    │   ├── scene0000_00/
    │   │   ├── color/
    │   │   ├── depth/
    │   │   ├── pose/
    │   │   └── intrinsic/
    │   └── ...
    ├── raw_sens/        # temporary
    └── official_splits/
        ├── scannetv2_train.txt
        └── scannetv2_val.txt
```

## Export one scene at stride 5

```bash
python tools/scannet/export_sens_py3.py \
  --sens /path/to/scene0000_00.sens \
  --dataset_root ~/26workspace/datasets/ScanNet \
  --stride 5
```

After checking the output, remove the temporary `.sens` manually.

## Build strict frame-level splits

```bash
python tools/scannet/build_scannet_frame_splits.py \
  --dataset_root ~/26workspace/datasets/ScanNet \
  --train_scenes ~/26workspace/datasets/ScanNet/official_splits/scannetv2_train.txt \
  --val_scenes ~/26workspace/datasets/ScanNet/official_splits/scannetv2_val.txt \
  --train_out ./data_split/scannet_train_stride5.txt \
  --val_out ./data_split/scannet_val_stride5.txt \
  --reference_stride 5 \
  --window_radius 20 \
  --num_source_views 4
```

Only reference frames with a complete RGB/depth/pose window and finite poses are written.

## First Orin smoke training

Do not reinstall PyTorch from the repository `requirements.txt`; keep the already working Jetson PyTorch/CUDA environment.

```bash
python train_StructMaGNet_scannet.py \
  --dataset_root ~/26workspace/datasets/ScanNet \
  --train_split ./data_split/scannet_train_stride5.txt \
  --val_split ./data_split/scannet_val_stride5.txt \
  --dnet_ckpt ./ckpts/DNET_scannet.pt \
  --fnet_ckpt ./ckpts/FNET_scannet.pt \
  --magnet_ckpt ./ckpts/MAGNET_scannet.pt \
  --batch_size 1 \
  --num_workers 1 \
  --epochs 1 \
  --lr 1e-4 \
  --gate_mode learned \
  --max_train_steps 10 \
  --val_max_samples 8 \
  --amp
```

Then increase to 50-100 steps before running a full epoch.

Monitor the Orin with:

```bash
tegrastats
```

If unified-memory pressure is high, keep `batch_size=1`, leave `--pin_memory` disabled, and use `num_workers=0` or `1`.

## Recommended dataset growth

1. Smoke: 1-2 train scenes + 1 val scene.
2. Pilot: 16-32 train scenes + 4-8 val scenes.
3. Medium: 100-200 train scenes + a fixed validation subset.
4. Final: expand the compact stride-5 set as far as available storage permits.

If the full official ScanNet train split cannot fit on the 1 TB device, train MaGNet and all StructMaGNet ablations on the **same scene/frame subset** for fair comparisons.

## Current training behavior

```text
D-Net          frozen
F-Net          frozen
G-Net          trainable
Upsampling     trainable
GeometryGate   trainable
translation    unchanged
rotation noise online
```

The existing Gaussian depth NLL optimizes G-Net, upsampling and the shared gate.
There is no oracle gate loss, and injected noise angles are not supplied to the model.
Old gate-only checkpoints cannot resume this training; use its `last.pt` or `best.pt`.
