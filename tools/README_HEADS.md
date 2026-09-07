# 7Scenes heads -> original MaGNet smoke test

Copy the patch files into the MaGNet repository root while preserving paths.

Expected current dataset layout:

```text
~/26workspace/datasets/SevenScenes/heads/
└── heads/
    ├── seq-01/
    ├── seq-02/
    ├── TrainSplit.txt
    └── TestSplit.txt
```

The dataset root passed to the scripts is therefore:

```text
~/26workspace/datasets/SevenScenes/heads
```

## 1. Data sanity check

```bash
python tools/check_7scenes_heads.py \
  --dataset_root ~/26workspace/datasets/SevenScenes/heads
```

## 2. Check checkpoints

```bash
ls -lh ckpts/DNET_scannet.pt ckpts/FNET_scannet.pt ckpts/MAGNET_scannet.pt
```

If missing, from the official MaGNet repository run:

```bash
python ckpts/download.py
```

## 3. One-sample original MaGNet smoke test

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python test_MaGNet_heads.py \
  --dataset_root ~/26workspace/datasets/SevenScenes/heads \
  --dnet_ckpt ./ckpts/DNET_scannet.pt \
  --fnet_ckpt ./ckpts/FNET_scannet.pt \
  --magnet_ckpt ./ckpts/MAGNET_scannet.pt \
  --gpu 0 \
  --max_samples 1
```

## 4. Run all heads evaluation frames

```bash
python test_MaGNet_heads.py \
  --dataset_root ~/26workspace/datasets/SevenScenes/heads \
  --max_samples 0
```

The official Long-style 7Scenes split contains heads sequence 1 frames 0,10,...,990.
With four source views and window radius 20, the nominal frame offsets are
-20,-10,0,+10,+20, with the center frame used as reference.

## If RTX 4060 8GB reports CUDA OOM

First prove the forward path with fewer source views:

```bash
python test_MaGNet_heads.py \
  --dataset_root ~/26workspace/datasets/SevenScenes/heads \
  --max_samples 1 \
  --num_source_views 2 \
  --window_radius 20 \
  --num_test_iter 1
```

This is only a smoke-test fallback. Return to 4 source views and 3 test iterations
for the original 7Scenes baseline.
