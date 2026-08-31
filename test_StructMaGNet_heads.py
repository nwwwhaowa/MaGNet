#!/usr/bin/env python3
"""Stage-1 StructMaGNet smoke test on the 7-Scenes heads subset.

Stage-1 goal:
    Original MaGNet
    + GeometryGate
    + gated Gaussian update

This script does NOT inject rotation noise yet.
rot_unc is explicitly set to zero so that we can first verify:
  1) original MaGNet weights can be transferred into STRUCTMAGNET;
  2) the new GeometryGate forward path runs correctly;
  3) auxiliary maps (gate / entropy / peak) can be inspected.

Example:
  python test_StructMaGNet_heads.py \
      --dataset_root ~/26workspace/datasets/SevenScenes/heads \
      --dnet_ckpt ./ckpts/DNET_scannet.pt \
      --fnet_ckpt ./ckpts/FNET_scannet.pt \
      --magnet_ckpt ./ckpts/MAGNET_scannet.pt \
      --max_samples 1

After Stage-1 training, optionally load the full StructMaGNet checkpoint:
  python test_StructMaGNet_heads.py \
      --dataset_root ~/26workspace/datasets/SevenScenes/heads \
      --magnet_ckpt ./ckpts/MAGNET_scannet.pt \
      --struct_ckpt ./exp/STRUCTMAGNET/model.pt \
      --max_samples 100
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Locate repository root.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = None
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    if ((_parent / "utils").is_dir()
            and (_parent / "data").is_dir()
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
from models.STRUCTMAGNET import STRUCTMAGNET


def build_model_args(cli):
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

        # misc
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


def load_magnet_backbone(fpath, model):
    """Load original MaGNet weights into STRUCTMAGNET with strict=False.

    New Stage-1 parameters such as geometry_gate.* are intentionally left at
    their own initialization values.
    """
    ckpt = torch.load(fpath, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(
            f"Checkpoint {fpath} does not contain key 'model'. "
            f"Available keys: {list(ckpt.keys())}"
        )

    raw_state = ckpt["model"]
    model_state = model.state_dict()
    load_state = {}
    skipped_shape = []

    for key, value in raw_state.items():
        clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key

        if clean_key not in model_state:
            continue

        if model_state[clean_key].shape != value.shape:
            skipped_shape.append(
                (clean_key, tuple(value.shape), tuple(model_state[clean_key].shape))
            )
            continue

        load_state[clean_key] = value

    msg = model.load_state_dict(load_state, strict=False)

    print("loaded original MaGNet tensors:", len(load_state))
    if msg.missing_keys:
        print("new/unloaded StructMaGNet keys:")
        for key in msg.missing_keys:
            print("  -", key)

    if msg.unexpected_keys:
        print("unexpected keys:")
        for key in msg.unexpected_keys:
            print("  -", key)

    if skipped_shape:
        print("shape-mismatched keys:")
        for key, old_shape, new_shape in skipped_shape:
            print(f"  - {key}: ckpt={old_shape}, model={new_shape}")

    return model


def _last_aux(aux, key):
    """Get the final-iteration tensor from an aux dictionary."""
    value = aux.get(key, None)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[-1]
    return value


def _to_2d_numpy(x):
    if x is None:
        return None

    x = x.detach().float().cpu()

    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]

    return x.numpy()


def save_visuals(
    save_dir,
    idx,
    ref_img,
    gt,
    pred,
    stdev,
    geometry_gate=None,
    cost_entropy=None,
    cost_peak=None,
):
    os.makedirs(save_dir, exist_ok=True)

    img = ref_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    img = utils.unnormalize(img)

    gt_np = gt.detach().cpu().numpy()[0, 0]
    pred_np = pred.detach().cpu().numpy()[0, 0]
    std_np = stdev.detach().cpu().numpy()[0, 0]

    err_np = np.abs(pred_np - gt_np)
    valid = np.logical_and(gt_np > 1e-3, gt_np < 10.0)
    err_np[~valid] = 0.0

    gate_np = _to_2d_numpy(geometry_gate)
    entropy_np = _to_2d_numpy(cost_entropy)
    peak_np = _to_2d_numpy(cost_peak)

    stem = f"{idx:04d}"

    plt.imsave(os.path.join(save_dir, f"{stem}_rgb.png"), img)
    plt.imsave(
        os.path.join(save_dir, f"{stem}_gt_depth.png"),
        gt_np,
        vmin=0.0,
        vmax=5.0,
    )
    plt.imsave(
        os.path.join(save_dir, f"{stem}_pred_depth.png"),
        pred_np,
        vmin=0.0,
        vmax=5.0,
    )
    plt.imsave(
        os.path.join(save_dir, f"{stem}_abs_error.png"),
        err_np,
        vmin=0.0,
        vmax=1.0,
    )
    plt.imsave(
        os.path.join(save_dir, f"{stem}_stdev.png"),
        std_np,
        vmin=0.0,
        vmax=1.0,
    )

    if gate_np is not None:
        plt.imsave(
            os.path.join(save_dir, f"{stem}_geometry_gate.png"),
            gate_np,
            vmin=0.0,
            vmax=1.0,
        )

    if entropy_np is not None:
        plt.imsave(
            os.path.join(save_dir, f"{stem}_cost_entropy.png"),
            entropy_np,
        )

    if peak_np is not None:
        plt.imsave(
            os.path.join(save_dir, f"{stem}_cost_peak.png"),
            peak_np,
            vmin=0.0,
            vmax=1.0,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", required=True)

    parser.add_argument("--dnet_ckpt", default="./ckpts/DNET_scannet.pt")
    parser.add_argument("--fnet_ckpt", default="./ckpts/FNET_scannet.pt")
    parser.add_argument("--magnet_ckpt", default="./ckpts/MAGNET_scannet.pt")

    # Optional full Stage-1 StructMaGNet checkpoint.
    parser.add_argument(
        "--struct_ckpt",
        default=None,
        help=(
            "Optional trained StructMaGNet checkpoint. "
            "If omitted, original MaGNet weights are loaded into the "
            "compatible backbone and the new GeometryGate keeps its initialization."
        ),
    )

    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1,
        help="1 for smoke test; 0 means all heads samples",
    )
    parser.add_argument("--num_source_views", type=int, default=4)
    parser.add_argument("--window_radius", type=int, default=20)
    parser.add_argument("--num_test_iter", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument(
        "--save_dir",
        default="./exp/STRUCTMAGNET/heads_stage1",
    )

    cli = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. StructMaGNet smoke test expects a CUDA GPU."
        )

    device = torch.device("cuda:0")
    args = build_model_args(cli)

    for name, path in [
        ("D-Net", args.DNET_ckpt),
        ("F-Net", args.FNET_ckpt),
        ("MaGNet", args.MAGNET_ckpt),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    if cli.struct_ckpt is not None and not os.path.isfile(cli.struct_ckpt):
        raise FileNotFoundError(
            f"StructMaGNet checkpoint not found: {cli.struct_ckpt}"
        )

    print("=== configuration ===")
    print("dataset_root      :", args.dataset_path)
    print("source views      :", args.MAGNET_num_source_views)
    print("window radius     :", args.MAGNET_window_radius)
    print("test iterations   :", args.MAGNET_num_test_iter)
    print("rotation noise    : disabled (Stage-1)")
    print("rot_unc input     : all zeros")
    print("device            :", torch.cuda.get_device_name(0))

    print("\n[1/3] building StructMaGNet...")
    model = STRUCTMAGNET(args).to(device)

    print("[2/3] loading original MaGNet backbone:", args.MAGNET_ckpt)
    model = load_magnet_backbone(args.MAGNET_ckpt, model)

    if cli.struct_ckpt is not None:
        print("loading trained StructMaGNet checkpoint:", cli.struct_ckpt)
        model = utils.load_checkpoint(cli.struct_ckpt, model)

    model.eval()

    print("[3/3] loading heads data...")
    test_loader = SevenScenesHeadsLoader(args, "test").data
    print("heads samples:", len(test_loader.dataset))

    metrics = utils.RunningAverageDict()
    forward_times = []
    gate_means = []

    limit = (
        len(test_loader.dataset)
        if cli.max_samples == 0
        else min(cli.max_samples, len(test_loader.dataset))
    )

    with torch.inference_mode():
        for batch_idx, (data_array, cam_intrins) in enumerate(test_loader):
            if batch_idx >= limit:
                break

            batch_size = data_array[0]["img"].shape[0]

            ref_dat, nghbr_dats, nghbr_poses, is_valid = utils.data_preprocess(
                data_array,
                batch_size,
            )

            ref_img = ref_dat["img"].to(device, non_blocking=True)
            gt = ref_dat["gt_dmap"].to(device, non_blocking=True)
            gt[gt > args.max_depth] = 0.0

            src_imgs = torch.cat(
                [
                    d["img"].to(device, non_blocking=True)
                    for d in nghbr_dats
                ],
                dim=0,
            )

            nghbr_poses = nghbr_poses.to(device, non_blocking=True)

            # Stage-1: no pose perturbation yet.
            # Shape: B x V
            
            V = nghbr_poses.shape[1]
            
            rot_unc = torch.zeros(
            batch_size,V,dtype=ref_img.dtype,device=device,)

            rot_hyp_vec = torch.zeros(
            batch_size,V,3,dtype=ref_img.dtype,device=device,)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            model_out = model(
                ref_img,
                src_imgs,
                nghbr_poses,
                is_valid,
                cam_intrins,
                rot_unc=rot_unc,
                rot_hyp_vec=rot_hyp_vec,
                mode="test",
                return_aux=True,
            )

            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - t0)

            if not (
                isinstance(model_out, tuple)
                and len(model_out) == 2
            ):
                raise RuntimeError(
                    "STRUCTMAGNET(return_aux=True) must return "
                    "(pred_list, aux)."
                )

            pred_list, aux = model_out

            pred, stdev = torch.split(pred_list[-1], 1, dim=1)

            gate_mean = (
            aux["geometry_gate"][-1]
            .detach()
            .mean()
            .item()
            )

            # Prefer the value returned by the model when available.
            # Fall back to the test input so this script also works when
            # STRUCTMAGNET.py does not expose rot_hyp_vec in aux yet.
            rot_hyp_out = aux.get("rot_hyp_vec", rot_hyp_vec)
            rot_hyp_norm = (
                rot_hyp_out
                .detach()
                .norm(dim=-1)
                .mean()
                .item()
            )

            geometry_gate = _last_aux(aux, "geometry_gate")
            cost_entropy = _last_aux(aux, "cost_entropy")
            cost_peak = _last_aux(aux, "cost_peak")

            if geometry_gate is not None:
                gate_mean = geometry_gate.detach().float().mean().item()
                gate_means.append(gate_mean)
            else:
                gate_mean = float("nan")

            gt_np = gt.detach().cpu().numpy()[0, 0]
            pred_np = pred.detach().cpu().numpy()[0, 0]
            var_np = np.square(
                stdev.detach().cpu().numpy()[0, 0]
            )

            valid = np.logical_and(
                gt_np > args.min_depth,
                gt_np < args.max_depth,
            )

            pred_np = np.nan_to_num(
                pred_np,
                nan=args.min_depth,
                posinf=args.max_depth,
                neginf=args.min_depth,
            )
            pred_np = np.clip(
                pred_np,
                args.min_depth,
                args.max_depth,
            )

            if not valid.any():
                print(
                    f"[WARN] sample {batch_idx}: "
                    "no valid GT pixels; skipped"
                )
                continue

            cur_metrics = utils.compute_depth_errors(
                gt_np[valid],
                pred_np[valid],
                var_np[valid],
            )

            metrics.update(cur_metrics)

            print(
                f"sample={batch_idx:03d} "
                f"ref={int(ref_dat['img_idx'][0]):06d} "
                f"RMSE={cur_metrics['rmse']:.4f} "
                f"AbsRel={cur_metrics['abs_rel']:.4f} "
                f"a1={cur_metrics['a1']:.4f} "
                f"gate={gate_mean:.4f} "
                f"rot_hyp_norm={rot_hyp_norm:.6f} "
                f"forward={forward_times[-1] * 1000:.1f} ms"
            )

            if batch_idx == 0:
                save_visuals(
                    cli.save_dir,
                    batch_idx,
                    ref_img,
                    gt,
                    pred,
                    stdev,
                    geometry_gate=geometry_gate,
                    cost_entropy=cost_entropy,
                    cost_peak=cost_peak,
                )

    result = metrics.get_value()

    if result:
        print("\n=== aggregate ===")
        print("samples :", limit)
        print("RMSE    : %.4f" % result["rmse"])
        print("AbsRel  : %.4f" % result["abs_rel"])
        print("a1      : %.4f" % result["a1"])
        print("NLL     : %.4f" % result["nll"])
        print(
            "forward : %.1f ms (mean)"
            % (1000.0 * np.mean(forward_times))
        )

        if gate_means:
            print(
                "gate    : %.4f (mean)"
                % np.mean(gate_means)
            )

        print("visuals :", cli.save_dir)

        print(
            "\n[PASS] StructMaGNet Stage-1 forward completed "
            "on 7Scenes/heads."
        )


if __name__ == "__main__":
    main()
