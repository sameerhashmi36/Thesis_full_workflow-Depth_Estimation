"""
Evaluation of (fine)tuned monodepth2 model on AGCO test set.

This script:
- loads encoder + depth decoder from a weights folder (encoder.pth, depth.pth)
- evaluates on the fixed AGCO test set (AGCO_TEST_BAG_NAMES)
- prints monodepth2-style metrics
- - --use-smooth : evaluate using smooth dataset version
- --test-fraction : frame subsampling within test set
- --vis-dir / --vis-prob : saves RGB | GT color | GT gray | Pred color | Pred gray


Usage example (from monodepth2 repo root):

  python Finetuning_and_eval_on_agco_dataset/eval_finetuned_monodepth2_agco.py \
  --raw-root /home/sameer/Documents/Zoedepth_v1/raw_dataset_cpu_manual_1 \
  --weights-folder ./models_finetuned_on_agco/agco_mono_train20_val20 \
  --test-fraction 0.3 \
  --vis-dir ./agco_test_vis \
  --vis-prob 0.02
"""

import sys
import argparse
from pathlib import Path
import random

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import networks
from layers import disp_to_depth

from agco_config import (
    DEFAULT_RAW_ROOT,
    DEFAULT_VAL_BAG_FRACTION,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VAL_FRACTION,
    DEFAULT_TEST_FRACTION,
    MIN_DEPTH,
    MAX_DEPTH,
    DEFAULT_SEED,
)


def compute_depth_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def colorize_depth(depth_m, vmin=None, vmax=None):
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    d[d < 0] = 0

    m = d > 0
    if not m.any():
        return np.zeros((d.shape[0], d.shape[1], 3), np.uint8)

    if vmin is None:
        vmin = np.percentile(d[m], 2)
    if vmax is None:
        vmax = np.percentile(d[m], 98)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    dn = np.zeros_like(d, np.uint8)
    dn[m] = np.clip((d[m] - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)

    cm = cv2.applyColorMap(dn, cv2.COLORMAP_JET)  # BGR
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def depth_to_grayscale(depth_m, vmin=None, vmax=None):
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    d[d < 0] = 0

    m = d > 0
    if not m.any():
        return np.zeros(d.shape, np.uint8)

    if vmin is None:
        vmin = np.percentile(d[m], 2)
    if vmax is None:
        vmax = np.percentile(d[m], 98)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    norm = np.clip((d - vmin) / (vmax - vmin) * 255.0, 0, 255)
    return norm.astype(np.uint8)


def load_model(weights_folder: Path, device: torch.device):
    enc_p = weights_folder / "encoder.pth"
    dep_p = weights_folder / "depth.pth"
    if not enc_p.exists() or not dep_p.exists():
        raise FileNotFoundError(f"Cannot find encoder.pth/depth.pth in {weights_folder}")

    print(f"-> Loading weights from {weights_folder}")
    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    enc = torch.load(enc_p, map_location=device)
    enc = {k: v for k, v in enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(enc)

    dec = torch.load(dep_p, map_location=device)
    decoder.load_state_dict(dec)

    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--weights-folder", type=str, default="./models_finetuned_on_agco/agco_mono_640x192_finetuned_smooth_train80_val80")

    p.add_argument("--use-smooth", action="store_true")
    p.add_argument("--img-width", type=int, default=640)
    p.add_argument("--img-height", type=int, default=192)

    # split params (test_fraction matters here)
    p.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    p.add_argument("--val-bag-fraction", type=float, default=DEFAULT_VAL_BAG_FRACTION)  # not used for test, but kept consistent
    p.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)      # not used for test
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)          # not used for test
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--no-median-scaling", action="store_true")

    p.add_argument("--vis-dir", type=str, default="")
    p.add_argument("--vis-prob", type=float, default=0.02)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.use_smooth:
        from agco_dataset_smooth import AGCODepthDataset
        print("Dataset: SMOOTH")
    else:
        from agco_dataset import AGCODepthDataset
        print("Dataset: NORMAL")

    test_ds = AGCODepthDataset(
        raw_root=args.raw_root,
        split="test",
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_bag_fraction=args.val_bag_fraction,
        seed=args.seed,
        img_width=args.img_width,
        img_height=args.img_height,
        verbose_bags=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Test samples: {len(test_ds)} (test_fraction={args.test_fraction})")

    encoder, decoder = load_model(Path(args.weights_folder), device)

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print("Visualizations →", vis_dir.resolve())

    rng = random.Random(42)
    all_errs = []
    vis_idx = 0

    pbar = tqdm(test_loader, desc="test", leave=True)
    with torch.no_grad():
        for rgb, depth_gt, valid_mask, meta in pbar:
            rgb = rgb.to(device)
            depth_gt = depth_gt.to(device)
            valid_mask = valid_mask.to(device)

            feats = encoder(rgb)
            disp = decoder(feats)[("disp", 0)]

            if disp.shape[-2:] != depth_gt.shape[-2:]:
                disp = F.interpolate(disp, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

            _, depth_pred = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)

            mask = (valid_mask > 0) & (depth_gt > MIN_DEPTH) & (depth_gt < MAX_DEPTH)
            if mask.sum().item() == 0:
                continue

            gt_flat = depth_gt[mask].cpu().numpy().astype(np.float32)
            pr_flat = depth_pred[mask].cpu().numpy().astype(np.float32)

            if not args.no_median_scaling:
                scale = np.median(gt_flat) / (np.median(pr_flat) + 1e-12)
                pr_eval = np.clip(pr_flat * scale, MIN_DEPTH, MAX_DEPTH)
            else:
                scale = 1.0
                pr_eval = np.clip(pr_flat, MIN_DEPTH, MAX_DEPTH)

            all_errs.append(compute_depth_errors(gt_flat, pr_eval))

            m = np.array(all_errs).mean(axis=0)
            pbar.set_postfix({"abs_rel": f"{m[0]:.3f}", "rmse": f"{m[2]:.3f}"})

            # visual save (first element only)
            if vis_dir is not None and rng.random() < float(args.vis_prob):
                rgb_np = (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

                gt_img = depth_gt[0, 0].cpu().numpy().astype(np.float32)
                pr_img = (depth_pred[0, 0].cpu().numpy().astype(np.float32) * float(scale))

                gt_img = np.clip(gt_img, MIN_DEPTH, MAX_DEPTH)
                pr_img = np.clip(pr_img, MIN_DEPTH, MAX_DEPTH)

                gt_color = colorize_depth(gt_img)
                pr_color = colorize_depth(pr_img)

                gt_gray = depth_to_grayscale(gt_img)
                pr_gray = depth_to_grayscale(pr_img)

                H, W = rgb_np.shape[:2]
                gt_color = cv2.resize(gt_color, (W, H), interpolation=cv2.INTER_NEAREST)
                pr_color = cv2.resize(pr_color, (W, H), interpolation=cv2.INTER_NEAREST)
                gt_gray = cv2.resize(gt_gray, (W, H), interpolation=cv2.INTER_NEAREST)
                pr_gray = cv2.resize(pr_gray, (W, H), interpolation=cv2.INTER_NEAREST)

                gt_gray_3c = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2RGB)
                pr_gray_3c = cv2.cvtColor(pr_gray, cv2.COLOR_GRAY2RGB)

                stacked = np.concatenate([rgb_np, gt_color, gt_gray_3c, pr_color, pr_gray_3c], axis=1)
                out_name = vis_dir / f"agco_test_sample_{vis_idx:05d}.png"
                cv2.imwrite(str(out_name), cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
                vis_idx += 1

    if not all_errs:
        print("No valid pixels found for metrics.")
        return

    mean = np.array(all_errs).mean(axis=0)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean

    print("\n# ---- AGCO monodepth2 evaluation (test set) ----")
    print(f"  abs_rel  : {abs_rel:.3f}")
    print(f"  sq_rel   : {sq_rel:.3f}")
    print(f"  rmse     : {rmse:.3f} m")
    print(f"  rmse_log : {rmse_log:.3f}")
    print(f"  a1       : {a1:.3f}")
    print(f"  a2       : {a2:.3f}")
    print(f"  a3       : {a3:.3f}")
    print("# ----------------------------------------------")


if __name__ == "__main__":
    main()
