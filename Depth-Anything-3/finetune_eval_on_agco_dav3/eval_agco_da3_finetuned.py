"""
eval_agco_da3_strict.py

Evaluation of finetuned Depth-Anything-3 (DA3) on AGCO test split with optional visualization.

- Loads base model: DepthAnything3.from_pretrained(model_id)
- Loads finetuned weights into da3.model (strict=True)
- Evaluates on valid LiDAR pixels: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
- Optional:
  --median-scaling
  --vis-dir + --vis-prob
  --save-npy

Visualization output:
  [RGB | GT(depth) | Pred(depth)]
and an extra grayscale "depth-map" image:
  [RGB | GT(gray near-bright) | Pred(gray near-bright)]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from agco_da3_config import (
    DEFAULT_RAW_ROOT, DEFAULT_DA3_REPO_ROOT, DEFAULT_DA3_MODEL_ID,
    MIN_DEPTH_M, MAX_DEPTH_M, TRAIN_H, TRAIN_W, SPLIT_SEED
)
# from agco_da3_dataset import AGCODA3DepthDataset
from agco_da3_dataset_smooth import AGCODA3DepthDatasetSmooth as AGCODA3DepthDataset



def choose_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_errors(gt: np.ndarray, pred: np.ndarray):
    pred = np.clip(pred, 1e-8, None)
    gt = np.clip(gt, 1e-8, None)
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def _normalize_depth_to_8bit(depth_m: np.ndarray, vmin: float, vmax: float, invert: bool = False):
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin + 1e-8)
    if invert:
        norm = 1.0 - norm
    return (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)


def colorize_depth(depth_m: np.ndarray, vmin: float, vmax: float):
    d8 = _normalize_depth_to_8bit(depth_m, vmin, vmax, invert=False)
    cm = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def gray_depth_near_bright(depth_m: np.ndarray, vmin: float, vmax: float):
    # near=bright => invert=True
    d8 = _normalize_depth_to_8bit(depth_m, vmin, vmax, invert=True)
    return np.stack([d8, d8, d8], axis=-1)


def forward_da3_metric(net: torch.nn.Module, rgb_bchw: torch.Tensor) -> torch.Tensor:
    """
    DA3 training wrapper expects (B,1,C,H,W). Keeping that.
    Returns depth as (B,1,h,w).
    """
    x = rgb_bchw.unsqueeze(1)  # (B,1,C,H,W)
    out = net(x, None, None, (), False)

    if isinstance(out, dict):
        for k in ["metric_depth", "depth", "pred", "out"]:
            if k in out:
                out = out[k]
                break
    if isinstance(out, (list, tuple)):
        out = out[0]

    # Normalize to (B,1,H,W)
    if out.dim() == 5:
        # (B,1,1,h,w) or (B,1,h,w,?) depending on impl; earlier assumption was (B,1,h,w)
        # Most DA3 wrappers: (B,1,1,h,w) -> squeeze the middle singleton
        if out.shape[1] == 1 and out.shape[2] == 1:
            out = out[:, 0]  # (B,1,h,w)
        else:
            out = out[:, 0]  # best-effort
    if out.dim() == 4:
        if out.shape[1] != 1:
            out = out[:, 0].unsqueeze(1)
    elif out.dim() == 3:
        out = out.unsqueeze(1)

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--da3-repo-root", type=str, default=str(DEFAULT_DA3_REPO_ROOT))
    p.add_argument("--model-id", type=str, default=DEFAULT_DA3_MODEL_ID)

    p.add_argument("--finetuned-weights", type=str, required=True)

    p.add_argument("--img-height", type=int, default=TRAIN_H)
    p.add_argument("--img-width", type=int, default=TRAIN_W)
    p.add_argument("--min-depth", type=float, default=MIN_DEPTH_M)
    p.add_argument("--max-depth", type=float, default=MAX_DEPTH_M)

    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=1.0)
    p.add_argument("--test-fraction", type=float, default=1.0)
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--val-bag-fraction", type=float, default=0.2)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--median-scaling", action="store_true")
    p.add_argument("--vis-dir", type=str, default="")
    p.add_argument("--vis-prob", type=float, default=0.02)
    p.add_argument("--save-npy", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # Import DA3 from local repo if provided
    da3_repo_root = Path(args.da3_repo_root)
    if da3_repo_root.exists() and str(da3_repo_root) not in sys.path:
        sys.path.append(str(da3_repo_root))

    from depth_anything_3.api import DepthAnything3

    device = choose_device()
    print("Device:", device)

    test_ds = AGCODA3DepthDataset(
        raw_root=args.raw_root,
        split="test",
        img_height=args.img_height,
        img_width=args.img_width,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        val_bag_fraction=args.val_bag_fraction,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        verbose=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Test samples: {len(test_ds)}")

    da3 = DepthAnything3.from_pretrained(args.model_id).to(device=device)
    net = da3.model.to(device).eval()

    ft = Path(args.finetuned_weights)
    if not ft.is_file():
        raise FileNotFoundError(f"Finetuned weights not found: {ft}")
    state = torch.load(str(ft), map_location=device)
    net.load_state_dict(state, strict=True)
    print("[FT] strict=True load OK ✅")

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print("Saving visualizations to:", vis_dir)

    rng = random.Random(42)
    all_errors = []
    vis_idx = 0

    with torch.no_grad():
        for rgb, depth_gt, valid_mask, meta in tqdm(test_loader, desc="[Test]", ncols=110):
            rgb = rgb.to(device, non_blocking=True)               # (B,3,H,W)
            depth_gt = depth_gt.to(device, non_blocking=True)     # (B,H,W) or (B,1,H,W) depending on dataset
            valid_mask = valid_mask.to(device, non_blocking=True) # (B,H,W) or (B,1,H,W)

            if depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(1)   # (B,1,H,W)
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.bool()

            pred = forward_da3_metric(net, rgb)    # (B,1,h,w)

            Hp, Wp = pred.shape[-2:]
            if depth_gt.shape[-2:] != (Hp, Wp):
                depth_gt_r = F.interpolate(depth_gt, size=(Hp, Wp), mode="nearest")
                valid_r = F.interpolate(valid_mask.float(), size=(Hp, Wp), mode="nearest").bool()
            else:
                depth_gt_r = depth_gt
                valid_r = valid_mask

            # Evaluate per-sample (IMPORTANT)
            B = rgb.shape[0]
            for b in range(B):
                gt_b = depth_gt_r[b, 0]
                pr_b = pred[b, 0]
                m_b = valid_r[b, 0] & (gt_b > args.min_depth) & (gt_b < args.max_depth) & torch.isfinite(pr_b)

                if m_b.sum().item() == 0:
                    continue

                gt_np = gt_b.detach().cpu().numpy().astype(np.float32)
                pr_np = pr_b.detach().cpu().numpy().astype(np.float32)
                m_np = m_b.detach().cpu().numpy().astype(bool)

                gt_flat = gt_np[m_np]
                pr_flat = pr_np[m_np]

                if args.median_scaling:
                    scale = float(np.median(gt_flat) / (np.median(pr_flat) + 1e-8))
                else:
                    scale = 1.0

                pr_scaled_full = np.clip(pr_np * scale, args.min_depth, args.max_depth)
                pr_scaled_flat = np.clip(pr_flat * scale, args.min_depth, args.max_depth)

                all_errors.append(compute_errors(gt_flat, pr_scaled_flat))

                # Visualization (per-sample, aligned with the same gt/pr/mask)
                if vis_dir and rng.random() < args.vis_prob:
                    rgb0 = (rgb[b].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

                    gt_c = np.clip(gt_np, args.min_depth, args.max_depth)

                    # Use shared range from valid GT pixels
                    vmin = np.percentile(gt_c[m_np], 2)
                    vmax = np.percentile(gt_c[m_np], 98)
                    if vmax <= vmin:
                        vmax = vmin + 1e-3

                    # ----- COLOR VIS (RGB | GT-color | PRED-color) -----
                    gt_color = colorize_depth(gt_c, vmin, vmax)
                    pr_color = colorize_depth(pr_scaled_full, vmin, vmax)

                    stacked_color = np.concatenate([rgb0, gt_color, pr_color], axis=1)
                    out0 = vis_dir / f"da3_test_{vis_idx:05d}.png"
                    cv2.imwrite(str(out0), cv2.cvtColor(stacked_color, cv2.COLOR_RGB2BGR))

                    # --- Depth-map (grayscale) ---
                    # GT: sparse (mask with GT valid pixels)
                    gt_sparse = gt_c.copy()
                    gt_sparse[~m_np] = 0.0

                    # Pred: FULL (do NOT mask with m_np)
                    pr_full = pr_scaled_full.copy()
                    pr_full[~np.isfinite(pr_full)] = 0.0  # safety

                    # Convert meters -> uint16 mm (z16-like)
                    zmax_m = float(args.max_depth)
                    gt_mm = np.clip(gt_sparse * 1000.0, 0.0, zmax_m * 1000.0).astype(np.uint16)
                    pr_mm = np.clip(pr_full   * 1000.0, 0.0, zmax_m * 1000.0).astype(np.uint16)

                    # Natural grayscale: 0..zmax -> 0..255 (no invert, no percentile scaling)
                    den = max(1.0, zmax_m * 1000.0)
                    gt_8 = np.clip((gt_mm.astype(np.float32) / den) * 255.0, 0, 255).astype(np.uint8)
                    pr_8 = np.clip((pr_mm.astype(np.float32) / den) * 255.0, 0, 255).astype(np.uint8)

                    gt_gray = np.stack([gt_8, gt_8, gt_8], axis=-1)
                    pr_gray = np.stack([pr_8, pr_8, pr_8], axis=-1)

                    stacked_gray = np.concatenate([rgb0, gt_gray, pr_gray], axis=1)
                    out1 = vis_dir / f"da3_test_{vis_idx:05d}_depth-map.png"
                    cv2.imwrite(str(out1), cv2.cvtColor(stacked_gray, cv2.COLOR_RGB2BGR))

                    if args.save_npy:
                        np.save(str(vis_dir / f"da3_test_{vis_idx:05d}_pred.npy"), pr_scaled_full.astype(np.float32))
                        np.save(str(vis_dir / f"da3_test_{vis_idx:05d}_gt.npy"), gt_c.astype(np.float32))

                    vis_idx += 1

    if not all_errors:
        print("No valid pixels found.")
        return

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = np.mean(np.array(all_errors), axis=0)

    print("\n# ---- AGCO DA3 (test) ----")
    print(
        f"abs_rel={abs_rel:.3f} sq_rel={sq_rel:.3f} rmse={rmse:.3f} rmse_log={rmse_log:.3f} "
        f"a1={a1:.3f} a2={a2:.3f} a3={a3:.3f}"
    )


if __name__ == "__main__":
    main()
