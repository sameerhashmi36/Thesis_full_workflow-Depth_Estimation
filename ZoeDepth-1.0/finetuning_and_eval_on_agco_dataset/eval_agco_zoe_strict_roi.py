"""
eval_agco_zoe_strict_roi.py

Purpose
-------
Evaluate finetuned ZoeDepth checkpoints on the AGCO test split, but ONLY on a
bottom fraction of the image (ROI), e.g. bottom 1/3.

Why to do this
-------------
My smooth/filled GT (or even sparse GT) can make full-image metrics look misleading.
By restricting evaluation to the bottom region, I can focus on the area where GT is
usually denser / more relevant (ground in front of the tractor).

What to compute
--------------
Monodepth2/KITTI-style metrics over ROI-valid pixels:
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

What to visualize (optional)
---------------------------
If --vis-dir is provided, save two images per selected sample:
  1) [RGB | GT(color) | Pred(color)]
  2) [RGB | GT(gray)  | Pred(gray)]   (near = white)

And draw overlays on ALL three panels:
  - ROI overlay (blue-ish)
  - evaluated pixels overlay (green)

Command example
---------------
python finetuning_and_eval_on_agco_dataset/eval_agco_zoe_strict_roi.py \
  --raw-root /path/to/dataset/raw_dataset_cpu_manual_1 \
  --model ZoeD_K \
  --base-ckpt checkpoints/ZoeD_M12_K.pt \
  --finetuned-weights models_finetuned_on_agco/ZoeD_K_train20_val20/best.pth \
  --roi-frac 0.333 \
  --vis-dir agco_eval_vis_roi --vis-prob 0.05 \
  --median-scaling
"""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from pathlib import Path
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from agco_zoe_config import DEFAULT_RAW_ROOT, MIN_DEPTH, MAX_DEPTH
# from agco_zoe_dataset import AGCOZoeDepthDataset
from agco_zoe_dataset_smooth import AGCOZoeDepthDatasetSmooth as AGCOZoeDepthDataset

from zoe_ckpt_utils import load_state_dict_strict

ZOE_H, ZOE_W = 384, 512


# ---------------------------
# Device + forward
# ---------------------------
def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def forward_zoe_metric(zoe: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = zoe(x)
    if isinstance(out, dict):
        for k in ["metric_depth", "depth", "pred", "out"]:
            if k in out:
                out = out[k]
                break
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.dim() == 3:
        out = out.unsqueeze(1)
    return out  # (B,1,H,W)


# ---------------------------
# Tensor shape helpers
# ---------------------------
def _to_b1hw(depth_gt, valid_mask):
    if depth_gt.dim() == 3:
        depth_gt = depth_gt.unsqueeze(1)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    return depth_gt, valid_mask.bool()


def resize_gt_to_pred(depth_gt, valid_mask, pred_hw):
    H, W = pred_hw
    if depth_gt.shape[-2:] != (H, W):
        depth_gt = F.interpolate(depth_gt, size=(H, W), mode="nearest")
        valid_mask = F.interpolate(valid_mask.float(), size=(H, W), mode="nearest").bool()
    return depth_gt, valid_mask


# ---------------------------
# Metrics
# ---------------------------
def compute_errors(gt, pred):
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


# ---------------------------
# Visualization utils
# ---------------------------
def _normalize_depth_to_8bit(depth_m, vmin, vmax):
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin + 1e-8)
    return (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)


def colorize_depth(depth_m, vmin, vmax):
    d8 = _normalize_depth_to_8bit(depth_m, vmin, vmax)
    cm = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def gray_depth_near_white(depth_m, vmin, vmax):
    """
    Near = white, far = black.
    """
    d8 = _normalize_depth_to_8bit(depth_m, vmin, vmax)  # near->0, far->255
    d8 = 255 - d8  # invert: near->255 (white), far->0 (black)
    return cv2.cvtColor(d8, cv2.COLOR_GRAY2RGB)


def draw_mask_overlay_rgb(img_rgb_u8: np.ndarray, mask_bool: np.ndarray, color_rgb=(0, 255, 0), alpha: float = 0.45):
    """
    Overlay of a boolean mask onto a uint8 RGB image.
    """
    out = img_rgb_u8.copy()
    m = mask_bool.astype(bool)
    if not m.any():
        return out
    col = np.array(color_rgb, dtype=np.uint8)[None, None, :]
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + col.astype(np.float32) * alpha).astype(np.uint8)
    return out


def apply_roi_and_eval_overlays(panel_rgb_u8: np.ndarray, roi_bool: np.ndarray, eval_bool: np.ndarray) -> np.ndarray:
    """
    Appling overlays to ANY panel (RGB, GT color, Pred color, GT gray, Pred gray):
      - ROI in blue-ish
      - evaluated pixels in green
    """
    out = panel_rgb_u8
    out = draw_mask_overlay_rgb(out, roi_bool, color_rgb=(40, 80, 255), alpha=0.20)  # ROI
    out = draw_mask_overlay_rgb(out, eval_bool, color_rgb=(0, 255, 0), alpha=0.45)   # eval pixels
    return out


# ---------------------------
# ROI helper (bottom fraction)
# ---------------------------
def make_bottom_roi_mask(H: int, W: int, roi_frac: float) -> np.ndarray:
    roi_frac = max(0.0, min(1.0, float(roi_frac)))
    y0 = int(round(H * (1.0 - roi_frac)))
    m = np.zeros((H, W), dtype=bool)
    m[y0:H, :] = True
    return m


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))

    p.add_argument("--model", type=str, default="ZoeD_K", choices=["ZoeD_K", "ZoeD_N", "ZoeD_NK"])
    p.add_argument("--base-ckpt", type=str, default="checkpoints/ZoeD_M12_K.pt")
    p.add_argument("--finetuned-weights", type=str, required=True)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--median-scaling", action="store_true")

    # ROI
    p.add_argument("--roi-frac", type=float, default=1.4 / 3.0, help="bottom fraction of image to evaluate (0..1)")

    # Visualization
    p.add_argument("--vis-dir", type=str, default="")
    p.add_argument("--vis-prob", type=float, default=0.02)
    p.add_argument("--save-npy", action="store_true")

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # Local ZoeDepth repo path
    repo_root = Path("path/to/repo/root/ZoeDepth-1.0")
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    device = choose_device()
    print("Device:", device)

    # Dataset
    test_ds = AGCOZoeDepthDataset(
        raw_root=args.raw_root,
        split="test",
        test_fraction=1.0,
        img_width=ZOE_W,
        img_height=ZOE_H,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Test samples: {len(test_ds)}")

    # Build model and load base strictly (so architecture mismatch is caught)
    zoe = torch.hub.load(str(repo_root), args.model, source="local", pretrained=False)
    base_ckpt_path = args.base_ckpt if Path(args.base_ckpt).is_absolute() else str(repo_root / args.base_ckpt)
    load_state_dict_strict(zoe, base_ckpt_path, device=device)

    # Load finetuned weights strictly
    ft = Path(args.finetuned_weights)
    if not ft.is_file():
        raise FileNotFoundError(f"Finetuned weights not found: {ft}")
    state = torch.load(str(ft), map_location=device)
    zoe.load_state_dict(state, strict=True)
    zoe.to(device).eval()
    print("[FT] strict=True load OK ✅")

    # Vis output
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print("Saving visualizations to:", vis_dir)

    rng = random.Random(42)
    all_errors = []
    all_counts = []
    vis_idx = 0

    with torch.no_grad():
        for rgb, depth_gt, valid_mask, meta in tqdm(test_loader, desc="[Test ROI]", ncols=110):
            rgb = rgb.to(device, non_blocking=True)  # (B,3,H,W)
            depth_gt = depth_gt.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            depth_gt, valid_mask = _to_b1hw(depth_gt, valid_mask)

            # Predict
            depth_pred = forward_zoe_metric(zoe, rgb)  # (B,1,Hp,Wp)
            Hp, Wp = depth_pred.shape[-2:]

            # Align GT/mask to pred resolution
            depth_gt, valid_mask = resize_gt_to_pred(depth_gt, valid_mask, (Hp, Wp))

            # ROI mask in pred resolution
            roi = make_bottom_roi_mask(Hp, Wp, roi_frac=args.roi_frac)
            roi_t = torch.from_numpy(roi).to(device=device).bool().unsqueeze(0).unsqueeze(0)  # (1,1,Hp,Wp)

            # Evaluate per-sample
            B = rgb.shape[0]
            for b in range(B):
                gt_b = depth_gt[b, 0]   # (Hp,Wp)
                pr_b = depth_pred[b, 0] # (Hp,Wp)
                vm_b = valid_mask[b, 0] # (Hp,Wp)

                # Full eval mask: ROI + valid + depth-range + finite pred
                m_b = vm_b & roi_t[0, 0] & (gt_b > MIN_DEPTH) & (gt_b < MAX_DEPTH) & torch.isfinite(pr_b)

                n_valid = int(m_b.sum().item())
                all_counts.append(n_valid)
                if n_valid == 0:
                    continue

                gt_np = gt_b.detach().cpu().numpy().astype(np.float32)
                pr_np = pr_b.detach().cpu().numpy().astype(np.float32)
                m_np = m_b.detach().cpu().numpy().astype(bool)

                gt_flat = gt_np[m_np]
                pr_flat = pr_np[m_np]

                # Optional median scaling (computed on ROI pixels only)
                if args.median_scaling:
                    scale = float(np.median(gt_flat) / (np.median(pr_flat) + 1e-8))
                else:
                    scale = 1.0

                pr_scaled_full = np.clip(pr_np * scale, MIN_DEPTH, MAX_DEPTH)
                pr_scaled_flat = np.clip(pr_flat * scale, MIN_DEPTH, MAX_DEPTH)

                all_errors.append(compute_errors(gt_flat, pr_scaled_flat))

                # -------------------- Visualization --------------------
                if vis_dir and rng.random() < args.vis_prob:
                    # RGB to uint8 at pred resolution
                    rgb0 = (rgb[b].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                    if rgb0.shape[:2] != (Hp, Wp):
                        rgb0 = cv2.resize(rgb0, (Wp, Hp), interpolation=cv2.INTER_AREA)

                    gt_c = np.clip(gt_np, MIN_DEPTH, MAX_DEPTH).astype(np.float32)

                    # Shared range from valid GT pixels in ROI mask
                    vmin = float(np.percentile(gt_c[m_np], 2))
                    vmax = float(np.percentile(gt_c[m_np], 98))
                    if vmax <= vmin:
                        vmax = vmin + 1e-3

                    # 1) COLOR VIS: [RGB | GT_color | Pred_color]
                    gt_color = colorize_depth(gt_c, vmin, vmax)
                    pr_color = colorize_depth(pr_scaled_full, vmin, vmax)

                    # Apply overlays to ALL panels
                    rgb_ov = apply_roi_and_eval_overlays(rgb0, roi, m_np)
                    gt_color_ov = apply_roi_and_eval_overlays(gt_color, roi, m_np)
                    pr_color_ov = apply_roi_and_eval_overlays(pr_color, roi, m_np)

                    stacked = np.concatenate([rgb_ov, gt_color_ov, pr_color_ov], axis=1)
                    out = vis_dir / f"zoe_roi_{vis_idx:05d}.png"
                    cv2.imwrite(str(out), cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

                    # 2) GRAY DEPTH-MAP VIS: [RGB | GT_gray | Pred_gray]
                    gt_gray = gray_depth_near_white(gt_c, vmin, vmax)
                    pr_gray = gray_depth_near_white(pr_scaled_full, vmin, vmax)

                    gt_gray_ov = apply_roi_and_eval_overlays(gt_gray, roi, m_np)
                    pr_gray_ov = apply_roi_and_eval_overlays(pr_gray, roi, m_np)

                    stacked_depth = np.concatenate([rgb_ov, gt_gray_ov, pr_gray_ov], axis=1)
                    out_depth = vis_dir / f"zoe_roi_{vis_idx:05d}_depth-map.png"
                    cv2.imwrite(str(out_depth), cv2.cvtColor(stacked_depth, cv2.COLOR_RGB2BGR))

                    if args.save_npy:
                        np.save(str(vis_dir / f"zoe_roi_{vis_idx:05d}_pred.npy"), pr_scaled_full.astype(np.float32))
                        np.save(str(vis_dir / f"zoe_roi_{vis_idx:05d}_gt.npy"), gt_c.astype(np.float32))
                        np.save(str(vis_dir / f"zoe_roi_{vis_idx:05d}_mask.npy"), m_np.astype(np.uint8))
                        np.save(str(vis_dir / f"zoe_roi_{vis_idx:05d}_roi.npy"), roi.astype(np.uint8))

                    vis_idx += 1

    if not all_errors:
        print("No valid pixels found in ROI.")
        return

    errs = np.array(all_errors, dtype=np.float64)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = errs.mean(axis=0)

    cnts = np.array(all_counts, dtype=np.int64)
    if cnts.size > 0:
        print("\n# ---- ROI pixel stats ----")
        print(
            f"ROI valid pixels (per sample): mean={cnts.mean():.1f}  median={np.median(cnts):.1f}  "
            f"min={cnts.min()}  max={cnts.max()}"
        )

    print("\n# ---- AGCO ZoeDepth (test, ROI bottom-frac) ----")
    print(f"roi_frac={float(args.roi_frac):.3f} depth_range=[{MIN_DEPTH},{MAX_DEPTH}]")
    print(
        f"abs_rel={abs_rel:.3f} sq_rel={sq_rel:.3f} rmse={rmse:.3f} rmse_log={rmse_log:.3f} "
        f"a1={a1:.3f} a2={a2:.3f} a3={a3:.3f}"
    )


if __name__ == "__main__":
    main()
