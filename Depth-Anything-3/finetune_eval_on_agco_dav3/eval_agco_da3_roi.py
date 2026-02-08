# eval_agco_da3_roi.py
"""
ROI-based evaluation for finetuned Depth-Anything-3 (DA3) on AGCO test split
with visualization matching eval_agco_da3_finetuned.py.

Outputs (when --vis-dir enabled):
  1) Color visualization:   [RGB | GT(color) | Pred(color)]
  2) Gray "depth-map" vis:  [RGB | GT(gray)  | Pred(gray)]
And NOW: ROI + eval-mask overlay is applied to ALL THREE PANELS.

Key eval modes:
- default (no --gt-only): uses dataset depth + valid_mask (smooth/filled if using smooth dataset)
- --gt-only: ignores dataset depth/mask for metrics; reloads RAW depth_z16 and uses sparse mask (depth>0)

Example:
python eval_agco_da3_roi.py \
  --raw-root /path/to/dataset/raw_dataset_cpu_manual_1 \
  --da3-repo-root /path/to/repo/root/Depth-Anything-3 \
  --model-id depth-anything/DA3-LARGE \
  --finetuned-weights ./models_finetuned_on_agco_smooth/<run>/best.pth \
  --roi-frac 0.333 \
  --gt-only \
  --vis-dir ./vis_roi \
  --vis-prob 0.03
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

# smooth dataset drop-in
from agco_da3_dataset_smooth import AGCODA3DepthDatasetSmooth as AGCODA3DepthDataset


# ----------------- device -----------------
def choose_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ----------------- pad/crop helpers (match training) -----------------
def pad_to_multiple_bschw(x: torch.Tensor, mult: int = 14):
    """
    Pad to make H and W divisible by mult.
    Accepts:
      - (B,C,H,W)
      - (B,S,C,H,W)
    Returns:
      x_pad, orig_hw
    """
    if x.dim() == 4:
        _, _, H, W = x.shape
        pad_h = (mult - (H % mult)) % mult
        pad_w = (mult - (W % mult)) % mult
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x_pad, (H, W)

    if x.dim() == 5:
        B, S, C, H, W = x.shape
        pad_h = (mult - (H % mult)) % mult
        pad_w = (mult - (W % mult)) % mult
        xs = x.reshape(B * S, C, H, W)
        xs_pad = F.pad(xs, (0, pad_w, 0, pad_h), mode="replicate")
        Hp, Wp = xs_pad.shape[-2], xs_pad.shape[-1]
        x_pad = xs_pad.reshape(B, S, C, Hp, Wp)
        return x_pad, (H, W)

    raise ValueError(f"Unexpected input rank: {x.dim()} (expected 4D or 5D)")


def crop_back(pred: torch.Tensor, orig_hw):
    H, W = orig_hw
    return pred[..., :H, :W]


def forward_da3_metric(net: torch.nn.Module, x_bchw: torch.Tensor, patch_mult: int = 14) -> torch.Tensor:
    """
    Forward wrapper:
    - (B,C,H,W) -> (B,1,C,H,W)
    - pad to patch multiple
    - net(x, None, None, (), False)
    - normalize output to (B,1,H,W)
    - crop back to original
    """
    if x_bchw.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(x_bchw.shape)}")

    x = x_bchw.unsqueeze(1)  # (B,1,C,H,W)
    x_pad, orig_hw = pad_to_multiple_bschw(x, mult=int(patch_mult))

    out = net(x_pad, None, None, (), False)

    if isinstance(out, dict):
        for k in ["metric_depth", "depth", "pred", "out"]:
            if k in out:
                out = out[k]
                break
    if isinstance(out, (list, tuple)):
        out = out[0]

    # normalize to (B,1,h,w)
    if out.dim() == 5:          # (B,S,1,h,w)
        out = out[:, 0]
    elif out.dim() == 4:
        if out.shape[1] != 1:   # (B,S,h,w)
            out = out[:, 0].unsqueeze(1)
    elif out.dim() == 3:
        out = out.unsqueeze(1)

    out = crop_back(out, orig_hw)
    return out


# ----------------- metrics -----------------
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


# ----------------- vis helpers -----------------
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


def draw_mask_overlay_rgb(img_rgb_u8: np.ndarray, mask_bool: np.ndarray, color_rgb=(0, 255, 0), alpha: float = 0.45):
    """
    Overlay a boolean mask onto a uint8 RGB image.
    color_rgb: tuple in RGB order.
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
    Apply overlays to ANY panel (RGB, GT color, Pred color, GT gray, Pred gray):
      - ROI in blue
      - evaluated pixels in green
    """
    out = panel_rgb_u8
    # ROI overlay (blue-ish)
    out = draw_mask_overlay_rgb(out, roi_bool, color_rgb=(40, 80, 255), alpha=0.20)
    # Evaluated pixels overlay (green)
    out = draw_mask_overlay_rgb(out, eval_bool, color_rgb=(0, 255, 0), alpha=0.45)
    return out


def make_bottom_roi_mask(H: int, W: int, roi_frac: float) -> np.ndarray:
    roi_frac = max(0.0, min(1.0, float(roi_frac)))
    y0 = int(round(H * (1.0 - roi_frac)))
    m = np.zeros((H, W), dtype=bool)
    m[y0:H, :] = True
    return m


def load_raw_depth_from_path(depth_path: str, out_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Read uint16 depth_z16 in mm -> meters, resize nearest.
    Returns:
      depth_m (H,W) float32
      gt_mask (H,W) bool  (depth>0 & finite)
    """
    d_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d_mm is None:
        raise RuntimeError(f"Failed to read depth: {depth_path}")

    d_m = d_mm.astype(np.float32) / 1000.0
    H, W = out_hw
    d_m = cv2.resize(d_m, (W, H), interpolation=cv2.INTER_NEAREST)

    gt_mask = (d_m > 0.0) & np.isfinite(d_m)
    d_m[~np.isfinite(d_m)] = 0.0
    return d_m.astype(np.float32), gt_mask.astype(bool)


# ----------------- args -----------------
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

    # ROI controls
    p.add_argument("--roi-frac", type=float, default=3.0 / 3.0, help="fraction of image from bottom, e.g. 0.333")
    p.add_argument("--gt-only", action="store_true", help="Use RAW sparse depth_z16 + raw GT mask (ignore filled).")

    # scaling and DA3 patch multiple
    p.add_argument("--median-scaling", action="store_true")
    p.add_argument("--patch-mult", type=int, default=14)

    # visualization
    p.add_argument("--vis-dir", type=str, default="")
    p.add_argument("--vis-prob", type=float, default=0.02)
    p.add_argument("--save-npy", action="store_true")

    return p.parse_args()


# ----------------- main -----------------
def main():
    args = parse_args()

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
    all_counts = []
    vis_idx = 0

    with torch.no_grad():
        for rgb, depth_gt, valid_mask, meta in tqdm(test_loader, desc="[Test ROI]", ncols=110):
            rgb = rgb.to(device, non_blocking=True)  # (B,3,H,W)

            # predict (B,1,H,W) after pad/crop
            pred = forward_da3_metric(net, rgb, patch_mult=args.patch_mult)
            Hp, Wp = pred.shape[-2:]

            B = rgb.shape[0]

            # Collated meta is dict of lists for string fields
            depth_paths = meta.get("depth_path", None)
            img_paths = meta.get("img_path", None)

            for b in range(B):
                roi = make_bottom_roi_mask(Hp, Wp, roi_frac=args.roi_frac)

                # ---------------- choose GT source for eval ----------------
                if args.gt_only:
                    if depth_paths is None:
                        raise RuntimeError("meta['depth_path'] not found; cannot run --gt-only")
                    dp = depth_paths[b]
                    gt_use, gt_valid = load_raw_depth_from_path(dp, out_hw=(Hp, Wp))
                    base_valid = gt_valid
                else:
                    gt_t = depth_gt[b]
                    vm_t = valid_mask[b].bool()

                    if gt_t.dim() == 2:
                        gt_t = gt_t.unsqueeze(0)  # (1,H,W)
                    if vm_t.dim() == 2:
                        vm_t = vm_t.unsqueeze(0)

                    # align to pred size if needed
                    if gt_t.shape[-2:] != (Hp, Wp):
                        gt_t = F.interpolate(gt_t.unsqueeze(0), size=(Hp, Wp), mode="nearest").squeeze(0)  # (1,Hp,Wp)
                        vm_t = F.interpolate(vm_t.float().unsqueeze(0), size=(Hp, Wp), mode="nearest").squeeze(0).bool()

                    gt_use = gt_t[0].detach().cpu().numpy().astype(np.float32)
                    base_valid = vm_t[0].detach().cpu().numpy().astype(bool)

                pr_np = pred[b, 0].detach().cpu().numpy().astype(np.float32)

                # full eval mask (ROI + valid + depth-range + finite pred)
                m = base_valid & roi
                m = m & (gt_use > args.min_depth) & (gt_use < args.max_depth) & np.isfinite(pr_np)

                n_valid = int(m.sum())
                all_counts.append(n_valid)
                if n_valid == 0:
                    continue

                gt_flat = gt_use[m]
                pr_flat = pr_np[m]

                # scaling (computed on ROI pixels)
                if args.median_scaling:
                    scale = float(np.median(gt_flat) / (np.median(pr_flat) + 1e-8))
                else:
                    scale = 1.0

                pr_scaled_full = np.clip(pr_np * scale, args.min_depth, args.max_depth)
                pr_scaled_flat = np.clip(pr_flat * scale, args.min_depth, args.max_depth)

                all_errors.append(compute_errors(gt_flat, pr_scaled_flat))

                # ---------------- visualization (match finetuned eval style) ----------------
                if vis_dir and rng.random() < float(args.vis_prob):
                    # RGB u8 at pred resolution
                    rgb_u8 = (rgb[b].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                    if rgb_u8.shape[:2] != (Hp, Wp):
                        rgb_u8 = cv2.resize(rgb_u8, (Wp, Hp), interpolation=cv2.INTER_AREA)

                    gt_c = np.clip(gt_use, args.min_depth, args.max_depth)
                    # vmin/vmax from valid GT pixels
                    vmin = float(np.percentile(gt_c[m], 2))
                    vmax = float(np.percentile(gt_c[m], 98))
                    if vmax <= vmin:
                        vmax = vmin + 1e-3

                    # ----- COLOR VIS: [RGB | GT-color | PRED-color] -----
                    gt_color = colorize_depth(gt_c, vmin, vmax)                 # (H,W,3)
                    pr_color = colorize_depth(pr_scaled_full, vmin, vmax)       # (H,W,3)

                    # Apply overlays to ALL panels
                    rgb_ov = apply_roi_and_eval_overlays(rgb_u8, roi, m)
                    gt_color_ov = apply_roi_and_eval_overlays(gt_color, roi, m)
                    pr_color_ov = apply_roi_and_eval_overlays(pr_color, roi, m)

                    stacked_color = np.concatenate([rgb_ov, gt_color_ov, pr_color_ov], axis=1)
                    out0 = vis_dir / f"da3_roi_{vis_idx:05d}.png"
                    cv2.imwrite(str(out0), cv2.cvtColor(stacked_color, cv2.COLOR_RGB2BGR))

                    # ----- GRAY DEPTH-MAP VIS: [RGB | GT(gray z16-like) | PRED(gray z16-like)] -----
                    # GT sparse in mask, Pred full
                    gt_sparse = gt_c.copy()
                    gt_sparse[~m] = 0.0

                    pr_full = pr_scaled_full.copy()
                    pr_full[~np.isfinite(pr_full)] = 0.0

                    zmax_m = float(args.max_depth)

                    gt_mm = np.clip(gt_sparse * 1000.0, 0.0, zmax_m * 1000.0).astype(np.uint16)
                    pr_mm = np.clip(pr_full   * 1000.0, 0.0, zmax_m * 1000.0).astype(np.uint16)

                    den = max(1.0, zmax_m * 1000.0)
                    gt_8 = np.clip((gt_mm.astype(np.float32) / den) * 255.0, 0, 255).astype(np.uint8)
                    pr_8 = np.clip((pr_mm.astype(np.float32) / den) * 255.0, 0, 255).astype(np.uint8)

                    gt_gray = np.stack([gt_8, gt_8, gt_8], axis=-1)
                    pr_gray = np.stack([pr_8, pr_8, pr_8], axis=-1)

                    # Apply overlays to ALL panels (gray too)
                    gt_gray_ov = apply_roi_and_eval_overlays(gt_gray, roi, m)
                    pr_gray_ov = apply_roi_and_eval_overlays(pr_gray, roi, m)

                    stacked_gray = np.concatenate([rgb_ov, gt_gray_ov, pr_gray_ov], axis=1)
                    out1 = vis_dir / f"da3_roi_{vis_idx:05d}_depth-map.png"
                    cv2.imwrite(str(out1), cv2.cvtColor(stacked_gray, cv2.COLOR_RGB2BGR))

                    if args.save_npy:
                        np.save(str(vis_dir / f"da3_roi_{vis_idx:05d}_pred.npy"), pr_scaled_full.astype(np.float32))
                        np.save(str(vis_dir / f"da3_roi_{vis_idx:05d}_gt.npy"), gt_c.astype(np.float32))
                        np.save(str(vis_dir / f"da3_roi_{vis_idx:05d}_mask.npy"), m.astype(np.uint8))
                        np.save(str(vis_dir / f"da3_roi_{vis_idx:05d}_roi.npy"), roi.astype(np.uint8))

                    # Also write traceability text
                    ip = img_paths[b] if img_paths is not None else ""
                    dp = depth_paths[b] if depth_paths is not None else ""
                    # with open(vis_dir / f"da3_roi_{vis_idx:05d}.txt", "w", encoding="utf-8") as f:
                    #     f.write(f"img_path: {ip}\n")
                    #     f.write(f"depth_path: {dp}\n")
                    #     f.write(f"gt_only: {bool(args.gt_only)}\n")
                    #     f.write(f"roi_frac: {float(args.roi_frac)}\n")
                    #     f.write(f"valid_pixels: {n_valid}\n")
                    #     f.write(f"median_scaling: {bool(args.median_scaling)}\n")
                    #     f.write(f"patch_mult: {int(args.patch_mult)}\n")

                    vis_idx += 1

    if not all_errors:
        print("No valid pixels found under ROI/mask/depth-range constraints.")
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

    print("\n# ---- AGCO DA3 (test, ROI bottom-frac) ----")
    print(f"gt_only={bool(args.gt_only)} roi_frac={float(args.roi_frac):.3f} depth_range=[{args.min_depth},{args.max_depth}]")
    print(
        f"abs_rel={abs_rel:.3f} sq_rel={sq_rel:.3f} rmse={rmse:.3f} rmse_log={rmse_log:.3f} "
        f"a1={a1:.3f} a2={a2:.3f} a3={a3:.3f}"
    )


if __name__ == "__main__":
    main()
