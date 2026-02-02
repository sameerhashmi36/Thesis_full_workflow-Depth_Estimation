# eval_finetuned_monodepth2_agco_roi.py
"""
Evaluation of (fine)tuned monodepth2 model on AGCO test set with DENSE ROI evaluation.

Key difference vs sparse/LiDAR eval:
- The evaluation mask is NOT based on valid_mask (LiDAR points).
- Instead, metrics are computed on ALL pixels inside the ROI, using the (smooth) GT depth map.

ROI:
- roi_frac = 1/3 means evaluate on the bottom 33% of the image (full dense region).

This script:
- loads encoder + depth decoder from a weights folder (encoder.pth, depth.pth)
- evaluates on the fixed AGCO test set (AGCO_TEST_BAG_NAMES)
- prints monodepth2-style metrics over dense ROI pixels:
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
- options:
    --use-smooth : uses smooth dataset version (recommended for dense ROI)
    --test-fraction : frame subsampling within test set
    --no-median-scaling : disable median scaling
    --vis-dir / --vis-prob : saves RGB|GT|Pred visualizations (color + gray) with ROI overlay
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


# ---------------------------
# Metrics
# ---------------------------
def compute_depth_errors(gt, pred):
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
# Torch 0.4.1 finite mask
# ---------------------------
def finite_mask_torch041(x, huge=1e10):
    """
    torch 0.4.1 has no torch.isfinite.
    Practical finite mask:
      - x == x filters NaNs
      - |x| < huge filters inf/very large blow-ups
    Returns ByteTensor {0,1}.
    """
    nan_ok = (x == x)
    mag_ok = (x.abs() < huge)
    return (nan_ok & mag_ok).byte()


# ---------------------------
# ROI + visualization helpers
# ---------------------------
def make_bottom_roi_mask_uint8(H, W, roi_frac):
    roi_frac = max(0.0, min(1.0, float(roi_frac)))
    y0 = int(round(H * (1.0 - roi_frac)))
    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:H, :] = 1
    return m


def _normalize_depth_to_8bit(depth_m, vmin, vmax, invert=False):
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin + 1e-8)
    if invert:
        norm = 1.0 - norm
    return (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)


def colorize_depth(depth_m, vmin, vmax):
    d8 = _normalize_depth_to_8bit(depth_m, vmin, vmax, invert=False)
    cm = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def gray_depth_near_white(depth_m, vmin, vmax):
    d8 = _normalize_depth_to_8bit(depth_m, vmin, vmax, invert=True)
    return cv2.cvtColor(d8, cv2.COLOR_GRAY2RGB)


def draw_mask_overlay_rgb(img_rgb_u8, mask_u8_01, color_rgb=(0, 255, 0), alpha=0.45):
    out = img_rgb_u8.copy()
    m = (mask_u8_01.astype(np.uint8) > 0)
    if not m.any():
        return out
    col = np.array(color_rgb, dtype=np.uint8)[None, None, :]
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + col.astype(np.float32) * alpha).astype(np.uint8)
    return out


def apply_roi_and_eval_overlays(panel_rgb_u8, roi_u8_01, eval_u8_01):
    # ROI overlay (blue-ish)
    out = draw_mask_overlay_rgb(panel_rgb_u8, roi_u8_01, color_rgb=(40, 80, 255), alpha=0.20)
    # evaluated pixels overlay (green)
    out = draw_mask_overlay_rgb(out, eval_u8_01, color_rgb=(0, 255, 0), alpha=0.35)
    return out


# ---------------------------
# Model loader
# ---------------------------
def load_model(weights_folder, device):
    enc_p = weights_folder / "encoder.pth"
    dep_p = weights_folder / "depth.pth"
    if not enc_p.exists() or not dep_p.exists():
        raise FileNotFoundError("Cannot find encoder.pth/depth.pth in {}".format(weights_folder))

    print("-> Loading weights from {}".format(weights_folder))
    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    enc = torch.load(str(enc_p), map_location=device)
    enc = {k: v for k, v in enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(enc)

    dec = torch.load(str(dep_p), map_location=device)
    decoder.load_state_dict(dec)

    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--weights-folder", type=str, default="./models_finetuned_on_agco/agco_mono_640x192_finetuned_smooth_train80_val80")

    p.add_argument("--use-smooth", action="store_true")
    p.add_argument("--img-width", type=int, default=640)
    p.add_argument("--img-height", type=int, default=192)

    # ROI
    p.add_argument("--roi-frac", type=float, default=1.4 / 3.0, help="bottom fraction of image to evaluate (0..1)")

    # split params
    p.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    p.add_argument("--val-bag-fraction", type=float, default=DEFAULT_VAL_BAG_FRACTION)
    p.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--no-median-scaling", action="store_true")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.use_smooth:
        from agco_dataset_smooth import AGCODepthDataset
        print("Dataset: SMOOTH (dense ROI eval)")
    else:
        from agco_dataset import AGCODepthDataset
        print("Dataset: NORMAL (ROI eval uses GT>0; may still be sparse if GT is sparse)")

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

    print("Test samples: {} (test_fraction={})".format(len(test_ds), args.test_fraction))

    encoder, decoder = load_model(Path(args.weights_folder), device)

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print("Visualizations →", vis_dir.resolve())

    rng = random.Random(42)
    all_errs = []
    all_counts = []
    vis_idx = 0

    pbar = tqdm(test_loader, desc="test_roi_dense", leave=True, ncols=110)

    with torch.no_grad():
        for rgb, depth_gt, _valid_mask_unused, meta in pbar:
            rgb = rgb.to(device)
            depth_gt = depth_gt.to(device)

            feats = encoder(rgb)
            disp = decoder(feats)[("disp", 0)]

            # Align disp to GT resolution
            if disp.shape[-2:] != depth_gt.shape[-2:]:
                disp = F.interpolate(disp, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

            # depth_pred: (B,1,H,W)
            _, depth_pred = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)

            B, _, H, W = depth_pred.shape

            # ROI masks
            roi_np = make_bottom_roi_mask_uint8(H, W, roi_frac=args.roi_frac)  # uint8 0/1 for overlays
            roi_t = torch.from_numpy(roi_np).to(device=device).byte().unsqueeze(0).unsqueeze(0)  # (1,1,H,W) Byte

            # Evaluate per-sample
            for b in range(B):
                gt_b = depth_gt[b]     # (1,H,W) or (H,W)
                pr_b = depth_pred[b]   # (1,H,W)

                if gt_b.dim() == 2:
                    gt_b = gt_b.unsqueeze(0)
                if pr_b.dim() == 2:
                    pr_b = pr_b.unsqueeze(0)

                # Dense ROI eval mask:
                # - ROI
                # - GT depth range (also removes zeros if MIN_DEPTH > 0)
                # - finite GT and finite pred
                # NOTE: no valid_mask (no sparse LiDAR gating)
                roi_b = roi_t[0]  # (1,H,W) Byte
                gt_finite = finite_mask_torch041(gt_b)
                pr_finite = finite_mask_torch041(pr_b)
                depth_range = ((gt_b > MIN_DEPTH) & (gt_b < MAX_DEPTH)).byte()

                m_b = roi_b & gt_finite & pr_finite & depth_range  # (1,H,W) Byte

                n_valid = int(m_b.sum().item())
                all_counts.append(n_valid)
                if n_valid == 0:
                    continue

                gt_flat = gt_b[m_b].cpu().numpy().astype(np.float32)
                pr_flat = pr_b[m_b].cpu().numpy().astype(np.float32)

                if not args.no_median_scaling:
                    scale = float(np.median(gt_flat) / (np.median(pr_flat) + 1e-12))
                    pr_eval = np.clip(pr_flat * scale, MIN_DEPTH, MAX_DEPTH)
                else:
                    scale = 1.0
                    pr_eval = np.clip(pr_flat, MIN_DEPTH, MAX_DEPTH)

                all_errs.append(compute_depth_errors(gt_flat, pr_eval))

            if all_errs:
                m = np.array(all_errs).mean(axis=0)
                pbar.set_postfix({"abs_rel": "{:.3f}".format(m[0]), "rmse": "{:.3f}".format(m[2])})

            # Visualization (save based on sample 0)
            if vis_dir is not None and rng.random() < float(args.vis_prob):
                rgb0 = (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

                gt0 = depth_gt[0]
                pr0 = depth_pred[0]
                if gt0.dim() == 3:
                    gt0 = gt0[0]
                if pr0.dim() == 3:
                    pr0 = pr0[0]

                gt_img = gt0.cpu().numpy().astype(np.float32)
                pr_img = pr0.cpu().numpy().astype(np.float32)

                roi0 = (roi_np > 0)
                finite_gt0 = np.isfinite(gt_img)
                finite_pr0 = np.isfinite(pr_img)
                eval0 = roi0 & finite_gt0 & finite_pr0 & (gt_img > MIN_DEPTH) & (gt_img < MAX_DEPTH)

                # median scaling on dense ROI pixels
                if (not args.no_median_scaling) and eval0.any():
                    scale0 = float(np.median(gt_img[eval0]) / (np.median(pr_img[eval0]) + 1e-12))
                else:
                    scale0 = 1.0

                gt_c = np.clip(gt_img, MIN_DEPTH, MAX_DEPTH).astype(np.float32)
                pr_c = np.clip(pr_img * scale0, MIN_DEPTH, MAX_DEPTH).astype(np.float32)

                # vmin/vmax from ROI pixels for consistent coloring
                if eval0.any():
                    vmin = float(np.percentile(gt_c[eval0], 2))
                    vmax = float(np.percentile(gt_c[eval0], 98))
                else:
                    vmin, vmax = float(MIN_DEPTH), float(MAX_DEPTH)
                if vmax <= vmin:
                    vmax = vmin + 1e-3

                gt_color = colorize_depth(gt_c, vmin, vmax)
                pr_color = colorize_depth(pr_c, vmin, vmax)
                gt_gray = gray_depth_near_white(gt_c, vmin, vmax)
                pr_gray = gray_depth_near_white(pr_c, vmin, vmax)

                H0, W0 = rgb0.shape[:2]
                if gt_color.shape[:2] != (H0, W0):
                    gt_color = cv2.resize(gt_color, (W0, H0), interpolation=cv2.INTER_NEAREST)
                    pr_color = cv2.resize(pr_color, (W0, H0), interpolation=cv2.INTER_NEAREST)
                    gt_gray = cv2.resize(gt_gray, (W0, H0), interpolation=cv2.INTER_NEAREST)
                    pr_gray = cv2.resize(pr_gray, (W0, H0), interpolation=cv2.INTER_NEAREST)

                roi_u8 = roi_np.astype(np.uint8)
                eval_u8 = eval0.astype(np.uint8)

                rgb_ov = apply_roi_and_eval_overlays(rgb0, roi_u8, eval_u8)
                gt_color_ov = apply_roi_and_eval_overlays(gt_color, roi_u8, eval_u8)
                pr_color_ov = apply_roi_and_eval_overlays(pr_color, roi_u8, eval_u8)
                gt_gray_ov = apply_roi_and_eval_overlays(gt_gray, roi_u8, eval_u8)
                pr_gray_ov = apply_roi_and_eval_overlays(pr_gray, roi_u8, eval_u8)

                # Save two files (DA3/Zoe style)
                stacked_color = np.concatenate([rgb_ov, gt_color_ov, pr_color_ov], axis=1)
                out0 = vis_dir / "mono_roi_{:05d}.png".format(vis_idx)
                cv2.imwrite(str(out0), cv2.cvtColor(stacked_color, cv2.COLOR_RGB2BGR))

                stacked_gray = np.concatenate([rgb_ov, gt_gray_ov, pr_gray_ov], axis=1)
                out1 = vis_dir / "mono_roi_{:05d}_depth-map.png".format(vis_idx)
                cv2.imwrite(str(out1), cv2.cvtColor(stacked_gray, cv2.COLOR_RGB2BGR))

                if args.save_npy:
                    np.save(str(vis_dir / "mono_roi_{:05d}_pred.npy".format(vis_idx)), pr_c.astype(np.float32))
                    np.save(str(vis_dir / "mono_roi_{:05d}_gt.npy".format(vis_idx)), gt_c.astype(np.float32))
                    np.save(str(vis_dir / "mono_roi_{:05d}_mask.npy".format(vis_idx)), eval_u8.astype(np.uint8))
                    np.save(str(vis_dir / "mono_roi_{:05d}_roi.npy".format(vis_idx)), roi_u8.astype(np.uint8))

                vis_idx += 1

    if not all_errs:
        print("No valid pixels found for dense ROI metrics.")
        return

    mean = np.array(all_errs).mean(axis=0)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean

    cnts = np.array(all_counts, dtype=np.int64)
    if cnts.size > 0:
        print("\n# ---- ROI pixel stats ----")
        print("ROI evaluated pixels (per sample): mean={:.1f}  median={:.1f}  min={}  max={}".format(
            cnts.mean(), np.median(cnts), int(cnts.min()), int(cnts.max())
        ))

    print("\n# ---- AGCO monodepth2 evaluation (test set, DENSE ROI bottom-frac) ----")
    print("roi_frac  : {:.3f}".format(float(args.roi_frac)))
    print("abs_rel   : {:.3f}".format(abs_rel))
    print("sq_rel    : {:.3f}".format(sq_rel))
    print("rmse      : {:.3f} m".format(rmse))
    print("rmse_log  : {:.3f}".format(rmse_log))
    print("a1        : {:.3f}".format(a1))
    print("a2        : {:.3f}".format(a2))
    print("a3        : {:.3f}".format(a3))
    print("# ---------------------------------------------------------------")


if __name__ == "__main__":
    main()
