"""
eval_agco_zoe_strict.py

Purpose
-------
Evaluation of finetuned ZoeDepth checkpoints on the AGCO test split with optional visualization.

What it does
------------
- Loads a ZoeDepth model (ZoeD_K / ZoeD_N / ZoeD_NK) from the local ZoeDepth repository via torch.hub.
- Loads the pretrained base checkpoint STRICTLY to ensure the model matches the intended architecture.
- Loads the finetuned weights (state_dict) with strict=True.
- Evaluates on the AGCO test split using Monodepth2-style metrics over valid LiDAR pixels:
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
- Optionally saves side-by-side visualizations.

Data assumptions
----------------
- Dataset root contains:
    raw_root/
      rectified/<bag_name>/rectified_idx*_t*.png
      depth_z16/<bag_name>/depth_idx******_t*.png
- Depth images are uint16 values in millimeters; converted to meters in the dataset.
- Test bags are fixed in agco_zoe_config.py.

Resolution handling
-------------------
- Inputs are provided at ZoeDepth input resolution (default 384x512).
- Ground-truth depth and masks are resized to prediction resolution using nearest-neighbor interpolation.

Checkpoint usage
----------------
- Base checkpoint: a ZoeDepth pretrained checkpoint (e.g., checkpoints/ZoeD_M12_K.pt).
- Finetuned weights: a state_dict saved by the finetuning script, e.g.:
    models_finetuned_on_agco/<MODEL>_trainXX_valYY/best.pth
  Debug snapshots saved by the trainer (best_eNNN_absrel*.pth) are also valid.

Visualization
-------------
- If --vis-dir is provided, random samples are saved with probability --vis-prob.
- Visualization layout (example):
    [RGB | GT(depth) color | Pred(depth) color]
- Coloring uses a shared (vmin, vmax) derived from valid GT pixels for consistent comparison.

Command examples
----------------
Evaluate best checkpoint:
  python finetuning_and_eval_on_agco_dataset/eval_agco_zoe_strict.py \
    --raw-root /path/to/dataset/raw_dataset_cpu_manual_1 \
    --model ZoeD_K \
    --base-ckpt checkpoints/ZoeD_M12_K.pt \
    --finetuned-weights models_finetuned_on_agco/ZoeD_K_train20_val20/best.pth \
    --vis-dir agco_eval_vis --vis-prob 0.05

Evaluate a debug snapshot:
  python finetuning_and_eval_on_agco_dataset/eval_agco_zoe_strict.py \
    --raw-root /path/to/dataset/raw_dataset_cpu_manual_1 \
    --model ZoeD_K \
    --base-ckpt checkpoints/ZoeD_M12_K.pt \
    --finetuned-weights models_finetuned_on_agco/ZoeD_K_train20_val20/best.pth \
    --vis-dir agco_eval_vis --vis-prob 0.05 --save-npy
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
from agco_zoe_dataset import AGCOZoeDepthDataset
# from agco_zoe_dataset_smooth import AGCOZoeDepthDatasetSmooth as AGCOZoeDepthDataset
from zoe_ckpt_utils import load_state_dict_strict


ZOE_H, ZOE_W = 384, 512


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
    return out


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


def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))

    p.add_argument("--model", type=str, default="ZoeD_K", choices=["ZoeD_K", "ZoeD_N", "ZoeD_NK"])
    p.add_argument("--base-ckpt", type=str, default="checkpoints/ZoeD_M12_K.pt")
    p.add_argument("--finetuned-weights", type=str, required=True)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--median-scaling", action="store_true")
    p.add_argument("--vis-dir", type=str, default="")
    p.add_argument("--vis-prob", type=float, default=0.02)
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path("/path/to/repo/root/ZoeDepth-1.0")
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    device = choose_device()
    print("Device:", device)

    test_ds = AGCOZoeDepthDataset(
        raw_root=args.raw_root,
        split="test",
        test_fraction=1.0,
        img_width=ZOE_W,
        img_height=ZOE_H,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    print(f"Test samples: {len(test_ds)}")

    # Build model and load base strictly
    zoe = torch.hub.load(str(repo_root), args.model, source="local", pretrained=False)
    base_ckpt_path = args.base_ckpt if Path(args.base_ckpt).is_absolute() else str(repo_root / args.base_ckpt)
    load_state_dict_strict(zoe, base_ckpt_path, device=device)

    # Load finetuned strictly (must match exactly)
    ft = Path(args.finetuned_weights)
    if not ft.is_file():
        raise FileNotFoundError(f"Finetuned weights not found: {ft}")
    state = torch.load(str(ft), map_location=device)
    zoe.load_state_dict(state, strict=True)
    zoe.to(device).eval()
    print("[FT] strict=True load OK")

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print("Saving visualizations to:", vis_dir)

    rng = random.Random(42)
    all_errors = []
    vis_idx = 0

    with torch.no_grad():
        for rgb, depth_gt, valid_mask, meta in tqdm(test_loader, desc="[Test]", ncols=110):
            rgb = rgb.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            depth_gt, valid_mask = _to_b1hw(depth_gt, valid_mask)
            depth_pred = forward_zoe_metric(zoe, rgb)

            Hp, Wp = depth_pred.shape[-2:]
            depth_gt, valid_mask = resize_gt_to_pred(depth_gt, valid_mask, (Hp, Wp))

            mask = valid_mask & (depth_gt > MIN_DEPTH) & (depth_gt < MAX_DEPTH)
            if mask.sum().item() == 0:
                continue

            gt = depth_gt[0, 0].cpu().numpy().astype(np.float32)
            pr = depth_pred[0, 0].cpu().numpy().astype(np.float32)
            m = mask[0, 0].cpu().numpy().astype(bool)

            gt_flat = gt[m]
            pr_flat = pr[m]

            if args.median_scaling:
                scale = np.median(gt_flat) / (np.median(pr_flat) + 1e-8)
            else:
                scale = 1.0

            pr_scaled_flat = np.clip(pr_flat * scale, MIN_DEPTH, MAX_DEPTH)
            all_errors.append(compute_errors(gt_flat, pr_scaled_flat))

            if vis_dir and rng.random() < args.vis_prob:
                rgb0 = (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                pr_scaled = np.clip(pr * scale, MIN_DEPTH, MAX_DEPTH)
                gt_c = np.clip(gt, MIN_DEPTH, MAX_DEPTH)

                vmin = np.percentile(gt_c[m], 2)
                vmax = np.percentile(gt_c[m], 98)
                if vmax <= vmin:
                    vmax = vmin + 1e-3

                # -------- 1) Existing image: RGB | GT_color | Pred_color --------
                gt_color = colorize_depth(gt_c, vmin, vmax)
                pr_color = colorize_depth(pr_scaled, vmin, vmax)

                stacked = np.concatenate([rgb0, gt_color, pr_color], axis=1)
                out = vis_dir / f"zoe_test_{vis_idx:05d}.png"
                cv2.imwrite(str(out), cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

                # -------- 2) New image: RGB | GT_gray | Pred_gray --------
                gt_gray = gray_depth_near_white(gt_c, vmin, vmax)
                pr_gray = gray_depth_near_white(pr_scaled, vmin, vmax)

                stacked_depth = np.concatenate([rgb0, gt_gray, pr_gray], axis=1)
                out_depth = vis_dir / f"zoe_test_{vis_idx:05d}_depth-map.png"
                cv2.imwrite(str(out_depth), cv2.cvtColor(stacked_depth, cv2.COLOR_RGB2BGR))

                vis_idx += 1

    if not all_errors:
        print("No valid pixels found.")
        return

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = np.mean(np.array(all_errors), axis=0)
    print("\n# ---- AGCO ZoeDepth (test) ----")
    print(f"abs_rel={abs_rel:.3f} sq_rel={sq_rel:.3f} rmse={rmse:.3f} rmse_log={rmse_log:.3f} a1={a1:.3f} a2={a2:.3f} a3={a3:.3f}")


if __name__ == "__main__":
    main()
