"""
eval_agco_da3.py

Zero-shot evaluation of Depth Anything 3 (DA3) on AGCO rectified + LiDAR depth dataset.


Dataset Structure:
  raw_dataset_cpu_manual_1/
    rectified/<bag_name>/rectified_idxXXX_t<time>.png
    depth_z16/<bag_name>/depth_idxXXXXXX_t<time>.png

This script:
- Builds (RGB, GT depth) pairs for selected AGCO bag folders
- Runs DA3 inference (using DepthAnything3.from_pretrained(...).inference())
- Resizes GT depth to DA3 output resolution
- Applies median scaling (like KITTI / monodepth2 eval)
- Computes abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
- Optionally saves random triplets [RGB, GT depth, Pred depth] as PNGs.

Example:

  python eval_agco_da3.py \
      --raw-root /path/to/dataset/raw_dataset_cpu_manual_1 \
      --model-id depth-anything/DA3-LARGE \
      --vis-dir agco_eval_da3_vis
"""

import os
import re
import glob
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.append("/path/to/repo/root/Depth-Anything-3")

from depth_anything_3.api import DepthAnything3  # DA3 official API


# --------- CONFIG: default AGCO root + bag names ---------

DEFAULT_RAW_ROOT = Path("raw_dataset_cpu_manual_1")

AGCO_BAG_NAMES = [
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-54_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-29-24_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-35-57_3",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-36-57_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-45-53_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-24_1",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-31-54_10",
]


# --------- Metrics (same style as monodepth2) ---------

def compute_errors(gt, pred):
    """
    gt, pred: 1D numpy arrays (valid pixels only), in meters.
    Returns:
      abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    """
    assert gt.shape == pred.shape
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# --------- Simple depth colorization ---------

def colorize_depth(depth_m, vmin=None, vmax=None):
    """
    depth_m: (H,W) float32, meters. 0 or NaN = invalid.
    Returns: (H,W,3) uint8 RGB visualization.
    """
    d = depth_m.copy()
    d[~np.isfinite(d)] = 0
    d[d < 0] = 0

    mask = d > 0
    if not mask.any():
        return np.zeros((*d.shape, 3), np.uint8)

    if vmin is None:
        vmin = np.percentile(d[mask], 2)
    if vmax is None:
        vmax = np.percentile(d[mask], 98)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    d_norm = np.zeros_like(d, np.uint8)
    d_norm[mask] = np.clip(
        (d[mask] - vmin) / (vmax - vmin) * 255.0, 0, 255
    ).astype(np.uint8)

    cm = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)  # BGR
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    return cm


# --------- Build (RGB, depth) pairs ---------

def build_agco_pairs(raw_root, bag_names):
    """
    Returns a list of (img_path, depth_path).
    """
    raw_root = Path(raw_root)
    rect_root = raw_root / "rectified"
    depth_root = raw_root / "depth_z16"

    samples = []
    total_pairs = 0
    idx_re = re.compile(r"rectified_idx(\d+)_t")

    print("Building (image, depth) pairs for DA3 eval...")
    for bag in bag_names:
        img_dir = rect_root / bag
        dep_dir = depth_root / bag

        if not img_dir.exists():
            print(f"  [WARN] rectified folder missing: {img_dir}")
            continue
        if not dep_dir.exists():
            print(f"  [WARN] depth_z16 folder missing: {dep_dir}")
            continue

        img_files = sorted(img_dir.glob("rectified_idx*_t*.png"))
        bag_pairs = 0

        for img_path in img_files:
            m = idx_re.search(img_path.name)
            if not m:
                continue
            cam_idx = int(m.group(1))
            depth_glob = dep_dir.glob(f"depth_idx{cam_idx:06d}_t*.png")
            depth_candidates = sorted(depth_glob)
            if not depth_candidates:
                continue
            depth_path = depth_candidates[0]
            samples.append((str(img_path), str(depth_path)))
            bag_pairs += 1

        print(f"  [Bag] {bag}: {bag_pairs} pairs")
        total_pairs += bag_pairs

    print(f"Total pairs: {total_pairs}")
    return samples


# --------- Load DA3 model ---------

def load_da3_model(model_id, device):
    """
    model_id can be:
      - HuggingFace repo id, e.g. "depth-anything/DA3-LARGE"
      - or a local directory with a DA3 model (as in official docs).
    """
    print(f"-> Loading DA3 model from '{model_id}' ...")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    model.eval()
    return model


# --------- Evaluation using DA3.inference(...) ---------

def evaluate_da3_agco(pairs,
                      model,
                      device,
                      batch_size=4,
                      vis_dir=None,
                      vis_prob=0.02,
                      min_depth=1e-3,
                      max_depth=80.0):
    """
    pairs: list of (img_path, depth_path)
    model: DepthAnything3 instance
    """
    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_errors = []
    rng = random.Random(42)
    sample_idx = 0

    num_pairs = len(pairs)
    print(f"-> Evaluating DA3 on {num_pairs} image/depth pairs...")

    # process in chunks because DA3.inference works on a list of paths
    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        batch_pairs = pairs[start:end]
        img_paths = [p[0] for p in batch_pairs]
        depth_paths = [p[1] for p in batch_pairs]

        # DA3 inference:
        # model.inference(list_of_paths) -> prediction with:
        #   prediction.processed_images : [N, H, W, 3] uint8
        #   prediction.depth            : [N, H, W] float32 (depth, not disparity)
        prediction = model.inference(img_paths)

        imgs_proc = prediction.processed_images  # (N, H, W, 3)
        depths_pred = prediction.depth          # (N, H, W)

        if imgs_proc is None or depths_pred is None:
            raise RuntimeError("DA3 prediction returned None for images or depth")

        N = depths_pred.shape[0]
        assert N == len(img_paths)

        for i in range(N):
            rgb_proc = imgs_proc[i]     # uint8, (Hp,Wp,3)
            pred_d = depths_pred[i]     # float32, (Hp,Wp)

            # --- read GT and resize to DA3 depth resolution ---
            d_mm = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
            if d_mm is None:
                raise RuntimeError(f"Failed to read GT depth: {depth_paths[i]}")

            gt_m = d_mm.astype(np.float32) / 1000.0  # meters
            Hp, Wp = pred_d.shape[0], pred_d.shape[1]
            gt_resized = cv2.resize(gt_m, (Wp, Hp), interpolation=cv2.INTER_NEAREST)

            # Valid pixels mask
            mask = (gt_resized > min_depth) & (gt_resized < max_depth) & np.isfinite(pred_d)

            if not np.any(mask):
                continue

            # Median scaling like monodepth2 KITTI eval
            scale = np.median(gt_resized[mask]) / np.median(pred_d[mask])
            pred_scaled = np.clip(pred_d * scale, min_depth, max_depth)

            err = compute_errors(gt_resized[mask], pred_scaled[mask])
            all_errors.append(err)

            # Visualization
            if vis_dir is not None and rng.random() < vis_prob:
                gt_vis = colorize_depth(gt_resized, vmin=min_depth, vmax=max_depth)
                pred_vis = colorize_depth(pred_scaled, vmin=min_depth, vmax=max_depth)

                # make sure the size is same
                H, W, _ = rgb_proc.shape
                gt_vis = cv2.resize(gt_vis, (W, H), interpolation=cv2.INTER_NEAREST)
                pred_vis = cv2.resize(pred_vis, (W, H), interpolation=cv2.INTER_NEAREST)

                stacked = np.concatenate([rgb_proc, gt_vis, pred_vis], axis=1)
                out_name = vis_dir / f"da3_sample_{sample_idx:05d}.png"
                cv2.imwrite(str(out_name), cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
                print(f"  [VIS] Saved {out_name}")
                sample_idx += 1

        if (start // batch_size) % 20 == 0:
            print(f"  processed {end}/{num_pairs} pairs ...")

    if not all_errors:
        raise RuntimeError("No valid pixels found for evaluation.")

    all_errors = np.array(all_errors)
    mean_errors = all_errors.mean(axis=0)
    return mean_errors


# --------- Main ---------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT),
                   help="Root of raw_dataset_cpu_manual_1 (containing rectified/ and depth_z16/)")
    p.add_argument("--model-id", type=str,
                   default="depth-anything/DA3-LARGE",
                   help="HuggingFace model id or local DA3 model directory")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of images per DA3 inference call")
    p.add_argument("--vis-dir", type=str, default="agco_eval_da3_vis",
                   help="Directory for saving random visualizations (set empty to disable)")
    p.add_argument("--no-vis", action="store_true", help="Disable saving visualizations")
    p.add_argument("--min-depth", type=float, default=1e-3)
    p.add_argument("--max-depth", type=float, default=80.0)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build pairs from AGCO bags
    pairs = build_agco_pairs(args.raw_root, AGCO_BAG_NAMES)
    if len(pairs) == 0:
        print(" No (image, depth) pairs found. Check paths and bag names.")
        return

    # Model
    da3_model = load_da3_model(args.model_id, device)

    # Vis dir
    vis_dir = None if args.no_vis or not args.vis_dir else args.vis_dir

    # Eval
    print("-> Starting DA3 evaluation on AGCO dataset...")
    mean_errors = evaluate_da3_agco(
        pairs,
        da3_model,
        device=device,
        batch_size=args.batch_size,
        vis_dir=vis_dir,
        vis_prob=0.02,  # ~2% frames saved
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors

    print("\n# ---- AGCO DA3 results (median-scaled) ----")
    print(f"  abs_rel  : {abs_rel:.3f}")
    print(f"  sq_rel   : {sq_rel:.3f}")
    print(f"  rmse     : {rmse:.3f} m")
    print(f"  rmse_log : {rmse_log:.3f}")
    print(f"  a1       : {a1:.3f}")
    print(f"  a2       : {a2:.3f}")
    print(f"  a3       : {a3:.3f}")
    print("# -----------------------------------------")


if __name__ == "__main__":
    main()



# # ---- AGCO DA3 results (median-scaled) ----
#   abs_rel  : 0.323
#   sq_rel   : 0.588
#   rmse     : 1.370 m
#   rmse_log : 0.380
#   a1       : 0.723
#   a2       : 0.795
#   a3       : 0.863
# # -----------------------------------------