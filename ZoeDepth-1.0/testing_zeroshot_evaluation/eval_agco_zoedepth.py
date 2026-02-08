"""
python ./testing_sameer/eval_agco_zoedepth.py --raw-root /path/to/dataset/raw_dataset_cpu_manual_1 \
  --vis-dir agco_eval_zoe_vis
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import re
import cv2
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

# ----------------------------------------------------
# Paths and constants
# ----------------------------------------------------

# ZoeDepth local repo root (with hubconf.py + zoedepth code)
ZOE_REPO_ROOT = (
    "/path/to/repo/ZoeDepth-1.0"
)

# AGCO raw dataset root (rectified + depth_z16)
DEFAULT_RAW_ROOT = Path(
    "/path/to/dataset/raw_dataset_cpu_manual_1"
)

# AGCO bags to evaluate
AGCO_BAG_NAMES = [
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-54_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-29-24_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-35-57_3",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-36-57_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-45-53_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-24_1",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-31-54_10",
]

MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0
MODEL_NAME = "ZoeD_N"  # hubconf entry
# ----------------------------------------------------


def choose_device():
    """
    prefer CUDA if capability >= 7.0, else CPU.
    """
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            print("CUDA available but capability check failed. Falling back to CPU.")
            return torch.device("cpu")

        if major >= 7:
            print(f"Using CUDA device with capability sm_{major}{minor}.")
            return torch.device("cuda")
        else:
            print(
                f"CUDA device has capability sm_{major}{minor}, "
                "but this PyTorch build supports only >= sm_70. "
                "Falling back to CPU."
            )
            return torch.device("cpu")

    print("CUDA not available. Using CPU.")
    return torch.device("cpu")


def compute_errors(gt, pred):
    """
    metrics as monodepth2
    gt, pred: 1D arrays of valid depth values (meters)
    """
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def colorize_depth(depth_m, vmin=None, vmax=None):
    """
    depth_m: (H,W) float32, meters. 0 or NaN = invalid.
    Returns: (H,W,3) uint8 RGB visualization.
    """
    d = depth_m.copy().astype(np.float32)
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


def build_agco_pairs(raw_root, bag_names):
    """
    Returns list of (img_path, depth_path).
    """
    raw_root = Path(raw_root)
    rect_root = raw_root / "rectified"
    depth_root = raw_root / "depth_z16"

    samples = []
    total_pairs = 0

    idx_re = re.compile(r"rectified_idx(\d+)_t")

    print("Building (image, depth) pairs for AGCO ZoeDepth eval...")
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


def load_zoe_model(device):
    if not os.path.isfile(os.path.join(ZOE_REPO_ROOT, "hubconf.py")):
        raise FileNotFoundError(
            f"hubconf.py not found in ZOE_REPO_ROOT: {ZOE_REPO_ROOT}\n"
            "Check ZOE_REPO_ROOT path."
        )

    print(
        f"Loading ZoeDepth model '{MODEL_NAME}' from local repo "
        f"'{ZOE_REPO_ROOT}'..."
    )

    # Add repo root to sys.path so that local zoedepth modules are found
    if ZOE_REPO_ROOT not in sys.path:
        sys.path.append(ZOE_REPO_ROOT)

    zoe = torch.hub.load(ZOE_REPO_ROOT, MODEL_NAME, source="local", pretrained=True)
    zoe = zoe.to(device).eval()

    # Patch missing drop_path attributes
    patched_blocks = 0
    for m in zoe.modules():
        if hasattr(m, "gamma_1") and not hasattr(m, "drop_path"):
            m.drop_path = nn.Identity()
            patched_blocks += 1
    print(f"Patched drop_path on {patched_blocks} blocks.")

    return zoe


def evaluate_agco_zoe(pairs,
                      zoe_model,
                      device,
                      vis_dir=None,
                      vis_prob=0.02,
                      min_depth=MIN_DEPTH,
                      max_depth=MAX_DEPTH):
    """
    Main evaluation loop on AGCO using ZoeDepth.

    pairs: list of (img_path, depth_path)
    """
    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    all_errors = []
    sample_idx = 0

    print(f"-> Evaluating ZoeDepth on {len(pairs)} AGCO frames...")

    for i, (img_path, depth_path) in enumerate(tqdm(pairs, ncols=80, desc="Zoe AGCO")):
        # --- Read RGB image (for inference + vis) ---
        pil_img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            depth_pred = zoe_model.infer_pil(pil_img)  # H x W in meters

        depth_pred = depth_pred.astype(np.float32)
        Hp, Wp = depth_pred.shape

        # --- Read GT depth (z16 mm) and resize to pred resolution ---
        d_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if d_mm is None:
            raise RuntimeError(f"Failed to read GT depth: {depth_path}")
        gt_m = d_mm.astype(np.float32) / 1000.0  # meters
        gt_resized = cv2.resize(gt_m, (Wp, Hp), interpolation=cv2.INTER_NEAREST)

        # Valid mask
        mask = (
            (gt_resized > min_depth)
            & (gt_resized < max_depth)
            & np.isfinite(depth_pred)
        )
        if not np.any(mask):
            continue

        # Median scaling (mono-style)
        ratio = np.median(gt_resized[mask]) / np.median(depth_pred[mask])
        depth_scaled = depth_pred * ratio

        # Clamp to evaluation range
        depth_scaled = np.clip(depth_scaled, min_depth, max_depth)

        # Metrics
        err = compute_errors(gt_resized[mask], depth_scaled[mask])
        all_errors.append(err)

        # Visualization
        if vis_dir is not None and rng.random() < vis_prob:
            # RGB for vis
            rgb_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if rgb_bgr is not None:
                rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                H0, W0, _ = rgb.shape
                gt_vis = colorize_depth(gt_resized, vmin=min_depth, vmax=max_depth)
                pred_vis = colorize_depth(depth_scaled, vmin=min_depth, vmax=max_depth)

                # resize GT and pred to match original RGB for nicer vis
                gt_vis = cv2.resize(gt_vis, (W0, H0), interpolation=cv2.INTER_NEAREST)
                pred_vis = cv2.resize(pred_vis, (W0, H0), interpolation=cv2.INTER_NEAREST)

                stacked = np.concatenate([rgb, gt_vis, pred_vis], axis=1)
                out_name = vis_dir / f"zoe_agco_sample_{sample_idx:05d}.png"
                cv2.imwrite(str(out_name), cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
                print(f"  [VIS] Saved {out_name}")
                sample_idx += 1

    if not all_errors:
        raise RuntimeError("No valid pixels found for evaluation.")

    all_errors = np.array(all_errors)
    mean_errors = all_errors.mean(0)
    return mean_errors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-root",
        type=str,
        default=str(DEFAULT_RAW_ROOT),
        help="Root of raw_dataset_cpu_manual_1 (with rectified/ and depth_z16/)",
    )
    p.add_argument(
        "--vis-dir",
        type=str,
        default="agco_eval_zoe_vis",
        help="Directory to save random RGB|GT|Pred triplets (empty to disable)",
    )
    p.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualizations",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = choose_device()
    print("Device:", device)

    pairs = build_agco_pairs(args.raw_root, AGCO_BAG_NAMES)
    if len(pairs) == 0:
        print("No (image, depth) pairs found. Check paths and bag names.")
        return

    zoe = load_zoe_model(device)

    vis_dir = None if args.no_vis or not args.vis_dir else args.vis_dir

    mean_errors = evaluate_agco_zoe(
        pairs,
        zoe_model=zoe,
        device=device,
        vis_dir=vis_dir,
        vis_prob=0.02,  # ~2% frames saved
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
    )

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors

    print("\n# ---- AGCO ZoeDepth (ZoeD_N) results ----")
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
