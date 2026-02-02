"""
eval_agco_monodepth2.py

Zero-shot evaluation of monodepth2 (mono_640x192) on AGCO rectified + LiDAR depth dataset.

Usage example:

  python eval_agco_monodepth2.py \
      --raw-root /media/sameer/ran_epav_disk/Thesis/bags_from_smb/data_preparation/raw_dataset_cpu_manual_1 \
      --weights-folder ./models/mono_640x192 \
      --vis-dir agco_eval_vis

"""

import os
import re
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("/media/sameer/ran_epav_disk/Thesis/public_dataset_and_models/monodepth2")

import networks
from layers import disp_to_depth


# --------- CONFIG: default AGCO root + bag names ---------

# raw_dataset path
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


# --------- Metrics (as monodepth2) ---------

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
    d_norm[mask] = np.clip((d[mask] - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)

    cm = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)  # BGR
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    return cm


# --------- Dataset ---------

class AGCODepthDataset(Dataset):
    """
    Provides (rgb_tensor, gt_depth_tensor, valid_mask) per frame.

    - RGB comes from rectified PNGs, resized to (640,192), normalized to [0,1].
    - Depth comes from z16 PNG (uint16 mm), converted to meters and resized to (640,192).
    """

    def __init__(self, raw_root, bag_names, img_w=640, img_h=192):
        self.raw_root = Path(raw_root)
        self.rect_root = self.raw_root / "rectified"
        self.depth_root = self.raw_root / "depth_z16"
        self.img_w = int(img_w)
        self.img_h = int(img_h)

        self.samples = []  # list of (img_path, depth_path)
        self._build_pairs(bag_names)

    def _build_pairs(self, bag_names):
        print("Building (image, depth) pairs...")
        total_pairs = 0

        idx_re = re.compile(r"rectified_idx(\d+)_t")

        for bag in bag_names:
            img_dir = self.rect_root / bag
            depth_dir = self.depth_root / bag

            if not img_dir.exists():
                print(f"  [WARN] rectified folder missing: {img_dir}")
                continue
            if not depth_dir.exists():
                print(f"  [WARN] depth_z16 folder missing: {depth_dir}")
                continue

            img_files = sorted(img_dir.glob("rectified_idx*_t*.png"))

            bag_pairs = 0
            for img_path in img_files:
                m = idx_re.search(img_path.name)
                if not m:
                    continue
                cam_idx = int(m.group(1))
                # depth files are zero-padded to 6 digits
                depth_glob = depth_dir.glob(f"depth_idx{cam_idx:06d}_t*.png")
                depth_candidates = sorted(depth_glob)
                if not depth_candidates:
                    continue
                depth_path = depth_candidates[0]
                self.samples.append((img_path, depth_path))
                bag_pairs += 1

            print(f"  [Bag] {bag}: {bag_pairs} pairs")
            total_pairs += bag_pairs

        print(f"Total pairs: {total_pairs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path = self.samples[idx]

        # --- RGB ---
        rgb_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_f).permute(2, 0, 1)  # [3,H,W]

        # --- Depth (z16, mm) -> meters ---
        d_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)  # uint16
        if d_mm is None:
            raise RuntimeError(f"Failed to read depth: {depth_path}")
        d_m = d_mm.astype(np.float32) / 1000.0  # meters
        d_m = cv2.resize(d_m, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        depth_tensor = torch.from_numpy(d_m)  # [H,W], float32

        valid_mask = depth_tensor > 0.0

        return rgb_tensor, depth_tensor, valid_mask, str(img_path), str(depth_path)


# --------- Model loading ---------

def load_monodepth2_model(weights_folder, device):
    """
    Load encoder + depth decoder with mono_640x192 weights.
    """
    weights_folder = Path(weights_folder)
    encoder_path = weights_folder / "encoder.pth"
    depth_path = weights_folder / "depth.pth"

    if not encoder_path.exists() or not depth_path.exists():
        raise FileNotFoundError(f"Cannot find encoder.pth/depth.pth in {weights_folder}")

    print(f"-> Loading weights from {weights_folder}")

    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    encoder_dict = torch.load(encoder_path, map_location=device)
    # Backwards compatibility for older checkpoints:
    filtered_dict = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict)

    depth_dict = torch.load(depth_path, map_location=device)
    depth_decoder.load_state_dict(depth_dict)

    encoder.to(device)
    depth_decoder.to(device)

    encoder.eval()
    depth_decoder.eval()

    return encoder, depth_decoder


# --------- Evaluation loop ---------

def evaluate(model_enc, model_dec, dataloader, device,
             vis_dir=None, vis_prob=0.01,
             min_depth=1e-3, max_depth=80.0):
    """
    Runs evaluation and returns average metrics.
    Optionally saves random visualizations to vis_dir.
    """
    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_errors = []
    sample_idx = 0
    rng = random.Random(42)  # deterministic

    for batch_idx, (rgb, gt_depth, valid_mask, img_path, depth_path) in enumerate(dataloader):
        rgb = rgb.to(device)                      # [B,3,H,W]
        gt_depth = gt_depth.to(device)            # [B,H,W]
        valid_mask = valid_mask.to(device)        # [B,H,W]

        with torch.no_grad():
            # Monodepth2: encoder -> decoder -> disp
            feats = model_enc(rgb)
            outputs = model_dec(feats)
            disp = outputs[("disp", 0)]           # [B,1,H,W]
            # Resize disp to match GT size (here already same, but keep it explicit)
            _, _, h, w = disp.shape
            gt_h, gt_w = gt_depth.shape[-2:]
            if (h != gt_h) or (w != gt_w):
                disp = torch.nn.functional.interpolate(
                    disp, size=(gt_h, gt_w),
                    mode="bilinear", align_corners=False
                )

            _, pred_depth = disp_to_depth(disp, min_depth, max_depth)  # meters
            pred_depth = pred_depth[:, 0, :, :]  # [B,H,W]

        # Move to numpy
        gt_np = gt_depth.cpu().numpy()
        pred_np = pred_depth.cpu().numpy()
        valid_np = valid_mask.cpu().numpy().astype(bool)

        for b in range(gt_np.shape[0]):  # batch loop (here B=1)
            gt = gt_np[b]
            pred = pred_np[b]
            mask = valid_np[b]

            # Apply depth range mask
            mask = mask & (gt > min_depth) & (gt < max_depth)

            if not np.any(mask):
                continue

            # Median scaling (like monodepth2 KITTI eval)
            scale = np.median(gt[mask]) / np.median(pred[mask])
            pred_scaled = np.clip(pred * scale, min_depth, max_depth)

            # Compute errors on valid pixels
            err = compute_errors(gt[mask], pred_scaled[mask])
            all_errors.append(err)

            # Visualization
            if vis_dir is not None and rng.random() < vis_prob:
                # --- RGB back to uint8 ---
                rgb_img = (rgb[b].cpu().numpy().transpose(1, 2, 0) * 255.0)
                rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

                gt_vis = colorize_depth(gt, vmin=min_depth, vmax=max_depth)
                pred_vis = colorize_depth(pred_scaled, vmin=min_depth, vmax=max_depth)

                # Ensure all same size
                H, W, _ = rgb_img.shape
                gt_vis = cv2.resize(gt_vis, (W, H), interpolation=cv2.INTER_NEAREST)
                pred_vis = cv2.resize(pred_vis, (W, H), interpolation=cv2.INTER_NEAREST)

                stacked = np.concatenate([rgb_img, gt_vis, pred_vis], axis=1)  # [H, 3W, 3]

                out_name = vis_dir / f"mono640x192_sample_{sample_idx:05d}.png"
                # cv2 expects BGR
                cv2.imwrite(str(out_name), cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
                print(f"  [VIS] Saved {out_name}")
                sample_idx += 1

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
    p.add_argument("--weights-folder", type=str, default="/media/sameer/ran_epav_disk/Thesis/public_dataset_and_models/monodepth2/models/mono_640x192",
                   help="Folder with encoder.pth and depth.pth")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--vis-dir", type=str, default="agco_eval_vis",
                   help="Directory for saving random visualizations (set empty to disable)")
    p.add_argument("--no-vis", action="store_true", help="Disable saving visualizations")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset & loader
    dataset = AGCODepthDataset(
        raw_root=args.raw_root,
        bag_names=AGCO_BAG_NAMES,
        img_w=640, img_h=192
    )

    if len(dataset) == 0:
        print(" No samples found. Check paths and bag names.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Model
    encoder, depth_decoder = load_monodepth2_model(args.weights_folder, device)

    # Vis dir
    vis_dir = None if args.no_vis or not args.vis_dir else args.vis_dir

    # Eval
    print("-> Starting evaluation on AGCO dataset...")
    mean_errors = evaluate(
        encoder, depth_decoder, dataloader, device,
        vis_dir=vis_dir, vis_prob=0.02,  # ~2% of frames saved
        min_depth=1e-3, max_depth=80.0
    )

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors

    print("\n# ---- AGCO Monodepth2 (mono_640x192) results ----")
    print(f"  abs_rel  : {abs_rel:.3f}")
    print(f"  sq_rel   : {sq_rel:.3f}")
    print(f"  rmse     : {rmse:.3f} m")
    print(f"  rmse_log : {rmse_log:.3f}")
    print(f"  a1       : {a1:.3f}")
    print(f"  a2       : {a2:.3f}")
    print(f"  a3       : {a3:.3f}")
    print("# -----------------------------------------------")


if __name__ == "__main__":
    main()


# # ---- AGCO Monodepth2 (mono_640x192) results ----
#   abs_rel  : 0.568
#   sq_rel   : 1.783
#   rmse     : 3.180 m
#   rmse_log : 0.570
#   a1       : 0.462
#   a2       : 0.667
#   a3       : 0.773
# # -----------------------------------------------