"""
    Drop-in compatible with AGCODA3DepthDataset (DA3):
      returns:
        rgb_tensor   : [3,H,W] float32 in [0,1]
        depth_tensor : [H,W]   float32 meters     
        valid_mask   : [H,W]   bool
        meta         : dict

    When smooth_depth=True:
      - depth_tensor is smoothed / filled (limited)
      - valid_mask is UNION mask (GT + filled)


      python finetune_eval_on_agco_dav3/agco_da3_dataset_smooth.py \
        --raw-root /path/to/dataset/raw_dataset_cpu_manual_1 \
        --split train \
        --idx 100 \
        --out-dir ./debug_vis_smooth_da3 \
        --train-fraction 0.01 \
        --vmax 25 \
        --max-fill-dist-px 12

"""


from __future__ import annotations

from pathlib import Path
import re
import random
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from agco_da3_config import (
    DEFAULT_RAW_ROOT,
    TRAIN_H,
    TRAIN_W,
    MIN_DEPTH_M,
    MAX_DEPTH_M,
    get_agco_bag_split,
)


# -------------------------
# Small visual helpers
# -------------------------
def depth_to_gray(depth_m: np.ndarray, vmax: float) -> np.ndarray:
    d = np.clip(depth_m.astype(np.float32), 0.0, max(float(vmax), 1e-6))
    return (d / max(float(vmax), 1e-6) * 255.0).astype(np.uint8)


def dilate_mask(mask_bool: np.ndarray, k: int = 5, iters: int = 2) -> np.ndarray:
    m = (mask_bool.astype(np.uint8) * 255)
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(m, kernel, iterations=int(iters))


def overlay_mask(rgb01: np.ndarray, mask_u8_255: np.ndarray, color_bgr=(0, 255, 0), alpha: float = 0.6) -> np.ndarray:
    rgb = (np.clip(rgb01, 0, 1) * 255.0).astype(np.uint8)
    out = rgb.copy()
    m = mask_u8_255 > 0
    color_rgb = np.array(color_bgr[::-1], dtype=np.uint8)  # BGR->RGB
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + color_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return out


# -------------------------
# Dataset
# -------------------------
class AGCODA3DepthDatasetSmooth(Dataset):
    

    def __init__(
        self,
        raw_root: str = str(DEFAULT_RAW_ROOT),
        split: str = "train",
        img_width: int = TRAIN_W,
        img_height: int = TRAIN_H,
        # fractions applied inside chosen split
        train_fraction: float = 1.0,
        val_fraction: float = 1.0,
        test_fraction: float = 1.0,
        # split control (bag-level)
        split_seed: int = 42,
        val_bag_fraction: float = 0.2,
        # smoothing controls
        smooth_depth: bool = True,
        smooth_max_m: float = 25.0,
        max_fill_dist_px: int = 12,
        use_bilateral: bool = True,
        min_depth_m: float = MIN_DEPTH_M,
        max_depth_m: float = MAX_DEPTH_M,
        # logging
        verbose_bags: bool = True,
        debug_print: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self.raw_root = Path(raw_root)
        self.rect_root = self.raw_root / "rectified"
        self.depth_root = self.raw_root / "depth_z16"

        self.split = str(split).lower().strip()
        self.W = int(img_width)
        self.H = int(img_height)

        self.split_seed = int(split_seed)
        self.val_bag_fraction = float(val_bag_fraction)

        self.smooth_depth = bool(smooth_depth)
        self.smooth_max_m = float(smooth_max_m)
        self.max_fill_dist_px = int(max_fill_dist_px)
        self.use_bilateral = bool(use_bilateral)

        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

        self.verbose_bags = bool(verbose_bags)
        self.debug_print = bool(debug_print)
        self.verbose_bags = bool(verbose)


        assert self.rect_root.is_dir(), f"Missing: {self.rect_root}"
        assert self.depth_root.is_dir(), f"Missing: {self.depth_root}"
        assert self.split in ("train", "val", "test"), f"split must be train/val/test, got: {split}"

        # ---- bag split (from config helper; do not touch config) ----
        train_bags, val_bags, test_bags = get_agco_bag_split(
            self.rect_root,
            val_bag_fraction=self.val_bag_fraction,
            seed=self.split_seed,
        )

        if self.split == "train":
            bag_names, frac = train_bags, float(train_fraction)
        elif self.split == "val":
            bag_names, frac = val_bags, float(val_fraction)
        else:
            bag_names, frac = test_bags, float(test_fraction)

        # build (img, depth, bag) list
        self.samples = self._build_samples(bag_names)

        # ---- fraction applied inside split only ----
        frac = max(0.0, min(1.0, frac))
        if frac < 1.0 and len(self.samples) > 0:
            n_keep = max(1, int(round(len(self.samples) * frac)))
            split_offset = {"train": 0, "val": 123, "test": 999}[self.split]
            rng = random.Random(self.split_seed + split_offset)
            rng.shuffle(self.samples)
            self.samples = self.samples[:n_keep]

        print(f"[AGCO {self.split}] samples: {len(self.samples)}")
        print(f"[AGCO {self.split}] smooth_depth={self.smooth_depth} vmax={self.smooth_max_m} max_fill_dist_px={self.max_fill_dist_px}")

    def _build_samples(self, bag_names):
        idx_re = re.compile(r"rectified_idx(\d+)_t")
        samples = []

        total = 0
        for bag in sorted(bag_names):
            img_dir = self.rect_root / bag
            dep_dir = self.depth_root / bag
            if not img_dir.exists() or not dep_dir.exists():
                continue

            img_files = sorted(img_dir.glob("rectified_idx*_t*.png"))
            bag_pairs = 0

            for img_path in img_files:
                m = idx_re.search(img_path.name)
                if not m:
                    continue
                cam_idx = int(m.group(1))

                depth_files = sorted(dep_dir.glob(f"depth_idx{cam_idx:06d}_t*.png"))
                if not depth_files:
                    continue

                samples.append((img_path, depth_files[0], bag))
                bag_pairs += 1

            total += bag_pairs
            if self.verbose_bags:
                print(f"[AGCO {self.split}] Bag {bag}: {bag_pairs} pairs")

        if self.verbose_bags:
            print(f"[AGCO {self.split}] Total pairs: {total}")

        return samples

    def _fill_depth_limited(self, depth_m: np.ndarray, gt_mask: np.ndarray):
        """
        Fill only holes close to GT pixels:
          - output depth is meters, clipped to smooth_max_m
          - fill_mask tells which pixels are synthetic (filled)
        """
        d = np.clip(depth_m.astype(np.float32), 0.0, max(self.smooth_max_m, 1e-6))
        gt_mask = gt_mask.astype(bool)

        # distance to nearest GT pixel (0 where GT exists, larger in holes)
        inv = np.where(gt_mask, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        fill_allow = dist <= float(self.max_fill_dist_px)

        fill_mask = (~gt_mask) & fill_allow

        # Telea inpaint expects uint8/float images; easiest: map depth -> uint8 0..255
        d01 = d / max(self.smooth_max_m, 1e-6)
        d8 = (d01 * 255.0).astype(np.uint8)
        inpaint_mask = (fill_mask.astype(np.uint8) * 255)

        d8_inp = cv2.inpaint(d8, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        d_fill = (d8_inp.astype(np.float32) / 255.0) * self.smooth_max_m

        # compose: keep GT, fill near holes, elsewhere 0
        out = np.zeros_like(d, dtype=np.float32)
        out[gt_mask] = d[gt_mask]
        out[fill_mask] = d_fill[fill_mask]

        # optional smoothing (keep GT exact)
        if self.use_bilateral:
            out2 = cv2.bilateralFilter(out, d=7, sigmaColor=0.08, sigmaSpace=7)
            out2[gt_mask] = d[gt_mask]
            out = out2

        out[~np.isfinite(out)] = 0.0
        out = np.clip(out, 0.0, self.smooth_max_m)
        return out.astype(np.float32), fill_mask.astype(bool)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, depth_path, bag = self.samples[idx]

        # --- RGB ---
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
        rgb01 = rgb.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb01).permute(2, 0, 1).float()  # [3,H,W]

        # --- Depth GT (uint16 mm -> meters) ---
        d_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d_mm is None:
            raise RuntimeError(f"Failed to read depth: {depth_path}")

        d_m = d_mm.astype(np.float32) / 1000.0
        d_m = cv2.resize(d_m, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # GT validity (raw)
        gt_mask = (d_m > 0.0) & np.isfinite(d_m)

        if self.smooth_depth:
            depth_out, fill_mask = self._fill_depth_limited(d_m, gt_mask)
            valid_mask_np = gt_mask | fill_mask  # ✅ union (GT + filled)
        else:
            depth_out = np.clip(d_m, 0.0, self.smooth_max_m).astype(np.float32)
            fill_mask = np.zeros_like(gt_mask, dtype=bool)
            valid_mask_np = gt_mask

        # Also keep training/eval depth-range idea available:
        # The train/eval scripts already apply (gt > min_depth) & (gt < max_depth) masks.
        depth_out[~np.isfinite(depth_out)] = 0.0

        depth_tensor = torch.from_numpy(depth_out).float()                 # [H,W]  (DA3-compatible)
        valid_mask = torch.from_numpy(valid_mask_np).bool()                # [H,W]

        if self.debug_print and idx == 0:
            print("\n[DEBUG idx=0]")
            print("bag:", bag)
            print("img:", img_path.name)
            print("dep:", Path(depth_path).name)
            print("GT pixels    :", int(gt_mask.sum()))
            print("Filled pixels:", int(fill_mask.sum()))
            print("Valid pixels :", int(valid_mask_np.sum()))
            print("smooth_depth :", self.smooth_depth)
            print("NOTE: If filled pixels are huge, reduce max_fill_dist_px.\n")

        meta = {
            "bag": bag,
            "img_path": str(img_path),
            "depth_path": str(depth_path),
            "gt_valid_count": int(gt_mask.sum()),
            "filled_count": int(fill_mask.sum()),
            "smooth_depth": bool(self.smooth_depth),
            "max_fill_dist_px": int(self.max_fill_dist_px),
            "smooth_max_m": float(self.smooth_max_m),
        }

        return rgb_tensor, depth_tensor, valid_mask, meta


# -------------------------
# Main visualization
# -------------------------
def visualize_one(ds, idx, out_dir, vmax=25.0, dilate_k=7, dilate_iters=2):
    rgb, depth_sm, valid_mask, meta = ds[idx]

    rgb01 = rgb.permute(1, 2, 0).numpy()
    d_sm = depth_sm.numpy()
    m_valid = valid_mask.numpy().astype(bool)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sm_gray = depth_to_gray(d_sm, vmax)
    valid_big = dilate_mask(m_valid, k=dilate_k, iters=dilate_iters)
    ov_valid = overlay_mask(rgb01, valid_big, color_bgr=(0, 255, 0), alpha=0.50)

    # cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # cv2.imwrite(str(out_dir / "sm_depth.png"), sm_gray)
    # cv2.imwrite(str(out_dir / "valid_mask_big.png"), valid_big)
    # cv2.imwrite(str(out_dir / "overlay_valid.png"), cv2.cvtColor(ov_valid, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1); plt.title("RGB"); plt.imshow(rgb01); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Smooth depth (Z16-like)"); plt.imshow(sm_gray, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Valid mask (overlay)"); plt.imshow(ov_valid); plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("[saved debug]", out_dir.resolve())
    print("meta:", meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--idx", type=int, default=100)

    # DA3 defaults
    ap.add_argument("--W", type=int, default=int(TRAIN_W))
    ap.add_argument("--H", type=int, default=int(TRAIN_H))

    ap.add_argument("--out-dir", type=str, default="./debug_vis_smooth_da3")
    ap.add_argument("--vmax", type=float, default=25.0)
    ap.add_argument("--max-fill-dist-px", type=int, default=12)
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-bilateral", action="store_true")

    ap.add_argument("--train-fraction", type=float, default=0.01)
    ap.add_argument("--val-fraction", type=float, default=1.0)
    ap.add_argument("--test-fraction", type=float, default=1.0)
    ap.add_argument("--val-bag-fraction", type=float, default=0.2)

    ap.add_argument("--debug-print", action="store_true")
    ap.add_argument("--quiet-bags", action="store_true")
    args = ap.parse_args()

    ds = AGCODA3DepthDatasetSmooth(
        raw_root=args.raw_root,
        split=args.split,
        img_width=args.W,
        img_height=args.H,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=42,
        val_bag_fraction=args.val_bag_fraction,
        smooth_depth=(not args.no_smooth),
        smooth_max_m=args.vmax,
        max_fill_dist_px=args.max_fill_dist_px,
        use_bilateral=(not args.no_bilateral),
        verbose_bags=(not args.quiet_bags),
        debug_print=args.debug_print,
    )

    if len(ds) == 0:
        raise RuntimeError("Dataset is empty")

    if args.idx < 0 or args.idx >= len(ds):
        args.idx = max(0, min(args.idx, len(ds) - 1))

    visualize_one(ds, args.idx, args.out_dir, vmax=args.vmax)
