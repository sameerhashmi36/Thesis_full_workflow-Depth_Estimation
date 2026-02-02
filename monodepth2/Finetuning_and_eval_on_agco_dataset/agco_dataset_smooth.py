"""
agco_dataset_smooth.py

Same splits as agco_dataset.py, but depth is filled/smoothed.

Returns:
  rgb_tensor   : [3,H,W] float32 in [0,1]
  depth_tensor : [1,H,W] float32 meters (smoothed)
  valid_mask   : [1,H,W] uint8 (0/1)  (old torch friendly)
  meta         : dict
"""

from pathlib import Path
from typing import List, Tuple
import argparse
import re
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from agco_config import (
    DEFAULT_RAW_ROOT,
    AGCO_TEST_BAG_NAMES,
    DEFAULT_VAL_BAG_FRACTION,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VAL_FRACTION,
    DEFAULT_TEST_FRACTION,
    DEFAULT_SEED,
)


def _discover_common_bags(raw_root):
    raw_root = Path(raw_root)
    rect_root = raw_root / "rectified"
    depth_root = raw_root / "depth_z16"
    if not rect_root.exists():
        raise FileNotFoundError("rectified/ not found under: {}".format(str(raw_root)))
    if not depth_root.exists():
        raise FileNotFoundError("depth_z16/ not found under: {}".format(str(raw_root)))

    rect_bags = set([p.name for p in rect_root.iterdir() if p.is_dir()])
    depth_bags = set([p.name for p in depth_root.iterdir() if p.is_dir()])
    common = sorted(list(rect_bags & depth_bags))
    if len(common) == 0:
        raise RuntimeError("No common bag folders found in rectified/ and depth_z16/")
    return common


def get_agco_bag_split(raw_root, val_bag_fraction=DEFAULT_VAL_BAG_FRACTION, seed=DEFAULT_SEED):
    all_common = _discover_common_bags(raw_root)
    test_set = set(AGCO_TEST_BAG_NAMES)
    non_test = [b for b in all_common if b not in test_set]
    if len(non_test) == 0:
        raise RuntimeError("All common bags are test bags; no train/val left.")

    rng = random.Random(int(seed))
    rng.shuffle(non_test)

    vf = float(val_bag_fraction)
    vf = max(0.0, min(1.0, vf))

    if vf <= 0:
        n_val = 0
    else:
        n_val = int(round(len(non_test) * vf))
        n_val = max(1, n_val)

    if len(non_test) > 1:
        n_val = min(n_val, len(non_test) - 1)
    else:
        n_val = 0

    val_bags = sorted(non_test[:n_val])
    train_bags = sorted(non_test[n_val:])

    print("[AGCO SPLIT] non-test bags : {}".format(len(non_test)))
    print("[AGCO SPLIT] train bags    : {}".format(len(train_bags)))
    print("[AGCO SPLIT] val bags      : {} (val_bag_fraction={})".format(len(val_bags), vf))

    return train_bags, val_bags, non_test


def _apply_frame_fraction(samples, frac, seed):
    frac = float(frac)
    frac = max(0.0, min(1.0, frac))

    n_total = len(samples)
    if n_total == 0:
        return samples

    n_keep = int(round(n_total * frac))
    n_keep = max(1, min(n_keep, n_total))

    if n_keep < n_total:
        rng = random.Random(int(seed))
        rng.shuffle(samples)
        samples = samples[:n_keep]
    return samples


def depth_to_gray(depth_m, vmax):
    d = np.clip(depth_m.astype(np.float32), 0.0, max(float(vmax), 1e-6))
    return (d / max(float(vmax), 1e-6) * 255.0).astype(np.uint8)


def dilate_mask(mask_bool, k=7, iters=2):
    m = (mask_bool.astype(np.uint8) * 255)
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(m, kernel, iterations=int(iters))


def overlay_mask(rgb01, mask_u8_255, color_bgr=(0, 255, 0), alpha=0.5):
    rgb = (np.clip(rgb01, 0, 1) * 255.0).astype(np.uint8)
    out = rgb.copy()
    m = mask_u8_255 > 0
    color_rgb = np.array(color_bgr[::-1], dtype=np.uint8)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + color_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return out


class AGCODepthDataset(Dataset):
    def __init__(
        self,
        raw_root=str(DEFAULT_RAW_ROOT),
        split="train",
        # frame-level
        train_fraction=DEFAULT_TRAIN_FRACTION,
        val_fraction=DEFAULT_VAL_FRACTION,
        test_fraction=DEFAULT_TEST_FRACTION,
        # bag-level
        val_bag_fraction=DEFAULT_VAL_BAG_FRACTION,
        seed=DEFAULT_SEED,
        img_width=640,
        img_height=192,
        # smoothing
        smooth_depth=True,
        smooth_max_m=25.0,
        max_fill_dist_px=12,
        use_bilateral=True,
        mask_mode="gt",  # "gt" or "gt+filled"
        verbose_bags=True,
        debug_print=False,
    ):
        super(AGCODepthDataset, self).__init__()
        assert split in ("train", "val", "test")
        assert mask_mode in ("gt", "gt+filled")

        self.raw_root = Path(raw_root)
        self.rect_root = self.raw_root / "rectified"
        self.depth_root = self.raw_root / "depth_z16"

        self.split = split
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)

        self.val_bag_fraction = float(val_bag_fraction)
        self.seed = int(seed)

        self.img_width = int(img_width)
        self.img_height = int(img_height)

        self.smooth_depth = bool(smooth_depth)
        self.smooth_max_m = float(smooth_max_m)
        self.max_fill_dist_px = int(max_fill_dist_px)
        self.use_bilateral = bool(use_bilateral)
        self.mask_mode = str(mask_mode)

        self.verbose_bags = bool(verbose_bags)
        self.debug_print = bool(debug_print)

        assert self.rect_root.is_dir(), "rectified/ not found at {}".format(str(self.rect_root))
        assert self.depth_root.is_dir(), "depth_z16/ not found at {}".format(str(self.depth_root))

        train_bags, val_bags, _ = get_agco_bag_split(
            self.raw_root, val_bag_fraction=self.val_bag_fraction, seed=self.seed
        )

        if self.split == "train":
            bag_names = train_bags
            frac = self.train_fraction
            frac_seed = self.seed + 0
        elif self.split == "val":
            bag_names = val_bags
            frac = self.val_fraction
            frac_seed = self.seed + 1
        else:
            bag_names = [b for b in AGCO_TEST_BAG_NAMES
                         if (self.rect_root / b).exists() and (self.depth_root / b).exists()]
            frac = self.test_fraction
            frac_seed = self.seed + 2

        self.samples = self._build_samples(bag_names)
        self.samples = _apply_frame_fraction(self.samples, frac, frac_seed)

        print("[AGCO {}] Final samples: {} (frame_fraction={})".format(self.split, len(self.samples), frac))

    def _build_samples(self, bag_names):
        idx_re = re.compile(r"rectified_idx(\d+)_t")
        samples = []
        total_pairs = 0

        for bag in sorted(bag_names):
            img_dir = self.rect_root / bag
            dep_dir = self.depth_root / bag
            if not img_dir.exists() or not dep_dir.exists():
                continue

            img_files = sorted(img_dir.glob("rectified_idx*_t*.png"))
            bag_pairs = 0

            for img_path in img_files:
                m = idx_re.search(img_path.name)
                if m is None:
                    continue
                cam_idx = int(m.group(1))
                depth_candidates = sorted(dep_dir.glob("depth_idx{:06d}_t*.png".format(cam_idx)))
                if len(depth_candidates) == 0:
                    continue
                samples.append((img_path, depth_candidates[0], bag))
                bag_pairs += 1

            total_pairs += bag_pairs
            if self.verbose_bags:
                print("[AGCO {}] Bag {}: {} pairs".format(self.split, bag, bag_pairs))

        if self.verbose_bags:
            print("[AGCO {}] Total pairs: {}".format(self.split, total_pairs))
        return samples

    def _fill_depth_telea_limited(self, depth_m, gt_mask):
        d = np.clip(depth_m.astype(np.float32), 0.0, self.smooth_max_m)
        gt_mask = gt_mask.astype(bool)

        inv = np.where(gt_mask, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        fill_allow = dist <= float(self.max_fill_dist_px)

        fill_mask = (~gt_mask) & fill_allow

        d01 = d / max(self.smooth_max_m, 1e-6)
        d8 = (d01 * 255.0).astype(np.uint8)
        inpaint_mask = (fill_mask.astype(np.uint8) * 255)

        d8_inp = cv2.inpaint(d8, inpaint_mask, 3, cv2.INPAINT_TELEA)
        d_fill = (d8_inp.astype(np.float32) / 255.0) * self.smooth_max_m

        out = np.zeros_like(d, dtype=np.float32)
        out[gt_mask] = d[gt_mask]
        out[fill_mask] = d_fill[fill_mask]

        if self.use_bilateral:
            out_sm = cv2.bilateralFilter(out, d=7, sigmaColor=0.08, sigmaSpace=7)
            out_sm[gt_mask] = d[gt_mask]
            out = out_sm

        out[~np.isfinite(out)] = 0.0
        out = np.clip(out, 0.0, self.smooth_max_m)
        return out.astype(np.float32), fill_mask.astype(bool)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path, bag = self.samples[idx]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to read image: {}".format(str(img_path)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_f).permute(2, 0, 1).float()

        d_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d_mm is None:
            raise RuntimeError("Failed to read depth: {}".format(str(depth_path)))
        d_m = d_mm.astype(np.float32) / 1000.0
        d_m = cv2.resize(d_m, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        gt_mask = (d_m > 0.0) & np.isfinite(d_m)

        if self.smooth_depth:
            d_out, fill_mask = self._fill_depth_telea_limited(d_m, gt_mask)
        else:
            d_out = np.clip(d_m, 0.0, self.smooth_max_m).astype(np.float32)
            fill_mask = np.zeros_like(gt_mask, dtype=bool)

        depth_tensor = torch.from_numpy(d_out).unsqueeze(0).float()

        if self.mask_mode == "gt":
            valid_np = gt_mask
        else:
            valid_np = gt_mask | fill_mask

        valid_mask = torch.from_numpy(valid_np.astype(np.uint8)).unsqueeze(0)  # uint8 (0/1)

        if self.debug_print and idx == 0:
            print("\n[DEBUG idx=0]")
            print("bag:", bag)
            print("img:", img_path.name)
            print("dep:", depth_path.name)
            print("GT pixels    :", int(gt_mask.sum()))
            print("Filled pixels:", int(fill_mask.sum()))
            print("Union pixels :", int((gt_mask | fill_mask).sum()))
            print("mask_mode    :", self.mask_mode)
            print("NOTE: valid_mask is uint8 (0/1)\n")

        meta = {
            "bag": bag,
            "img_path": str(img_path),
            "depth_path": str(depth_path),
            "gt_valid_count": int(gt_mask.sum()),
            "filled_count": int(fill_mask.sum()),
            "mask_mode": self.mask_mode,
            "max_fill_dist_px": int(self.max_fill_dist_px),
        }
        return rgb_tensor, depth_tensor, valid_mask, meta


def visualize_one(ds, idx, out_dir, vmax=25.0, dilate_k=7, dilate_iters=2):
    rgb, depth_sm, valid_mask, meta = ds[idx]

    # load GT again (sparse) for comparison
    d_mm = cv2.imread(meta["depth_path"], cv2.IMREAD_UNCHANGED)
    d_m = d_mm.astype(np.float32) / 1000.0
    d_m = cv2.resize(d_m, (ds.img_width, ds.img_height), interpolation=cv2.INTER_NEAREST)
    gt_mask = (d_m > 0.0) & np.isfinite(d_m)

    rgb01 = rgb.permute(1, 2, 0).numpy()
    d_sm = depth_sm.squeeze(0).numpy()

    gt_gray = depth_to_gray(d_m, vmax)
    sm_gray = depth_to_gray(d_sm, vmax)

    gt_big = dilate_mask(gt_mask, k=dilate_k, iters=dilate_iters)
    valid_np = valid_mask.squeeze(0).numpy().astype(np.uint8) > 0
    valid_big = dilate_mask(valid_np, k=dilate_k, iters=dilate_iters)

    ov_gt = overlay_mask(rgb01, gt_big, color_bgr=(0, 0, 255), alpha=0.55)
    ov_valid = overlay_mask(rgb01, valid_big, color_bgr=(0, 255, 0), alpha=0.50)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "gt_z16.png"), gt_gray)
    cv2.imwrite(str(out_dir / "sm_z16.png"), sm_gray)
    cv2.imwrite(str(out_dir / "gt_mask_big.png"), gt_big)
    cv2.imwrite(str(out_dir / "valid_mask_big.png"), valid_big)
    cv2.imwrite(str(out_dir / "overlay_gt_red.png"), cv2.cvtColor(ov_gt, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "overlay_valid_green.png"), cv2.cvtColor(ov_valid, cv2.COLOR_RGB2BGR))

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs[0, 0].set_title("RGB"); axs[0, 0].imshow(rgb01); axs[0, 0].axis("off")
    axs[0, 1].set_title("GT depth (Z16-like)"); axs[0, 1].imshow(gt_gray, cmap="gray"); axs[0, 1].axis("off")
    axs[0, 2].set_title("Smooth depth (Z16-like)"); axs[0, 2].imshow(sm_gray, cmap="gray"); axs[0, 2].axis("off")

    im1 = axs[1, 0].imshow(d_m, cmap="inferno", vmin=0, vmax=vmax)
    axs[1, 0].set_title("GT depth (colored)"); axs[1, 0].axis("off")
    fig.colorbar(im1, ax=axs[1, 0])

    im2 = axs[1, 1].imshow(d_sm, cmap="inferno", vmin=0, vmax=vmax)
    axs[1, 1].set_title("Smooth depth (colored)"); axs[1, 1].axis("off")
    fig.colorbar(im2, ax=axs[1, 1])

    axs[1, 2].set_title("Overlay: GT=red, valid=green")
    axs[1, 2].imshow(overlay_mask(ov_gt.astype(np.float32)/255.0, valid_big, color_bgr=(0, 255, 0), alpha=0.45))
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    print("[saved debug]", str(out_dir.resolve()))
    print("meta keys:", list(meta.keys()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--idx", type=int, default=120)
    ap.add_argument("--W", type=int, default=640)
    ap.add_argument("--H", type=int, default=364)

    ap.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    ap.add_argument("--val-bag-fraction", type=float, default=DEFAULT_VAL_BAG_FRACTION)

    ap.add_argument("--vmax", type=float, default=25.0)
    ap.add_argument("--max-fill-dist-px", type=int, default=12)
    ap.add_argument("--mask-mode", type=str, default="gt", choices=["gt", "gt+filled"])
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-bilateral", action="store_true")
    ap.add_argument("--quiet-bags", action="store_true")
    ap.add_argument("--debug-print", action="store_true")

    ap.add_argument("--out-dir", type=str, default="./debug_agco_dataset_smooth")
    ap.add_argument("--dilate-k", type=int, default=7)
    ap.add_argument("--dilate-iters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    ds = AGCODepthDataset(
        raw_root=args.raw_root,
        split=args.split,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_bag_fraction=args.val_bag_fraction,
        seed=args.seed,
        img_width=args.W,
        img_height=args.H,
        smooth_depth=(not args.no_smooth),
        smooth_max_m=args.vmax,
        max_fill_dist_px=args.max_fill_dist_px,
        use_bilateral=(not args.no_bilateral),
        mask_mode=args.mask_mode,
        verbose_bags=(not args.quiet_bags),
        debug_print=args.debug_print,
    )

    if len(ds) == 0:
        raise RuntimeError("Dataset is empty for split={}".format(args.split))

    idx = max(0, min(args.idx, len(ds) - 1))
    visualize_one(ds, idx, args.out_dir, vmax=args.vmax, dilate_k=args.dilate_k, dilate_iters=args.dilate_iters)
