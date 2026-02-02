"""
agco_dataset.py

raw_root/
  rectified/<bag_name>/rectified_idxXXX_t<time>.png
  depth_z16/<bag_name>/depth_idxXXXXXX_t<time>.png

Normal AGCO dataset (no Telea fill), Zoe/DA3-like fractions:
- val_bag_fraction: bag-level split from NON-test bags
- train_fraction, val_fraction, test_fraction: frame-level subsampling per split

Returns:
  rgb_tensor   : [3,H,W] float32 in [0,1]
  depth_tensor : [1,H,W] float32 meters
  valid_mask   : [1,H,W] bool
  meta         : dict
  
"""

from pathlib import Path
from typing import List, Tuple, Dict
import re
import random
import argparse

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


def _discover_common_bags(raw_root: Path) -> List[str]:
    raw_root = Path(raw_root)
    rect_root = raw_root / "rectified"
    depth_root = raw_root / "depth_z16"

    if not rect_root.exists():
        raise FileNotFoundError(f"rectified/ not found under {raw_root}")
    if not depth_root.exists():
        raise FileNotFoundError(f"depth_z16/ not found under {raw_root}")

    rect_bags = {p.name for p in rect_root.iterdir() if p.is_dir()}
    depth_bags = {p.name for p in depth_root.iterdir() if p.is_dir()}
    common = sorted(rect_bags & depth_bags)

    if not common:
        raise RuntimeError(f"No common bag folders found under {rect_root} and {depth_root}")
    return common


def get_agco_bag_split(
    raw_root: Path,
    val_bag_fraction: float = DEFAULT_VAL_BAG_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Bag-level split from NON-test bags:
      - remove fixed test bags
      - shuffle remaining
      - val_bag_fraction -> val bags, rest -> train bags
    Returns:
      train_bags, val_bags, non_test_bags_shuffled
    """
    raw_root = Path(raw_root)
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

    print(f"[AGCO SPLIT] non-test bags : {len(non_test)}")
    print(f"[AGCO SPLIT] train bags    : {len(train_bags)}")
    print(f"[AGCO SPLIT] val bags      : {len(val_bags)} (val_bag_fraction={vf})")

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


def depth_to_gray(depth_m, vmax=25.0):
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
    color_rgb = np.array(color_bgr[::-1], dtype=np.uint8)  # BGR->RGB
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + color_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return out


class AGCODepthDataset(Dataset):
    def __init__(
        self,
        raw_root: str = str(DEFAULT_RAW_ROOT),
        split: str = "train",
        # frame-level
        train_fraction: float = DEFAULT_TRAIN_FRACTION,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        test_fraction: float = DEFAULT_TEST_FRACTION,
        # bag-level
        val_bag_fraction: float = DEFAULT_VAL_BAG_FRACTION,
        seed: int = DEFAULT_SEED,
        img_width: int = 640,
        img_height: int = 192,
        verbose_bags: bool = True,
    ):
        super().__init__()
        assert split in ("train", "val", "test")

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
        self.verbose_bags = bool(verbose_bags)

        assert self.rect_root.is_dir(), f"rectified/ not found at {self.rect_root}"
        assert self.depth_root.is_dir(), f"depth_z16/ not found at {self.depth_root}"

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
            # fixed test bags that exist
            bag_names = [
                b for b in AGCO_TEST_BAG_NAMES
                if (self.rect_root / b).exists() and (self.depth_root / b).exists()
            ]
            frac = self.test_fraction
            frac_seed = self.seed + 2

        self.samples = self._build_samples(bag_names)
        self.samples = _apply_frame_fraction(self.samples, frac, frac_seed)

        print(f"[AGCO {self.split}] Final samples: {len(self.samples)} (frame_fraction={frac})")

    def _build_samples(self, bag_names: List[str]) -> List[Tuple[Path, Path, str]]:
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
                depth_candidates = sorted(dep_dir.glob(f"depth_idx{cam_idx:06d}_t*.png"))
                if len(depth_candidates) == 0:
                    continue
                samples.append((img_path, depth_candidates[0], bag))
                bag_pairs += 1

            total_pairs += bag_pairs
            if self.verbose_bags:
                print(f"[AGCO {self.split}] Bag {bag}: {bag_pairs} pairs")

        if self.verbose_bags:
            print(f"[AGCO {self.split}] Total pairs: {total_pairs}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, depth_path, bag = self.samples[idx]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_f).permute(2, 0, 1).float()

        d_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d_mm is None:
            raise RuntimeError(f"Failed to read depth: {depth_path}")
        d_m = d_mm.astype(np.float32) / 1000.0
        d_m = cv2.resize(d_m, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        depth_tensor = torch.from_numpy(d_m).unsqueeze(0).float()

        valid_mask = depth_tensor > 0.0  # bool

        meta: Dict[str, str] = {
            "bag": bag,
            "img_path": str(img_path),
            "depth_path": str(depth_path),
        }
        return rgb_tensor, depth_tensor, valid_mask, meta


def visualize_one(ds: AGCODepthDataset, idx: int, out_dir: str, vmax: float = 25.0, dilate_k: int = 7, dilate_iters: int = 2):
    rgb, depth, valid_mask, meta = ds[idx]

    rgb01 = rgb.permute(1, 2, 0).numpy()
    d = depth.squeeze(0).numpy()
    vm = valid_mask.squeeze(0).numpy().astype(bool)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_gray = depth_to_gray(d, vmax=vmax)
    vm_big = dilate_mask(vm, k=dilate_k, iters=dilate_iters)
    ov_valid = overlay_mask(rgb01, vm_big, color_bgr=(0, 255, 0), alpha=0.45)

    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "gt_z16.png"), gt_gray)
    cv2.imwrite(str(out_dir / "valid_mask_big.png"), vm_big)
    cv2.imwrite(str(out_dir / "overlay_valid_green.png"), cv2.cvtColor(ov_valid, cv2.COLOR_RGB2BGR))

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(2, 2, 1); ax1.set_title("RGB"); ax1.imshow(rgb01); ax1.axis("off")
    ax2 = plt.subplot(2, 2, 2); ax2.set_title("GT depth (Z16-like)"); ax2.imshow(gt_gray, cmap="gray"); ax2.axis("off")
    ax3 = plt.subplot(2, 2, 3); ax3.set_title("Valid mask overlay"); ax3.imshow(ov_valid); ax3.axis("off")
    ax4 = plt.subplot(2, 2, 4); ax4.set_title("GT depth (colored)"); im = ax4.imshow(d, cmap="inferno", vmin=0, vmax=vmax); ax4.axis("off")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    print("[saved debug]", str(out_dir.resolve()))
    print("meta:", meta)


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

    ap.add_argument("--out-dir", type=str, default="./debug_agco_dataset")
    ap.add_argument("--vmax", type=float, default=25.0)
    ap.add_argument("--dilate-k", type=int, default=7)
    ap.add_argument("--dilate-iters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quiet-bags", action="store_true")

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
        verbose_bags=(not args.quiet_bags),
    )

    if len(ds) == 0:
        raise RuntimeError(f"Dataset empty for split={args.split}")

    idx = max(0, min(args.idx, len(ds) - 1))
    visualize_one(ds, idx, args.out_dir, vmax=args.vmax, dilate_k=args.dilate_k, dilate_iters=args.dilate_iters)
