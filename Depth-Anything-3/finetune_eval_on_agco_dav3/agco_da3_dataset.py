"""
AGCO dataset for DA3 supervised finetuning.

Folder layout (raw_root):
  raw_root/
    rectified/<bag>/rectified_idxXXXXXX_t*.png
    depth_z16/<bag>/depth_idxXXXXXX_t*.png

Key points:
- depth_z16 is uint16 in millimeters -> converted to float meters
- split is at BAG LEVEL:
    - test set is fixed by bag names
    - train/val are split from remaining bags
- train_fraction/val_fraction/test_fraction sub-sample only inside that split

Return from __getitem__:
  rgb_tensor  : FloatTensor [3,H,W] in [0,1]
  depth_tensor: FloatTensor [H,W] in meters
  valid_mask  : BoolTensor  [H,W] (depth > 0 and finite)
  meta        : dict with bag/img/depth paths
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import re
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from agco_da3_config import (
    MIN_DEPTH_M,
    MAX_DEPTH_M,
    get_agco_bag_split,
)


class AGCODA3DepthDataset(Dataset):
    def __init__(
        self,
        raw_root: str,
        split: str = "train",
        img_width: int = 448,
        img_height: int = 224,
        # fractions are applied inside chosen split
        train_fraction: float = 1.0,
        val_fraction: float = 1.0,
        test_fraction: float = 1.0,
        # split control
        split_seed: int = 42,
        val_bag_fraction: float = 0.2,
        # depth filtering constants (kept here so train/eval can override safely)
        min_depth_m: float = MIN_DEPTH_M,
        max_depth_m: float = MAX_DEPTH_M,
        verbose: bool = True,
    ):
        super().__init__()

        self.raw_root = Path(raw_root)
        self.rect_root = self.raw_root / "rectified"
        self.depth_root = self.raw_root / "depth_z16"

        self.split = str(split).lower().strip()
        self.img_width = int(img_width)
        self.img_height = int(img_height)

        self.split_seed = int(split_seed)
        self.val_bag_fraction = float(val_bag_fraction)

        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.verbose = bool(verbose)

        assert self.rect_root.is_dir(), f"rectified/ not found at: {self.rect_root}"
        assert self.depth_root.is_dir(), f"depth_z16/ not found at: {self.depth_root}"

        # ---- bag split ----
        train_bags, val_bags, test_bags = get_agco_bag_split(
            self.rect_root,
            val_bag_fraction=self.val_bag_fraction,
            seed=self.split_seed,
        )

        if self.split == "train":
            bag_names = train_bags
            frac = float(train_fraction)
            tag = "train"
        elif self.split == "val":
            bag_names = val_bags
            frac = float(val_fraction)
            tag = "val"
        elif self.split == "test":
            bag_names = test_bags
            frac = float(test_fraction)
            tag = "test"
        else:
            raise ValueError(f"Unknown split: {split} (use train/val/test)")

        self.samples: List[Tuple[Path, Path, str]] = []
        self._build_samples(bag_names=bag_names, tag=tag)

        # ---- fraction applied inside split only ----
        frac = max(0.0, min(1.0, frac))
        n_total = len(self.samples)
        n_keep = int(round(n_total * frac))

        if n_keep < n_total:
            split_offset = {"train": 0, "val": 123, "test": 999}.get(self.split, 0)
            rng = random.Random(self.split_seed + split_offset)
            rng.shuffle(self.samples)
            keep = self.samples[:n_keep]
            self.samples = sorted(keep, key=lambda x: (x[2], x[0].name))

        if self.verbose:
            print(f"[AGCO {self.split}] Final samples: {len(self.samples)}")
            print(f"[AGCO {self.split}] depth mask range: [{self.min_depth_m}, {self.max_depth_m}] m")

    def _build_samples(self, bag_names: List[str], tag: str):
        """
        Build (rectified_rgb, depth_z16) pairs by matching idx numbers.
        """
        idx_re = re.compile(r"rectified_idx(\d+)_t")
        total_pairs = 0

        for bag in bag_names:
            img_dir = self.rect_root / bag
            dep_dir = self.depth_root / bag

            if not img_dir.exists():
                if self.verbose:
                    print(f"[AGCO {tag}] WARN: missing rectified dir: {img_dir}")
                continue
            if not dep_dir.exists():
                if self.verbose:
                    print(f"[AGCO {tag}] WARN: missing depth_z16 dir: {dep_dir}")
                continue

            img_files = sorted(img_dir.glob("rectified_idx*_t*.png"))
            bag_pairs = 0

            for img_path in img_files:
                m = idx_re.search(img_path.name)
                if not m:
                    continue
                cam_idx = int(m.group(1))

                depth_candidates = sorted(dep_dir.glob(f"depth_idx{cam_idx:06d}_t*.png"))
                if not depth_candidates:
                    continue

                depth_path = depth_candidates[0]
                self.samples.append((img_path, depth_path, bag))
                bag_pairs += 1

            if self.verbose:
                print(f"[AGCO {tag}] Bag {bag}: {bag_pairs} pairs")
            total_pairs += bag_pairs

        if self.verbose:
            print(f"[AGCO {tag}] Total pairs: {total_pairs}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, depth_path, bag = self.samples[idx]

        # --- RGB ---
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_f).permute(2, 0, 1)  # [3,H,W]

        # --- Depth uint16 mm -> meters ---
        d_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d_mm is None:
            raise RuntimeError(f"Failed to read depth: {depth_path}")
        if d_mm.dtype != np.uint16:
            # still convert, but warn (helps debugging)
            if self.verbose:
                print(f"[AGCO {self.split}] WARN: depth dtype is {d_mm.dtype}, expected uint16: {depth_path}")

        d_m = d_mm.astype(np.float32) / 1000.0
        d_m = cv2.resize(d_m, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        depth_tensor = torch.from_numpy(d_m)  # [H,W]

        valid_mask = (depth_tensor > 0.0) & torch.isfinite(depth_tensor)

        meta: Dict[str, str] = {
            "bag": bag,
            "img_path": str(img_path),
            "depth_path": str(depth_path),
        }

        return rgb_tensor, depth_tensor, valid_mask, meta
