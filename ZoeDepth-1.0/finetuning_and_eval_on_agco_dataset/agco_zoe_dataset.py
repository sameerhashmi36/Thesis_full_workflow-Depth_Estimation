"""
AGCOZoeDepthDataset

- Reads rectified RGB images + LiDAR depth_z16 (uint16 mm) for AGCO.
- Supports split="train" / "val" / "test".
- Uses bag-level split from agco_zoe_config.get_agco_bag_split.
- train_fraction / val_fraction / test_fraction control how many samples
  (pairs) are kept from each split.

Returned items per __getitem__:
  rgb        : FloatTensor [3,H,W] in [0,1]
  depth      : FloatTensor [1,H,W] in meters
  valid_mask : BoolTensor  [1,H,W] (depth > 0)
  meta       : dict with 'bag', 'img_path', 'depth_path'
"""

from pathlib import Path
from typing import List, Tuple, Dict
import re
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from agco_zoe_config import (
    DEFAULT_RAW_ROOT,
    AGCO_TEST_BAG_NAMES,
    MIN_DEPTH,
    MAX_DEPTH,
    get_agco_bag_split,
)


class AGCOZoeDepthDataset(Dataset):
    def __init__(
        self,
        raw_root: str,
        split: str = "train",
        train_fraction: float = 1.0,
        val_fraction: float = 1.0,
        test_fraction: float = 1.0,
        img_width: int = 640,
        img_height: int = 192,
        seed: int = 42,
        val_bag_fraction: float = 0.2,
    ):
        """
        raw_root: path to raw_dataset_cpu_manual_1 (with rectified/ and depth_z16/)
        split   : "train", "val", or "test"
        *_fraction: how many samples to keep from that split (0..1)
        """
        super().__init__()

        self.raw_root = Path(raw_root)
        self.rect_root = self.raw_root / "rectified"
        self.depth_root = self.raw_root / "depth_z16"
        self.split = split
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.seed = seed

        assert self.rect_root.is_dir(), f"rectified/ not found at {self.rect_root}"
        assert self.depth_root.is_dir(), f"depth_z16/ not found at {self.depth_root}"

        # Decide which bags belong to which split
        train_bags, val_bags, test_bags = get_agco_bag_split(
            self.rect_root,
            val_bag_fraction=val_bag_fraction,
            seed=seed,
        )

        if split == "train":
            bag_names = train_bags
            frac = train_fraction
            tag = "train"
        elif split == "val":
            bag_names = val_bags
            frac = val_fraction
            tag = "val"
        elif split == "test":
            bag_names = test_bags
            frac = test_fraction
            tag = "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples: List[Tuple[Path, Path, str]] = []
        self._build_samples(bag_names, tag)

        # Subsample according to fraction
        frac = float(frac)
        frac = max(0.0, min(1.0, frac))
        n_total = len(self.samples)
        n_keep = int(round(n_total * frac))
        if n_keep < n_total:
            rng = random.Random(seed + 123 if split == "val" else seed)
            rng.shuffle(self.samples)
            self.samples = sorted(self.samples[:n_keep], key=lambda x: (x[2], x[0].name))

        print(f"[AGCO {split}] Final samples after fraction: {len(self.samples)}")

    def _build_samples(self, bag_names: List[str], tag: str):
        idx_re = re.compile(r"rectified_idx(\d+)_t")

        total_pairs = 0
        for bag in bag_names:
            img_dir = self.rect_root / bag
            dep_dir = self.depth_root / bag

            if not img_dir.exists():
                print(f"[AGCO {tag}] WARN: rectified folder missing: {img_dir}")
                continue
            if not dep_dir.exists():
                print(f"[AGCO {tag}] WARN: depth_z16 folder missing: {dep_dir}")
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
                self.samples.append((img_path, depth_path, bag))
                bag_pairs += 1

            print(f"[AGCO {tag}] Bag {bag}: {bag_pairs} pairs")
            total_pairs += bag_pairs

        print(f"[AGCO {tag}] Total pairs: {total_pairs}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, depth_path, bag = self.samples[idx]

        # RGB
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA
        )
        rgb_f = rgb.astype(np.float32) / 255.0  # [H,W,3] in [0,1]
        rgb_tensor = torch.from_numpy(rgb_f).permute(2, 0, 1)  # [3,H,W]

        # Depth (uint16 mm -> meters)
        d_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d_mm is None:
            raise RuntimeError(f"Failed to read depth: {depth_path}")
        d_m = d_mm.astype(np.float32) / 1000.0
        d_m = cv2.resize(
            d_m, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST
        )
        depth_tensor = torch.from_numpy(d_m).unsqueeze(0)  # [1,H,W]

        valid_mask = (depth_tensor > 0.0) & torch.isfinite(depth_tensor)

        meta = {
            "bag": bag,
            "img_path": str(img_path),
            "depth_path": str(depth_path),
        }

        return rgb_tensor, depth_tensor, valid_mask, meta