#!/usr/bin/env python3
"""
predicting_depth_maps_da3.py

Depth-Anything-3 (DA3) finetuned model predictor + visualization for AGCO.

Modes:
1) Index mode: --idx N
   - Builds "ALL" dataset across all common bags (rectified + depth_z16 paired).
   - GT always exists.

2) Folder mode: --in-rgb-dir <folder>
   - Predict depth for each RGB.
   - If filename matches rectified_idx{X}_t...png, tries fast GT resolve:
       depth_idx{X:06d}_t...png under raw_root/depth_z16/<bag>/
   - If GT not found => GT zeros (still writes all outputs).

3) Bag clip mode: --bag <bag_name>
   - Creates MP4:
       RGB | GT(gray) | Pred(gray) | GT(color) | Pred(color)

Options:
- --use-smooth-gt: smooth/fill GT with Telea-limited method (fast, cacheable)
- --median-scaling: scales pred per-image by median(gt)/median(pred) on valid GT pixels

Outputs per image:
out_dir/<model_tag>/<image_stem>/
  rgb.png
  gt_mm.png
  pred_mm.png
  gt_gray.png
  pred_gray.png
  gt_color.png
  pred_color.png
  stack_rgb_gt_pred.png
  stack_rgb_gt_pred_color.png
  stack_full.png
"""

import sys
import argparse
from pathlib import Path
import re
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from agco_da3_config import (
    DEFAULT_RAW_ROOT,
    DEFAULT_DA3_REPO_ROOT,
    DEFAULT_DA3_MODEL_ID,
    MIN_DEPTH_M,
    MAX_DEPTH_M,
    TRAIN_H,
    TRAIN_W,
)

# -------------------------
# regex / naming helpers
# -------------------------
RECT_RE = re.compile(r"^rectified_idx(\d+)(_t.*)\.png$", re.IGNORECASE)

def rectified_name_to_depth_name(rect_name: str) -> Optional[str]:
    """
    rectified_idx123_tXYZ.png -> depth_idx000123_tXYZ.png
    """
    m = RECT_RE.match(rect_name)
    if not m:
        return None
    idx = int(m.group(1))
    tail = m.group(2)
    return f"depth_idx{idx:06d}{tail}.png"


# -------------------------
# basic I/O helpers
# -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_rgb_resize(path: Path, W: int, H: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    return rgb

def rgb_u8_to_tensor_1chw(rgb_u8: np.ndarray) -> torch.Tensor:
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    return torch.from_numpy(rgb_f).permute(2, 0, 1).unsqueeze(0).float()

def read_depth_mm_to_m_resize(path: Path, W: int, H: int) -> np.ndarray:
    d_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d_mm is None:
        raise RuntimeError(f"Failed to read depth: {path}")
    d_m = d_mm.astype(np.float32) / 1000.0
    d_m = cv2.resize(d_m, (W, H), interpolation=cv2.INTER_NEAREST)
    d_m[~np.isfinite(d_m)] = 0.0
    d_m[d_m < 0] = 0.0
    return d_m.astype(np.float32)

def depth_to_u16_mm(depth_m: np.ndarray) -> np.ndarray:
    d = np.clip(depth_m.astype(np.float32), 0.0, 65.535)
    return (d * 1000.0).round().astype(np.uint16)

def save_png_rgb(path: Path, rgb_u8: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

def save_png_gray(path: Path, gray_u8: np.ndarray) -> None:
    cv2.imwrite(str(path), gray_u8)

def save_png_u16(path: Path, u16: np.ndarray) -> None:
    cv2.imwrite(str(path), u16)


# -------------------------
# Depth visualization (OpenCV 3.3.1 friendly)
# -------------------------
def _robust_vmin_vmax(depth_m: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    m = mask & np.isfinite(depth_m) & (depth_m > 0)
    if not m.any():
        return 0.0, 1.0
    vmin = float(np.percentile(depth_m[m], 2))
    vmax = float(np.percentile(depth_m[m], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-3
    return vmin, vmax

def depth_to_gray_near_white(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin + 1e-8)  # near->0
    g = (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)
    g = 255 - g  # near white
    return g

def colorize_depth_near_bright(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Near=bright, far=dark:
      normalize depth -> [0..255], invert, apply COLORMAP_HOT.
    """
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin + 1e-8)   # near->0
    inv = 1.0 - np.clip(norm, 0.0, 1.0)        # near->1
    img8 = (inv * 255.0).astype(np.uint8)
    cm_bgr = cv2.applyColorMap(img8, cv2.COLORMAP_HOT)
    return cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)


# -------------------------
# Smooth GT (Telea-limited + optional bilateral) + LRU cache
# -------------------------
def smooth_depth_telea_limited(
    depth_m: np.ndarray,
    smooth_max_m: float = 25.0,
    max_fill_dist_px: int = 12,
    use_bilateral: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.clip(depth_m.astype(np.float32), 0.0, float(smooth_max_m))
    gt_mask = (d > 0.0) & np.isfinite(d)

    if not gt_mask.any():
        return d.astype(np.float32), np.zeros_like(gt_mask, dtype=bool)

    inv = np.where(gt_mask, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    fill_allow = dist <= float(max_fill_dist_px)
    fill_mask = (~gt_mask) & fill_allow

    if not fill_mask.any():
        return d.astype(np.float32), fill_mask.astype(bool)

    d01 = d / max(float(smooth_max_m), 1e-6)
    d8 = (d01 * 255.0).astype(np.uint8)
    inpaint_mask = (fill_mask.astype(np.uint8) * 255)

    d8_inp = cv2.inpaint(d8, inpaint_mask, 3, cv2.INPAINT_TELEA)
    d_fill = (d8_inp.astype(np.float32) / 255.0) * float(smooth_max_m)

    out = np.zeros_like(d, dtype=np.float32)
    out[gt_mask] = d[gt_mask]
    out[fill_mask] = d_fill[fill_mask]

    if use_bilateral:
        out_sm = cv2.bilateralFilter(out, d=7, sigmaColor=0.08, sigmaSpace=7)
        out_sm[gt_mask] = d[gt_mask]
        out = out_sm

    out[~np.isfinite(out)] = 0.0
    out = np.clip(out, 0.0, float(smooth_max_m))
    return out.astype(np.float32), fill_mask.astype(bool)

class LRUCache:
    def __init__(self, max_items: int = 256):
        self.max_items = int(max(0, max_items))
        self._d = OrderedDict()

    def get(self, k):
        if k not in self._d:
            return None
        v = self._d.pop(k)
        self._d[k] = v
        return v

    def put(self, k, v):
        if self.max_items <= 0:
            return
        if k in self._d:
            self._d.pop(k)
        self._d[k] = v
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)

def get_gt_map(
    depth_path: Path,
    W: int,
    H: int,
    use_smooth_gt: bool,
    smooth_max_m: float,
    max_fill_dist_px: int,
    use_bilateral: bool,
    cache: Optional[LRUCache],
) -> np.ndarray:
    if not use_smooth_gt:
        return read_depth_mm_to_m_resize(depth_path, W, H)

    key = str(depth_path)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            depth_sm, _fill = hit
            return depth_sm

    sparse = read_depth_mm_to_m_resize(depth_path, W, H)
    depth_sm, fill_mask = smooth_depth_telea_limited(
        sparse,
        smooth_max_m=smooth_max_m,
        max_fill_dist_px=max_fill_dist_px,
        use_bilateral=use_bilateral,
    )
    if cache is not None:
        cache.put(key, (depth_sm, fill_mask))
    return depth_sm


# -------------------------
# GT resolver for folder mode
# -------------------------
class FastGTResolver:
    """
    Resolve GT depth path from a rectified filename by checking:
      raw_root/depth_z16/<bag>/<depth_filename>
    """
    def __init__(self, raw_root: Path):
        self.raw_root = Path(raw_root)
        self.depth_root = self.raw_root / "depth_z16"
        if not self.depth_root.exists():
            raise FileNotFoundError(f"depth_z16/ not found under: {self.raw_root}")
        self.bags: List[Path] = sorted([p for p in self.depth_root.iterdir() if p.is_dir()])
        self.cache: Dict[str, Optional[Path]] = {}

    def resolve_depth_path(self, rectified_filename: str) -> Optional[Path]:
        depth_name = rectified_name_to_depth_name(rectified_filename)
        if depth_name is None:
            return None
        if depth_name in self.cache:
            return self.cache[depth_name]
        for bag_dir in self.bags:
            cand = bag_dir / depth_name
            if cand.exists():
                self.cache[depth_name] = cand
                return cand
        self.cache[depth_name] = None
        return None


# -------------------------
# DA3 model loading + forward 
# -------------------------
def choose_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_da3_metric(net: nn.Module, rgb_bchw: torch.Tensor) -> torch.Tensor:
    """
      x = rgb.unsqueeze(1)  # (B,1,C,H,W)
      out = net(x, None, None, (), False)
    Returns: (B,1,H,W)
    """
    x = rgb_bchw.unsqueeze(1)
    out = net(x, None, None, (), False)

    if isinstance(out, dict):
        for k in ["metric_depth", "depth", "pred", "out"]:
            if k in out:
                out = out[k]
                break
    if isinstance(out, (list, tuple)):
        out = out[0]

    # Normalize to (B,1,H,W)
    if out.dim() == 5:
        # common: (B,1,1,h,w) -> out[:,0] => (B,1,h,w)
        if out.shape[1] == 1 and out.shape[2] == 1:
            out = out[:, 0]
        else:
            out = out[:, 0]
    if out.dim() == 4:
        if out.shape[1] != 1:
            out = out[:, 0].unsqueeze(1)
    elif out.dim() == 3:
        out = out.unsqueeze(1)

    return out

def build_da3_model(
    da3_repo_root: Path,
    model_id: str,
    finetuned_weights: Path,
    device: torch.device,
) -> nn.Module:
    # allow local import
    if da3_repo_root.exists() and str(da3_repo_root) not in sys.path:
        sys.path.append(str(da3_repo_root))

    from depth_anything_3.api import DepthAnything3

    da3 = DepthAnything3.from_pretrained(model_id).to(device=device)
    net = da3.model.to(device).eval()

    state = torch.load(str(finetuned_weights), map_location=device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


@torch.no_grad()
def predict_depth_m_da3(net: nn.Module, rgb_1chw: torch.Tensor, device: torch.device, out_hw: Tuple[int, int]) -> np.ndarray:
    rgb_1chw = rgb_1chw.to(device)
    pred = forward_da3_metric(net, rgb_1chw)  # (1,1,h,w)

    if pred.shape[-2:] != out_hw:
        pred = F.interpolate(pred, size=out_hw, mode="bilinear", align_corners=False)

    d = pred[0, 0].detach().cpu().numpy().astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    d = np.clip(d, 0.0, float(MAX_DEPTH_M))
    return d


# -------------------------
# ALL dataset (no split)
# -------------------------
def discover_common_bags(raw_root: Path) -> List[str]:
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
        raise RuntimeError("No common bags found between rectified/ and depth_z16/")
    return common

class AGCOAllDepthDataset(Dataset):
    """
    Single list of (rgb_u8, gt_sparse_m, img_path, depth_path) across ALL common bags.
    """
    def __init__(self, raw_root: str, img_width: int, img_height: int, verbose: bool = True):
        super().__init__()
        self.raw_root = Path(raw_root)
        self.rect_root = self.raw_root / "rectified"
        self.depth_root = self.raw_root / "depth_z16"
        self.W = int(img_width)
        self.H = int(img_height)

        bags = discover_common_bags(self.raw_root)
        samples: List[Tuple[Path, Path]] = []

        for bag in bags:
            img_dir = self.rect_root / bag
            dep_dir = self.depth_root / bag
            if not img_dir.exists() or not dep_dir.exists():
                continue

            img_files = sorted(img_dir.glob("rectified_idx*_t*.png"))
            bag_pairs = 0

            for img_path in img_files:
                depth_name = rectified_name_to_depth_name(img_path.name)
                if depth_name is None:
                    continue
                depth_path = dep_dir / depth_name
                if not depth_path.exists():
                    continue
                samples.append((img_path, depth_path))
                bag_pairs += 1

            if verbose:
                print(f"[ALL] Bag {bag}: {bag_pairs} pairs")

        if not samples:
            raise RuntimeError("No paired samples found for ALL dataset.")
        self.samples = samples
        if verbose:
            print(f"[ALL] Total pairs: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, depth_path = self.samples[idx]
        rgb_u8 = read_rgb_resize(img_path, self.W, self.H)
        gt_m = read_depth_mm_to_m_resize(depth_path, self.W, self.H)
        return rgb_u8, gt_m, img_path, depth_path


# -------------------------
# Saving packs
# -------------------------
def make_full_pack(rgb_u8: np.ndarray, gt_m: np.ndarray, pr_m: np.ndarray) -> Dict[str, np.ndarray]:
    gt_valid = (gt_m > 0) & np.isfinite(gt_m)
    pr_valid = (pr_m > 0) & np.isfinite(pr_m)
    if gt_valid.any():
        vmin, vmax = _robust_vmin_vmax(gt_m, gt_valid)
    elif pr_valid.any():
        vmin, vmax = _robust_vmin_vmax(pr_m, pr_valid)
    else:
        vmin, vmax = 0.0, 1.0

    gt_gray = depth_to_gray_near_white(gt_m, vmin, vmax)
    pr_gray = depth_to_gray_near_white(pr_m, vmin, vmax)
    gt_color = colorize_depth_near_bright(gt_m, vmin, vmax)
    pr_color = colorize_depth_near_bright(pr_m, vmin, vmax)

    gt_gray_3c = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2RGB)
    pr_gray_3c = cv2.cvtColor(pr_gray, cv2.COLOR_GRAY2RGB)

    stack_rgb_gt_pred = np.concatenate([rgb_u8, gt_gray_3c, pr_gray_3c], axis=1)
    stack_rgb_gt_pred_color = np.concatenate([rgb_u8, gt_color, pr_color], axis=1)
    stack_full = np.concatenate([rgb_u8, gt_color, gt_gray_3c, pr_color, pr_gray_3c], axis=1)

    return {
        "gt_gray": gt_gray,
        "pred_gray": pr_gray,
        "gt_color": gt_color,
        "pred_color": pr_color,
        "stack_rgb_gt_pred": stack_rgb_gt_pred,
        "stack_rgb_gt_pred_color": stack_rgb_gt_pred_color,
        "stack_full": stack_full,
    }

def save_outputs(out_base: Path, rgb_u8: np.ndarray, gt_m: np.ndarray, pr_m: np.ndarray):
    ensure_dir(out_base)

    save_png_rgb(out_base / "rgb.png", rgb_u8)
    save_png_u16(out_base / "gt_mm.png", depth_to_u16_mm(gt_m))
    save_png_u16(out_base / "pred_mm.png", depth_to_u16_mm(pr_m))

    pack = make_full_pack(rgb_u8, gt_m, pr_m)

    save_png_gray(out_base / "gt_gray.png", pack["gt_gray"])
    save_png_gray(out_base / "pred_gray.png", pack["pred_gray"])

    save_png_rgb(out_base / "gt_color.png", pack["gt_color"])
    save_png_rgb(out_base / "pred_color.png", pack["pred_color"])

    save_png_rgb(out_base / "stack_rgb_gt_pred.png", pack["stack_rgb_gt_pred"])
    save_png_rgb(out_base / "stack_rgb_gt_pred_color.png", pack["stack_rgb_gt_pred_color"])
    save_png_rgb(out_base / "stack_full.png", pack["stack_full"])


# -------------------------
# Median scaling (for visuals)
# -------------------------
def apply_median_scaling_if_enabled(pr_m: np.ndarray, gt_m: np.ndarray, do_scale: bool) -> np.ndarray:
    if not do_scale:
        return pr_m
    m = (gt_m > MIN_DEPTH_M) & (gt_m < MAX_DEPTH_M) & np.isfinite(gt_m) & (pr_m > 0) & np.isfinite(pr_m)
    if not m.any():
        return pr_m
    gt_flat = gt_m[m]
    pr_flat = pr_m[m]
    scale = float(np.median(gt_flat) / (np.median(pr_flat) + 1e-8))
    pr_scaled = np.clip(pr_m * scale, 0.0, float(MAX_DEPTH_M))
    return pr_scaled.astype(np.float32)


# -------------------------
# modes
# -------------------------
def run_index_mode(args, net, device, model_tag: str):
    ds = AGCOAllDepthDataset(
        raw_root=args.raw_root,
        img_width=args.img_width,
        img_height=args.img_height,
        verbose=False,
    )
    idx = max(0, min(int(args.idx), len(ds) - 1))
    rgb_u8, _gt_sparse, img_path, depth_path = ds[idx]

    gt_m = get_gt_map(
        depth_path=depth_path,
        W=args.img_width, H=args.img_height,
        use_smooth_gt=args.use_smooth_gt,
        smooth_max_m=args.smooth_max_m,
        max_fill_dist_px=args.max_fill_dist_px,
        use_bilateral=(not args.no_bilateral),
        cache=args._gt_cache,
    )

    pr_m = predict_depth_m_da3(net, rgb_u8_to_tensor_1chw(rgb_u8), device, out_hw=(args.img_height, args.img_width))
    pr_m = apply_median_scaling_if_enabled(pr_m, gt_m, args.median_scaling)

    out_base = Path(args.out_dir) / model_tag / img_path.stem
    save_outputs(out_base, rgb_u8, gt_m, pr_m)
    print("[OK] saved:", out_base.resolve())
    print("img:", img_path)

def run_folder_mode(args, net, device, model_tag: str):
    in_dir = Path(args.in_rgb_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"--in-rgb-dir not found: {in_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise RuntimeError(f"No images found in {in_dir}")

    out_root = ensure_dir(Path(args.out_dir) / model_tag)
    gt_resolver = FastGTResolver(Path(args.raw_root))
    gt_found = 0

    print(f"[Folder mode] {len(files)} images")

    for p in files:
        rgb_u8 = read_rgb_resize(p, args.img_width, args.img_height)
        pr_m = predict_depth_m_da3(net, rgb_u8_to_tensor_1chw(rgb_u8), device, out_hw=(args.img_height, args.img_width))

        gt_path = gt_resolver.resolve_depth_path(p.name)
        if gt_path is not None:
            gt_m = get_gt_map(
                depth_path=gt_path,
                W=args.img_width, H=args.img_height,
                use_smooth_gt=args.use_smooth_gt,
                smooth_max_m=args.smooth_max_m,
                max_fill_dist_px=args.max_fill_dist_px,
                use_bilateral=(not args.no_bilateral),
                cache=args._gt_cache,
            )
            gt_found += 1
        else:
            gt_m = np.zeros((args.img_height, args.img_width), np.float32)

        pr_m = apply_median_scaling_if_enabled(pr_m, gt_m, args.median_scaling)

        out_base = out_root / p.stem
        save_outputs(out_base, rgb_u8, gt_m, pr_m)

    print(f"[OK] saved to: {out_root.resolve()}")
    print(f"[Folder mode] GT found for {gt_found}/{len(files)} images")

def run_bag_clip_mode(args, net, device, model_tag: str):
    raw_root = Path(args.raw_root)
    rect_dir = raw_root / "rectified" / args.bag
    depth_dir = raw_root / "depth_z16" / args.bag

    if not rect_dir.exists():
        raise FileNotFoundError(f"Bag rectified folder not found: {rect_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Bag depth folder not found: {depth_dir}")

    img_files = sorted(rect_dir.glob("rectified_idx*_t*.png"))
    if not img_files:
        raise RuntimeError(f"No rectified images in: {rect_dir}")

    if args.max_frames > 0:
        img_files = img_files[: int(args.max_frames)]

    clips_dir = ensure_dir(Path(args.out_dir) / model_tag / "clips")
    out_mp4 = clips_dir / f"{args.bag}.mp4"

    def get_depth_path_for_rectified(rect_name: str) -> Optional[Path]:
        depth_name = rectified_name_to_depth_name(rect_name)
        if depth_name is None:
            return None
        dp = depth_dir / depth_name
        return dp if dp.exists() else None

    # Init writer
    first_rgb = read_rgb_resize(img_files[0], args.img_width, args.img_height)

    dp0 = get_depth_path_for_rectified(img_files[0].name)
    if dp0 is None:
        gt0 = np.zeros((args.img_height, args.img_width), np.float32)
    else:
        gt0 = get_gt_map(
            depth_path=dp0,
            W=args.img_width, H=args.img_height,
            use_smooth_gt=args.use_smooth_gt,
            smooth_max_m=args.smooth_max_m,
            max_fill_dist_px=args.max_fill_dist_px,
            use_bilateral=(not args.no_bilateral),
            cache=args._gt_cache,
        )

    pr0 = predict_depth_m_da3(net, rgb_u8_to_tensor_1chw(first_rgb), device, out_hw=(args.img_height, args.img_width))
    pr0 = apply_median_scaling_if_enabled(pr0, gt0, args.median_scaling)

    gt_valid = (gt0 > 0) & np.isfinite(gt0)
    pr_valid = (pr0 > 0) & np.isfinite(pr0)
    if gt_valid.any():
        vmin, vmax = _robust_vmin_vmax(gt0, gt_valid)
    elif pr_valid.any():
        vmin, vmax = _robust_vmin_vmax(pr0, pr_valid)
    else:
        vmin, vmax = 0.0, 1.0

    gt0_gray = depth_to_gray_near_white(gt0, vmin, vmax)
    pr0_gray = depth_to_gray_near_white(pr0, vmin, vmax)
    gt0_col = colorize_depth_near_bright(gt0, vmin, vmax)
    pr0_col = colorize_depth_near_bright(pr0, vmin, vmax)

    gt0_gray3 = cv2.cvtColor(gt0_gray, cv2.COLOR_GRAY2RGB)
    pr0_gray3 = cv2.cvtColor(pr0_gray, cv2.COLOR_GRAY2RGB)

    frame0 = np.concatenate([first_rgb, gt0_gray3, pr0_gray3, gt0_col, pr0_col], axis=1)
    Hf, Wf = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, float(args.fps), (Wf, Hf))
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter (mp4v).")

    vw.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))

    for p in img_files[1:]:
        rgb_u8 = read_rgb_resize(p, args.img_width, args.img_height)

        dp = get_depth_path_for_rectified(p.name)
        if dp is None:
            gt_m = np.zeros((args.img_height, args.img_width), np.float32)
        else:
            gt_m = get_gt_map(
                depth_path=dp,
                W=args.img_width, H=args.img_height,
                use_smooth_gt=args.use_smooth_gt,
                smooth_max_m=args.smooth_max_m,
                max_fill_dist_px=args.max_fill_dist_px,
                use_bilateral=(not args.no_bilateral),
                cache=args._gt_cache,
            )

        pr_m = predict_depth_m_da3(net, rgb_u8_to_tensor_1chw(rgb_u8), device, out_hw=(args.img_height, args.img_width))
        pr_m = apply_median_scaling_if_enabled(pr_m, gt_m, args.median_scaling)

        gt_valid = (gt_m > 0) & np.isfinite(gt_m)
        pr_valid = (pr_m > 0) & np.isfinite(pr_m)
        if gt_valid.any():
            vmin, vmax = _robust_vmin_vmax(gt_m, gt_valid)
        elif pr_valid.any():
            vmin, vmax = _robust_vmin_vmax(pr_m, pr_valid)
        else:
            vmin, vmax = 0.0, 1.0

        gt_g = depth_to_gray_near_white(gt_m, vmin, vmax)
        pr_g = depth_to_gray_near_white(pr_m, vmin, vmax)
        gt_c = colorize_depth_near_bright(gt_m, vmin, vmax)
        pr_c = colorize_depth_near_bright(pr_m, vmin, vmax)

        gt_g3 = cv2.cvtColor(gt_g, cv2.COLOR_GRAY2RGB)
        pr_g3 = cv2.cvtColor(pr_g, cv2.COLOR_GRAY2RGB)

        fr = np.concatenate([rgb_u8, gt_g3, pr_g3, gt_c, pr_c], axis=1)
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    vw.release()
    print("[OK] clip saved:", out_mp4.resolve())
    print(f"frames: {len(img_files)} | fps: {args.fps}")


# -------------------------
# main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT),
                   help="AGCO raw root that has rectified/ and depth_z16/")

    p.add_argument("--in-rgb-dir", type=str, default="image_prediction/input_img_to_pred",
                   help="folder containing RGB images to process")
    p.add_argument("--out-dir", type=str, default="image_prediction/out_vis_da3",
                   help="output directory")

    # DA3 repo + weights
    p.add_argument("--da3-repo-root", type=str, default=str(DEFAULT_DA3_REPO_ROOT),
                   help="local Depth-Anything-3 repo path (for import)")
    p.add_argument("--model-id", type=str, default=str(DEFAULT_DA3_MODEL_ID),
                   help="HF model id used by DepthAnything3.from_pretrained(...)")
    p.add_argument("--finetuned-weights", type=str, default="models_finetuned_on_agco_smooth/depth-anything_DA3-LARGE_e30_train80_val80_smooth/best.pth",
                   help="finetuned state_dict .pth (strict=True into da3.model)")

    p.add_argument("--img-width", type=int, default=int(TRAIN_W*2))
    p.add_argument("--img-height", type=int, default=int(TRAIN_H*2))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # modes
    p.add_argument("--idx", type=int, default=-1, help="if >=0 => index mode (ALL pairs)")
    p.add_argument("--bag", type=str, default="", help="if set => bag clip mode")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames, else limit")

    # GT smoothing
    p.add_argument("--use-smooth-gt", action="store_true",
                   help="apply Telea-limited smoothing to GT depth before visualization")
    p.add_argument("--smooth-max-m", type=float, default=25.0)
    p.add_argument("--max-fill-dist-px", type=int, default=12)
    p.add_argument("--no-bilateral", action="store_true")
    p.add_argument("--gt-cache", type=int, default=256, help="LRU cache size for smoothed GT maps (0 disables)")

    # visuals
    p.add_argument("--median-scaling", action="store_true",
                   help="scale pred by median(gt)/median(pred) on valid GT pixels")

    return p.parse_args()

def main():
    args = parse_args()
    device = choose_device(args.device)
    print("Device:", device)

    finetuned = Path(args.finetuned_weights)
    if not finetuned.exists():
        raise FileNotFoundError(f"Finetuned weights not found: {finetuned}")

    # model tag for output folder (similar style as zoe)
    model_tag = f"DA3__{finetuned.parent.name}__{finetuned.stem}"

    # cache
    args._gt_cache = LRUCache(max_items=int(args.gt_cache)) if int(args.gt_cache) > 0 else None

    net = build_da3_model(
        da3_repo_root=Path(args.da3_repo_root),
        model_id=args.model_id,
        finetuned_weights=finetuned,
        device=device,
    )
    print("[FT] strict=True load OK ✅")

    # Priority: bag clip > index > folder
    if args.bag.strip():
        run_bag_clip_mode(args, net, device, model_tag)
        return
    if int(args.idx) >= 0:
        run_index_mode(args, net, device, model_tag)
        return
    run_folder_mode(args, net, device, model_tag)

if __name__ == "__main__":
    main()
