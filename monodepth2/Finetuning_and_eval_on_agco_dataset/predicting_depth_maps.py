"""
predicting_depth_maps.py

Modes:
1) Index mode: --idx N
   - Builds a single "ALL" dataset across all common bags (including test bags).
   - GT always available because we pick paired rectified+depth.

2) Folder mode: --in-rgb-dir <folder>
   - Predicts depth for each RGB.
   - ALSO finds GT fast by filename:
       rectified_idx{X}_t...png  -> depth_idx{X:06d}_t...png
     then checks existence under: raw_root/depth_z16/<bag>/<depth_filename>
   - Saves SAME outputs as index mode.

3) Bag clip mode: --bag <bag_name>
   - Creates MP4:
       RGB | GT(gray) | Pred(gray) | GT(color) | Pred(color)
   - GT found fast by direct filename transform (no glob per frame).

Smooth GT option:
  --use-smooth-gt
    Apply Telea-limited inpainting (only within max_fill_dist_px from real GT pixels)
    + optional bilateral filtering.
  This mimics agco_dataset_smooth.py behavior without requiring precomputed smooth files.

Outputs (per image):
out_dir/<model_name>/<image_stem>/
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
import torch.nn.functional as F
from torch.utils.data import Dataset

# monodepth2 repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import networks
from layers import disp_to_depth

from agco_config import (
    DEFAULT_RAW_ROOT,
    MIN_DEPTH,
    MAX_DEPTH,
    DEFAULT_SEED,
)

# -------------------------
# regex / naming helpers
# -------------------------

# rectified_idx123_t....png
RECT_RE = re.compile(r"^rectified_idx(\d+)(_t.*)\.png$", re.IGNORECASE)

def rectified_name_to_depth_name(rect_name: str) -> Optional[str]:
    """
    Convert rectified filename -> expected depth filename.
    rectified_idx123_tXYZ.png -> depth_idx000123_tXYZ.png
    """
    m = RECT_RE.match(rect_name)
    if not m:
        return None
    idx = int(m.group(1))
    tail = m.group(2)  # includes "_t...."
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

def rgb_to_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    return torch.from_numpy(rgb_f).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,H,W]

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

def depth_to_grayscale(depth_m: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    d[d < 0] = 0
    m = d > 0
    if not m.any():
        return np.zeros(d.shape, np.uint8)

    if vmin is None:
        vmin = np.percentile(d[m], 2)
    if vmax is None:
        vmax = np.percentile(d[m], 98)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    norm = np.clip((d - vmin) / (vmax - vmin) * 255.0, 0, 255)
    return norm.astype(np.uint8)

def colorize_depth(depth_m: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """
    OpenCV 3.3.1 compatible.
    Near = bright, Far = dark (inverted), using COLORMAP_HOT.
    """
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    d[d < 0] = 0

    m = d > 0
    if not m.any():
        return np.zeros((d.shape[0], d.shape[1], 3), np.uint8)

    if vmin is None:
        vmin = np.percentile(d[m], 2)
    if vmax is None:
        vmax = np.percentile(d[m], 98)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    dn = np.zeros_like(d, np.float32)
    dn[m] = (d[m] - vmin) / (vmax - vmin)
    dn = np.clip(dn, 0.0, 1.0)
    inv = 1.0 - dn  # near bright, far dark

    img8 = (inv * 255.0).astype(np.uint8)
    cm_bgr = cv2.applyColorMap(img8, cv2.COLORMAP_HOT)
    return cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)

def save_png_rgb(path: Path, rgb_u8: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

def save_png_gray(path: Path, gray_u8: np.ndarray) -> None:
    cv2.imwrite(str(path), gray_u8)

def save_png_u16(path: Path, u16: np.ndarray) -> None:
    cv2.imwrite(str(path), u16)


# -------------------------
# Smooth GT (Telea-limited + optional bilateral)
# -------------------------

def smooth_depth_telea_limited(
    depth_m: np.ndarray,
    smooth_max_m: float = 25.0,
    max_fill_dist_px: int = 12,
    use_bilateral: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    implementation of agco_dataset_smooth.py idea:
    - clamp depth to [0, smooth_max_m]
    - GT mask: depth > 0
    - allow fill only within max_fill_dist_px from GT pixels
    - inpaint in uint8 space (normalized depth), then convert back
    - (optional) bilateral filter, but preserve GT pixels

    Returns:
      depth_smoothed_m : float32 [H,W]
      fill_mask        : bool [H,W]
    """
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
        # Bilateral filter (works on float32 in OpenCV)
        out_sm = cv2.bilateralFilter(out, d=7, sigmaColor=0.08, sigmaSpace=7)
        out_sm[gt_mask] = d[gt_mask]  # preserve original GT
        out = out_sm

    out[~np.isfinite(out)] = 0.0
    out = np.clip(out, 0.0, float(smooth_max_m))
    return out.astype(np.float32), fill_mask.astype(bool)


class LRUCache:
    """
    Small LRU cache for smoothed GT results, to keep bag clips & folders fast.
    Key: str (depth_path)
    Value: (depth_sm, fill_mask)
    """
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


# -------------------------
# stacks + saving
# -------------------------

def make_full_pack(rgb_u8: np.ndarray, gt_m: np.ndarray, pr_m: np.ndarray) -> Dict[str, np.ndarray]:
    gt_gray = depth_to_grayscale(gt_m)
    pr_gray = depth_to_grayscale(pr_m)

    gt_color = colorize_depth(gt_m)
    pr_color = colorize_depth(pr_m)

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
# GT lookup: fast & works from ANY folder
# -------------------------

class FastGTResolver:
    """
    Resolve GT depth path from a rectified filename by checking bags quickly:
      raw_root/depth_z16/<bag>/<depth_filename>
    only need expected depth filename; bag is discovered by exists().
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
            candidate = bag_dir / depth_name
            if candidate.exists():
                self.cache[depth_name] = candidate
                return candidate

        self.cache[depth_name] = None
        return None


# -------------------------
# Model
# -------------------------

def load_model(weights_folder: Path, device: torch.device):
    enc_p = weights_folder / "encoder.pth"
    dep_p = weights_folder / "depth.pth"
    if not enc_p.exists() or not dep_p.exists():
        raise FileNotFoundError(f"Cannot find encoder.pth/depth.pth in {weights_folder}")

    print(f"-> Loading weights from {weights_folder}")
    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    enc = torch.load(enc_p, map_location=device)
    enc = {k: v for k, v in enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(enc)

    dec = torch.load(dep_p, map_location=device)
    decoder.load_state_dict(dec)

    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder

@torch.no_grad()
def predict_depth_m(encoder, decoder, rgb_tensor_1chw: torch.Tensor, device: torch.device,
                    out_hw: Tuple[int, int]) -> np.ndarray:
    rgb_tensor_1chw = rgb_tensor_1chw.to(device)
    feats = encoder(rgb_tensor_1chw)
    disp = decoder(feats)[("disp", 0)]
    if disp.shape[-2:] != out_hw:
        disp = F.interpolate(disp, size=out_hw, mode="bilinear", align_corners=False)
    _, depth_pred = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)  # meters
    d = depth_pred[0, 0].detach().cpu().numpy().astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    d = np.clip(d, 0.0, MAX_DEPTH)
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
    Single list of (rgb, gt_depth) pairs across ALL common bags.
    Uses exact filename transform per bag:
      rectified_idxX_t... -> depth_idx00000X_t...
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
# helper: get GT (sparse or smooth) with cache
# -------------------------

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
    """
    Load GT from depth_path and optionally smooth it.
    Uses cache keyed by str(depth_path).
    """
    if not use_smooth_gt:
        return read_depth_mm_to_m_resize(depth_path, W, H)

    key = str(depth_path)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            depth_sm, _fill = hit
            return depth_sm

    depth_sparse = read_depth_mm_to_m_resize(depth_path, W, H)
    depth_sm, fill_mask = smooth_depth_telea_limited(
        depth_sparse,
        smooth_max_m=smooth_max_m,
        max_fill_dist_px=max_fill_dist_px,
        use_bilateral=use_bilateral,
    )

    if cache is not None:
        cache.put(key, (depth_sm, fill_mask))
    return depth_sm


# -------------------------
# modes
# -------------------------

def run_index_mode(args, encoder, decoder, device, model_name: str):
    ds = AGCOAllDepthDataset(
        raw_root=args.raw_root,
        img_width=args.img_width,
        img_height=args.img_height,
        verbose=False,
    )

    idx = max(0, min(int(args.idx), len(ds) - 1))
    rgb_u8, gt_sparse_m, img_path, depth_path = ds[idx]

    gt_m = get_gt_map(
        depth_path=depth_path,
        W=args.img_width,
        H=args.img_height,
        use_smooth_gt=args.use_smooth_gt,
        smooth_max_m=args.smooth_max_m,
        max_fill_dist_px=args.max_fill_dist_px,
        use_bilateral=(not args.no_bilateral),
        cache=args._gt_cache,
    )

    pr_m = predict_depth_m(
        encoder, decoder,
        rgb_to_tensor(rgb_u8),
        device,
        out_hw=(args.img_height, args.img_width),
    )

    out_base = Path(args.out_dir) / model_name / img_path.stem
    save_outputs(out_base, rgb_u8, gt_m, pr_m)
    print("[OK] saved:", out_base.resolve())
    print("img:", img_path)
    if args.use_smooth_gt:
        print(f"GT: smooth (max_m={args.smooth_max_m}, fill_px={args.max_fill_dist_px}, bilateral={not args.no_bilateral})")
    else:
        print("GT: sparse (depth_z16)")

def run_folder_mode(args, encoder, decoder, device, model_name: str):
    in_dir = Path(args.in_rgb_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"--in-rgb-dir not found: {in_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise RuntimeError(f"No images found in {in_dir}")

    out_root = ensure_dir(Path(args.out_dir) / model_name)
    gt_resolver = FastGTResolver(Path(args.raw_root))

    gt_found = 0
    print(f"[Folder mode] {len(files)} images")

    for p in files:
        rgb_u8 = read_rgb_resize(p, args.img_width, args.img_height)
        pr_m = predict_depth_m(
            encoder, decoder,
            rgb_to_tensor(rgb_u8),
            device,
            out_hw=(args.img_height, args.img_width),
        )

        gt_path = gt_resolver.resolve_depth_path(p.name)
        if gt_path is not None:
            gt_m = get_gt_map(
                depth_path=gt_path,
                W=args.img_width,
                H=args.img_height,
                use_smooth_gt=args.use_smooth_gt,
                smooth_max_m=args.smooth_max_m,
                max_fill_dist_px=args.max_fill_dist_px,
                use_bilateral=(not args.no_bilateral),
                cache=args._gt_cache,
            )
            gt_found += 1
        else:
            gt_m = np.zeros((args.img_height, args.img_width), np.float32)

        out_base = out_root / p.stem
        save_outputs(out_base, rgb_u8, gt_m, pr_m)

    print(f"[OK] saved to: {out_root.resolve()}")
    print(f"[Folder mode] GT found for {gt_found}/{len(files)} images")
    if args.use_smooth_gt:
        print(f"GT smoothing enabled (max_m={args.smooth_max_m}, fill_px={args.max_fill_dist_px}, bilateral={not args.no_bilateral})")

def run_bag_clip_mode(args, encoder, decoder, device, model_name: str):
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

    clips_dir = ensure_dir(Path(args.out_dir) / model_name / "clips")
    out_mp4 = clips_dir / f"{args.bag}.mp4"

    def get_depth_path_for_rectified(rect_name: str) -> Optional[Path]:
        depth_name = rectified_name_to_depth_name(rect_name)
        if depth_name is None:
            return None
        dp = depth_dir / depth_name
        return dp if dp.exists() else None

    # First frame to init writer
    first_rgb = read_rgb_resize(img_files[0], args.img_width, args.img_height)

    first_dp = get_depth_path_for_rectified(img_files[0].name)
    if first_dp is None:
        first_gt = np.zeros((args.img_height, args.img_width), np.float32)
    else:
        first_gt = get_gt_map(
            depth_path=first_dp,
            W=args.img_width,
            H=args.img_height,
            use_smooth_gt=args.use_smooth_gt,
            smooth_max_m=args.smooth_max_m,
            max_fill_dist_px=args.max_fill_dist_px,
            use_bilateral=(not args.no_bilateral),
            cache=args._gt_cache,
        )

    first_pr = predict_depth_m(
        encoder, decoder,
        rgb_to_tensor(first_rgb),
        device,
        out_hw=(args.img_height, args.img_width),
    )

    gt_gray = depth_to_grayscale(first_gt)
    pr_gray = depth_to_grayscale(first_pr)
    gt_color = colorize_depth(first_gt)
    pr_color = colorize_depth(first_pr)

    gt_gray_3c = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2RGB)
    pr_gray_3c = cv2.cvtColor(pr_gray, cv2.COLOR_GRAY2RGB)

    frame = np.concatenate([first_rgb, gt_gray_3c, pr_gray_3c, gt_color, pr_color], axis=1)
    Hf, Wf = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, float(args.fps), (Wf, Hf))
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter (mp4v).")

    vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    for p in img_files[1:]:
        rgb_u8 = read_rgb_resize(p, args.img_width, args.img_height)

        dp = get_depth_path_for_rectified(p.name)
        if dp is None:
            gt_m = np.zeros((args.img_height, args.img_width), np.float32)
        else:
            gt_m = get_gt_map(
                depth_path=dp,
                W=args.img_width,
                H=args.img_height,
                use_smooth_gt=args.use_smooth_gt,
                smooth_max_m=args.smooth_max_m,
                max_fill_dist_px=args.max_fill_dist_px,
                use_bilateral=(not args.no_bilateral),
                cache=args._gt_cache,
            )

        pr_m = predict_depth_m(
            encoder, decoder,
            rgb_to_tensor(rgb_u8),
            device,
            out_hw=(args.img_height, args.img_width),
        )

        gt_g = depth_to_grayscale(gt_m)
        pr_g = depth_to_grayscale(pr_m)
        gt_c = colorize_depth(gt_m)
        pr_c = colorize_depth(pr_m)

        gt_g3 = cv2.cvtColor(gt_g, cv2.COLOR_GRAY2RGB)
        pr_g3 = cv2.cvtColor(pr_g, cv2.COLOR_GRAY2RGB)

        fr = np.concatenate([rgb_u8, gt_g3, pr_g3, gt_c, pr_c], axis=1)
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    vw.release()
    print("[OK] clip saved:", out_mp4.resolve())
    print(f"frames: {len(img_files)} | fps: {args.fps}")
    if args.use_smooth_gt:
        print(f"GT: smooth (max_m={args.smooth_max_m}, fill_px={args.max_fill_dist_px}, bilateral={not args.no_bilateral})")


# -------------------------
# main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT),
                    help="AGCO raw_root that has rectified/ and depth_z16/")
    ap.add_argument("--in-rgb-dir", type=str, default="image_prediction/input_img_to_pred",
                    help="folder containing RGB images to process")
    ap.add_argument("--out-dir", type=str, default="image_prediction/out_vis",
                    help="output directory")

    ap.add_argument("--weights", type=str,
                    default="models_finetuned_on_agco/agco_mono_640x192_finetuned_smooth_train80_val80",
                    help="weights folder containing encoder.pth and depth.pth")

    ap.add_argument("--img-width", type=int, default=640)
    ap.add_argument("--img-height", type=int, default=192)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # index mode (no split)
    ap.add_argument("--idx", type=int, default=-1, help="if >=0 => index mode from ALL pairs")

    # bag clip mode
    ap.add_argument("--bag", type=str, default="", help="if set => bag clip mode")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames, else limit")

    # smooth GT options
    ap.add_argument("--use-smooth-gt", action="store_true",
                    help="apply Telea-limited smoothing to GT depth before visualization")
    ap.add_argument("--smooth-max-m", type=float, default=25.0,
                    help="max depth (meters) used for smoothing normalization/clipping")
    ap.add_argument("--max-fill-dist-px", type=int, default=12,
                    help="max pixel distance from GT where fill is allowed")
    ap.add_argument("--no-bilateral", action="store_true",
                    help="disable bilateral smoothing after fill")
    ap.add_argument("--gt-cache", type=int, default=256,
                    help="LRU cache size for smoothed GT maps (0 disables)")

    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args()

def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    weights_folder = Path(args.weights)
    model_name = weights_folder.name

    encoder, decoder = load_model(weights_folder, device)

    # attach cache into args (simple)
    args._gt_cache = LRUCache(max_items=int(args.gt_cache)) if int(args.gt_cache) > 0 else None

    # Priority: bag clip > index > folder
    if args.bag.strip():
        run_bag_clip_mode(args, encoder, decoder, device, model_name)
        return

    if int(args.idx) >= 0:
        run_index_mode(args, encoder, decoder, device, model_name)
        return

    run_folder_mode(args, encoder, decoder, device, model_name)

if __name__ == "__main__":
    main()
