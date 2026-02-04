"""
predicting_depth_maps_zoe_zeroshot.py

ZoeDepth ZERO-SHOT predicting/visualization script.

Modes:
1) Index mode: --idx N
   - Builds paired list from ALL common bags: rectified/<bag> + depth_z16/<bag>
   - GT always available (paired by filename transform)
   - Saves:
       RGB | GT(gray) | Pred(gray) | GT(color) | Pred(color)

2) Folder mode: --in-rgb-dir <folder>
   - Predicts for images in that folder.
   - If --try-gt:
       rectified_idxX_t*.png -> depth_idx00000X_t*.png
     and searches under raw_root/depth_z16/<bag>/... across all bags (cached).
   - If GT not found -> prediction-only outputs.

3) Bag mode: --bag <bag_name>
   - Predicts for raw_root/rectified/<bag> frames.
   - If --try-gt -> loads GT from raw_root/depth_z16/<bag>.
   - Optional mp4 clip: --make-clip
       if GT exists:
         RGB | GT(gray) | Pred(gray) | GT(color) | Pred(color)
       else:
         RGB | Pred(gray) | Pred(color)
   - Optional per-frame saving: --save-frames

Visualization:
- gray: near white, far black (robust percentile scaling)
- color: near bright, far dark (OpenCV 3.3.1 friendly: COLORMAP_HOT)

Notes:
- Zoe inference uses zoe.infer_pil(PIL.Image) -> depth (meters) at original image resolution.
- We keep Zoe output resolution; GT is resized to match prediction for fair visual comparison.

Outputs:
out_dir/zoe_zeroshot/<model_name>/<image_stem>/
  rgb.png
  pred_mm.png
  pred_gray.png
  pred_color.png
  stack_rgb_pred_gray.png
  stack_rgb_pred_color.png
If GT exists:
  gt_mm.png
  gt_gray.png
  gt_color.png
  stack_rgb_gt_pred_gray.png
  stack_rgb_gt_pred_color.png
  stack_full.png
"""

import os
import sys
import argparse
from pathlib import Path
import re
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# -------------------------
# filename helpers
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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------
# IO helpers
# -------------------------
def read_depth_mm_to_m(path: Path) -> np.ndarray:
    d_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d_mm is None:
        raise RuntimeError(f"Failed to read depth: {path}")
    d_m = d_mm.astype(np.float32) / 1000.0
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
# visualization (OpenCV 3.3.1 friendly)
# -------------------------
def robust_vmin_vmax(depth_m: np.ndarray) -> Tuple[float, float]:
    d = depth_m.astype(np.float32)
    m = np.isfinite(d) & (d > 0)
    if not m.any():
        return 0.0, 1.0
    vmin = float(np.percentile(d[m], 2))
    vmax = float(np.percentile(d[m], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-3
    return vmin, vmax


def gray_near_white(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)

    norm = (d - vmin) / (vmax - vmin + 1e-8)  # near->0
    g = (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)
    g = 255 - g  # near white
    return g


def color_near_bright(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Near=bright, far=dark using COLORMAP_HOT (OpenCV 3.3.1 available).
    """
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)

    norm = (d - vmin) / (vmax - vmin + 1e-8)      # near->0
    inv = 1.0 - np.clip(norm, 0.0, 1.0)           # near->1
    img8 = (inv * 255.0).astype(np.uint8)

    cm_bgr = cv2.applyColorMap(img8, cv2.COLORMAP_HOT)
    return cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)


def make_stacks(rgb_u8: np.ndarray, pred_m: np.ndarray, gt_m: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    # Use vmin/vmax from GT if available else pred
    if gt_m is not None and np.any(gt_m > 0):
        vmin, vmax = robust_vmin_vmax(gt_m)
    else:
        vmin, vmax = robust_vmin_vmax(pred_m)

    pred_gray = gray_near_white(pred_m, vmin, vmax)
    pred_gray3 = cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2RGB)
    pred_col = color_near_bright(pred_m, vmin, vmax)

    out = {
        "pred_gray": pred_gray,
        "pred_color": pred_col,
        "stack_rgb_pred_gray": np.concatenate([rgb_u8, pred_gray3], axis=1),
        "stack_rgb_pred_color": np.concatenate([rgb_u8, pred_col], axis=1),
    }

    if gt_m is not None:
        gt_gray = gray_near_white(gt_m, vmin, vmax)
        gt_gray3 = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2RGB)
        gt_col = color_near_bright(gt_m, vmin, vmax)

        out.update({
            "gt_gray": gt_gray,
            "gt_color": gt_col,
            "stack_rgb_gt_pred_gray": np.concatenate([rgb_u8, gt_gray3, pred_gray3], axis=1),
            "stack_rgb_gt_pred_color": np.concatenate([rgb_u8, gt_col, pred_col], axis=1),
            "stack_full": np.concatenate([rgb_u8, gt_gray3, pred_gray3, gt_col, pred_col], axis=1),
        })

    return out


def save_prediction_pack(out_base: Path, rgb_u8: np.ndarray, pred_m: np.ndarray, gt_m: Optional[np.ndarray]):
    ensure_dir(out_base)

    save_png_rgb(out_base / "rgb.png", rgb_u8)
    save_png_u16(out_base / "pred_mm.png", depth_to_u16_mm(pred_m))

    stacks = make_stacks(rgb_u8, pred_m, gt_m)

    save_png_gray(out_base / "pred_gray.png", stacks["pred_gray"])
    save_png_rgb(out_base / "pred_color.png", stacks["pred_color"])
    save_png_rgb(out_base / "stack_rgb_pred_gray.png", stacks["stack_rgb_pred_gray"])
    save_png_rgb(out_base / "stack_rgb_pred_color.png", stacks["stack_rgb_pred_color"])

    if gt_m is not None:
        save_png_u16(out_base / "gt_mm.png", depth_to_u16_mm(gt_m))
        save_png_gray(out_base / "gt_gray.png", stacks["gt_gray"])
        save_png_rgb(out_base / "gt_color.png", stacks["gt_color"])
        save_png_rgb(out_base / "stack_rgb_gt_pred_gray.png", stacks["stack_rgb_gt_pred_gray"])
        save_png_rgb(out_base / "stack_rgb_gt_pred_color.png", stacks["stack_rgb_gt_pred_color"])
        save_png_rgb(out_base / "stack_full.png", stacks["stack_full"])


# -------------------------
# GT resolving (fast)
# -------------------------
class FastGTResolverAcrossBags:
    """
    Folder mode:
      Given rectified filename -> depth filename and search across all bags in depth_z16 (cached).
    """
    def __init__(self, raw_root: Path):
        self.depth_root = Path(raw_root) / "depth_z16"
        if not self.depth_root.exists():
            raise FileNotFoundError(f"depth_z16 not found under {raw_root}")
        self.bags = sorted([p for p in self.depth_root.iterdir() if p.is_dir()])
        self.cache: Dict[str, Optional[Path]] = {}

    def resolve(self, rectified_filename: str) -> Optional[Path]:
        dn = rectified_name_to_depth_name(rectified_filename)
        if dn is None:
            return None
        if dn in self.cache:
            return self.cache[dn]
        for b in self.bags:
            cand = b / dn
            if cand.exists():
                self.cache[dn] = cand
                return cand
        self.cache[dn] = None
        return None


# -------------------------
# index-mode pairing
# -------------------------
def discover_common_bags(raw_root: Path) -> List[str]:
    rect_root = raw_root / "rectified"
    depth_root = raw_root / "depth_z16"
    if not rect_root.exists() or not depth_root.exists():
        raise FileNotFoundError("raw_root must contain rectified/ and depth_z16/")
    rect_bags = {p.name for p in rect_root.iterdir() if p.is_dir()}
    depth_bags = {p.name for p in depth_root.iterdir() if p.is_dir()}
    common = sorted(rect_bags & depth_bags)
    if not common:
        raise RuntimeError("No common bags between rectified/ and depth_z16/")
    return common


def build_all_pairs(raw_root: Path) -> List[Tuple[Path, Path]]:
    rect_root = raw_root / "rectified"
    depth_root = raw_root / "depth_z16"
    bags = discover_common_bags(raw_root)

    pairs: List[Tuple[Path, Path]] = []
    for bag in bags:
        img_dir = rect_root / bag
        dep_dir = depth_root / bag
        for ip in sorted(img_dir.glob("rectified_idx*_t*.png")):
            dn = rectified_name_to_depth_name(ip.name)
            if dn is None:
                continue
            dp = dep_dir / dn
            if dp.exists():
                pairs.append((ip, dp))
    if not pairs:
        raise RuntimeError("No paired samples found for index mode.")
    return pairs


# -------------------------
# ZoeDepth loader + inference
# -------------------------
def choose_device(req: str) -> torch.device:
    if req == "cuda":
        return torch.device("cuda")
    if req == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_zoe_zeroshot(zoe_repo_root: Path, model_name: str, device: torch.device):
    zoe_repo_root = Path(zoe_repo_root)
    if not (zoe_repo_root / "hubconf.py").is_file():
        raise FileNotFoundError(f"hubconf.py not found in: {zoe_repo_root}")

    if str(zoe_repo_root) not in sys.path:
        sys.path.append(str(zoe_repo_root))

    print(f"-> Loading ZoeDepth zeroshot: {model_name} from {zoe_repo_root}")
    zoe = torch.hub.load(str(zoe_repo_root), model_name, source="local", pretrained=True)
    zoe = zoe.to(device).eval()

    # Patch drop_path if needed
    patched = 0
    for m in zoe.modules():
        if hasattr(m, "gamma_1") and not hasattr(m, "drop_path"):
            m.drop_path = nn.Identity()
            patched += 1
    if patched > 0:
        print(f"Patched drop_path on {patched} blocks.")

    return zoe


@torch.no_grad()
def zoe_infer_rgb_u8(zoe, rgb_u8: np.ndarray) -> np.ndarray:
    """
    rgb_u8: (H,W,3) uint8 RGB
    returns pred depth (H,W) float32 meters at same resolution (Zoe infer_pil)
    """
    pil = Image.fromarray(rgb_u8, mode="RGB")
    d = zoe.infer_pil(pil)  # meters
    d = np.asarray(d).astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d


def read_rgb_as_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# -------------------------
# modes
# -------------------------
def run_folder_mode(args, zoe):
    in_dir = Path(args.in_rgb_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"--in-rgb-dir not found: {in_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise RuntimeError(f"No images found in {in_dir}")

    out_root = ensure_dir(Path(args.out_dir) / "zoe_zeroshot" / args.model)

    gt_resolver = FastGTResolverAcrossBags(Path(args.raw_root)) if args.try_gt else None
    gt_found = 0

    print(f"[Folder mode] {len(files)} images")

    for p in files:
        rgb_u8 = read_rgb_as_u8(p)
        pr_m = zoe_infer_rgb_u8(zoe, rgb_u8)

        gt_m = None
        if gt_resolver is not None:
            dp = gt_resolver.resolve(p.name)
            if dp is not None:
                gt_raw = read_depth_mm_to_m(dp)
                # resize GT to match Zoe pred resolution
                gt_m = cv2.resize(gt_raw, (pr_m.shape[1], pr_m.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt_found += 1

        out_base = out_root / p.stem
        save_prediction_pack(out_base, rgb_u8, pr_m, gt_m)

    print(f"[OK] saved to: {out_root.resolve()}")
    if gt_resolver is not None:
        print(f"[Folder mode] GT found for {gt_found}/{len(files)}")


def run_index_mode(args, zoe):
    pairs = build_all_pairs(Path(args.raw_root))
    idx = max(0, min(int(args.idx), len(pairs) - 1))
    img_path, depth_path = pairs[idx]

    rgb_u8 = read_rgb_as_u8(img_path)
    pr_m = zoe_infer_rgb_u8(zoe, rgb_u8)

    gt_raw = read_depth_mm_to_m(depth_path)
    gt_m = cv2.resize(gt_raw, (pr_m.shape[1], pr_m.shape[0]), interpolation=cv2.INTER_NEAREST)

    out_root = ensure_dir(Path(args.out_dir) / "zoe_zeroshot" / args.model)
    out_base = out_root / img_path.stem
    save_prediction_pack(out_base, rgb_u8, pr_m, gt_m)

    print("[OK] saved:", out_base.resolve())
    print("img:", img_path)
    print("gt :", depth_path)


def run_bag_mode(args, zoe):
    raw_root = Path(args.raw_root)
    rect_dir = raw_root / "rectified" / args.bag
    depth_dir = raw_root / "depth_z16" / args.bag

    if not rect_dir.exists():
        raise FileNotFoundError(f"Bag rectified folder not found: {rect_dir}")
    if args.try_gt and not depth_dir.exists():
        raise FileNotFoundError(f"Bag depth folder not found: {depth_dir}")

    img_files = sorted(rect_dir.glob("rectified_idx*_t*.png"))
    if not img_files:
        raise RuntimeError(f"No rectified images in: {rect_dir}")

    if args.max_frames > 0:
        img_files = img_files[: int(args.max_frames)]

    out_root = ensure_dir(Path(args.out_dir) / "zoe_zeroshot" / args.model / "bags" / args.bag)
    clips_dir = ensure_dir(out_root / "clips")
    out_mp4 = clips_dir / f"{args.bag}.mp4"

    def gt_for_rectified(rect_name: str, pred_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        if not args.try_gt:
            return None
        dn = rectified_name_to_depth_name(rect_name)
        if dn is None:
            return None
        dp = depth_dir / dn
        if not dp.exists():
            return None
        gt_raw = read_depth_mm_to_m(dp)
        W, H = pred_hw
        return cv2.resize(gt_raw, (W, H), interpolation=cv2.INTER_NEAREST)

    vw = None
    print(f"[Bag mode] frames: {len(img_files)}")

    for p in img_files:
        rgb_u8 = read_rgb_as_u8(p)
        pr_m = zoe_infer_rgb_u8(zoe, rgb_u8)
        H, W = pr_m.shape

        gt_m = gt_for_rectified(p.name, (W, H))

        if args.save_frames:
            save_prediction_pack(out_root / p.stem, rgb_u8, pr_m, gt_m)

        if args.make_clip:
            stacks = make_stacks(rgb_u8, pr_m, gt_m)

            if gt_m is None:
                frame = np.concatenate([
                    rgb_u8,
                    cv2.cvtColor(stacks["pred_gray"], cv2.COLOR_GRAY2RGB),
                    stacks["pred_color"]
                ], axis=1)
            else:
                frame = stacks["stack_full"]

            if vw is None:
                Hf, Wf = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(out_mp4), fourcc, float(args.fps), (Wf, Hf))
                if not vw.isOpened():
                    raise RuntimeError("Failed to open VideoWriter (mp4v).")

            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if vw is not None:
        vw.release()
        print("[OK] clip saved:", out_mp4.resolve())

    print("[OK] bag outputs saved to:", out_root.resolve())


# -------------------------
# main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--raw-root", type=str, default="/home/sameer/Documents/raw_dataset_cpu_manual_1",
                   help="raw_dataset_cpu_manual_1 root (has rectified/ and depth_z16/)")
    p.add_argument("--zoe-repo-root", type=str, default="/home/sameer/Documents/Zoedepth_v1/ZoeDepth-1.0",
                   help="local ZoeDepth repo root (contains hubconf.py)")
    p.add_argument("--model", type=str, default="ZoeD_K", choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK"])

    p.add_argument("--out-dir", type=str, default="image_prediction/out_vis_zoe_zeroshot")

    # modes
    p.add_argument("--in-rgb-dir", type=str, default="image_prediction/input_img_to_pred")
    p.add_argument("--idx", type=int, default=-1)
    p.add_argument("--bag", type=str, default="")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # GT
    p.add_argument("--try-gt", action="store_true",
                   help="try to locate GT in depth_z16 and include GT views if found")

    # bag extras
    p.add_argument("--make-clip", action="store_true")
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--max-frames", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    print("Device:", device)

    zoe = load_zoe_zeroshot(Path(args.zoe_repo_root), args.model, device=device)

    # Priority: bag > idx > folder
    if args.bag.strip():
        run_bag_mode(args, zoe)
        return

    if int(args.idx) >= 0:
        run_index_mode(args, zoe)
        return

    if not args.in_rgb_dir:
        raise RuntimeError("Provide one of: --bag, --idx, or --in-rgb-dir")

    run_folder_mode(args, zoe)


if __name__ == "__main__":
    main()
