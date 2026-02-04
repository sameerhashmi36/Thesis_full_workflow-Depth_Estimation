"""
predicting_depth_maps_da3_zeroshot.py

DA3 ZERO-SHOT visualization.

Uses official DA3 API:
  DepthAnything3.from_pretrained(model_id).inference(list_of_img_paths)

Modes:
1) Index mode: --idx N
   - Builds ALL paired samples across all common bags (rectified + depth_z16).
   - Saves RGB + GT + Pred.

2) Folder mode: --in-rgb-dir <folder>
   - Predicts for each image file.
   - If image name matches rectified_idxX_t*.png, tries fast GT lookup across bags.
   - Saves RGB + Pred (+ GT if found).

3) Bag mode: --bag <bag_name>
   - Predicts for all images in raw_root/rectified/<bag>.
   - Writes per-frame outputs and optional MP4:
       RGB | GT(gray) | Pred(gray) | GT(color) | Pred(color)

Coloring:
- gray: near = white, far = black (percentile robust)
- color: near bright, far dark using COLORMAP_HOT (OpenCV 3.3.1 friendly)

"""

import sys
import argparse
from pathlib import Path
import re
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch

# DA3 repo path for local import
DEFAULT_DA3_REPO_ROOT = Path("/path/to/repo/root/Depth-Anything-3")

RECT_RE = re.compile(r"^rectified_idx(\d+)(_t.*)\.png$", re.IGNORECASE)

# -------------------------
# helpers
# -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

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

def robust_vmin_vmax(depth_m: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
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
    Near=bright, far=dark using COLORMAP_HOT (OpenCV 3.3.1).
    """
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin + 1e-8)   # near->0
    inv = 1.0 - np.clip(norm, 0.0, 1.0)        # near->1
    img8 = (inv * 255.0).astype(np.uint8)
    cm_bgr = cv2.applyColorMap(img8, cv2.COLORMAP_HOT)
    return cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)

def make_stacks(rgb_u8: np.ndarray, pred_m: np.ndarray, gt_m: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    # choose vmin/vmax from GT if available else pred
    if gt_m is not None and np.any(gt_m > 0):
        mask = (gt_m > 0) & np.isfinite(gt_m)
        vmin, vmax = robust_vmin_vmax(gt_m, mask)
    else:
        mask = (pred_m > 0) & np.isfinite(pred_m)
        vmin, vmax = robust_vmin_vmax(pred_m, mask)

    pred_gray = depth_to_gray_near_white(pred_m, vmin, vmax)
    pred_col = colorize_depth_near_bright(pred_m, vmin, vmax)
    pred_gray3 = cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2RGB)

    out = {
        "pred_gray": pred_gray,
        "pred_color": pred_col,
        "stack_rgb_pred_gray": np.concatenate([rgb_u8, pred_gray3], axis=1),
        "stack_rgb_pred_color": np.concatenate([rgb_u8, pred_col], axis=1),
    }

    if gt_m is not None:
        gt_gray = depth_to_gray_near_white(gt_m, vmin, vmax)
        gt_col = colorize_depth_near_bright(gt_m, vmin, vmax)
        gt_gray3 = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2RGB)

        out.update({
            "gt_gray": gt_gray,
            "gt_color": gt_col,
            "stack_rgb_gt_pred_gray": np.concatenate([rgb_u8, gt_gray3, pred_gray3], axis=1),
            "stack_rgb_gt_pred_color": np.concatenate([rgb_u8, gt_col, pred_col], axis=1),
            "stack_full": np.concatenate([rgb_u8, gt_gray3, pred_gray3, gt_col, pred_col], axis=1),
        })
    return out


# -------------------------
# GT resolvers
# -------------------------
class FastGTResolverAcrossBags:
    """
    Folder mode GT resolver:
    given rectified filename, compute depth filename and search across all bag dirs in depth_z16.
    caches by depth filename.
    """
    def __init__(self, raw_root: Path):
        self.depth_root = Path(raw_root) / "depth_z16"
        if not self.depth_root.exists():
            raise FileNotFoundError(f"depth_z16 not found under {raw_root}")
        self.bags = sorted([p for p in self.depth_root.iterdir() if p.is_dir()])
        self.cache: Dict[str, Optional[Path]] = {}

    def resolve(self, rectified_filename: str) -> Optional[Path]:
        depth_name = rectified_name_to_depth_name(rectified_filename)
        if depth_name is None:
            return None
        if depth_name in self.cache:
            return self.cache[depth_name]
        for b in self.bags:
            cand = b / depth_name
            if cand.exists():
                self.cache[depth_name] = cand
                return cand
        self.cache[depth_name] = None
        return None


def discover_common_bags(raw_root: Path) -> List[str]:
    rect_root = raw_root / "rectified"
    depth_root = raw_root / "depth_z16"
    rect_bags = {p.name for p in rect_root.iterdir() if p.is_dir()}
    depth_bags = {p.name for p in depth_root.iterdir() if p.is_dir()}
    common = sorted(rect_bags & depth_bags)
    if not common:
        raise RuntimeError("No common bags found between rectified/ and depth_z16/")
    return common

def build_all_pairs(raw_root: Path) -> List[Tuple[Path, Path]]:
    rect_root = raw_root / "rectified"
    depth_root = raw_root / "depth_z16"
    bags = discover_common_bags(raw_root)

    pairs: List[Tuple[Path, Path]] = []
    for bag in bags:
        img_dir = rect_root / bag
        dep_dir = depth_root / bag
        img_files = sorted(img_dir.glob("rectified_idx*_t*.png"))
        for ip in img_files:
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
# DA3 model
# -------------------------
def load_da3(model_id: str, da3_repo_root: Path, device: torch.device):
    if da3_repo_root.exists() and str(da3_repo_root) not in sys.path:
        sys.path.append(str(da3_repo_root))
    from depth_anything_3.api import DepthAnything3
    print(f"-> Loading DA3 model: {model_id}")
    m = DepthAnything3.from_pretrained(model_id).to(device=device)
    m.eval()
    return m

def da3_infer_paths(model, img_paths: List[str]):
    """
    model.inference(img_paths) returns:
      prediction.processed_images: (N,H,W,3) uint8 RGB
      prediction.depth          : (N,H,W) float32 meters
    """
    pred = model.inference(img_paths)
    if pred is None or pred.depth is None or pred.processed_images is None:
        raise RuntimeError("DA3 inference returned None outputs.")
    return pred.processed_images, pred.depth


# -------------------------
# saving per-frame
# -------------------------
def save_prediction_pack(out_base: Path, rgb_u8: np.ndarray, pred_m: np.ndarray, gt_m: Optional[np.ndarray]):
    ensure_dir(out_base)

    save_png_rgb(out_base / "rgb.png", rgb_u8)
    save_png_u16(out_base / "pred_mm.png", depth_to_u16_mm(pred_m))

    if gt_m is not None:
        save_png_u16(out_base / "gt_mm.png", depth_to_u16_mm(gt_m))

    stacks = make_stacks(rgb_u8, pred_m, gt_m)

    save_png_gray(out_base / "pred_gray.png", stacks["pred_gray"])
    save_png_rgb(out_base / "pred_color.png", stacks["pred_color"])
    save_png_rgb(out_base / "stack_rgb_pred_gray.png", stacks["stack_rgb_pred_gray"])
    save_png_rgb(out_base / "stack_rgb_pred_color.png", stacks["stack_rgb_pred_color"])

    if gt_m is not None:
        save_png_gray(out_base / "gt_gray.png", stacks["gt_gray"])
        save_png_rgb(out_base / "gt_color.png", stacks["gt_color"])
        save_png_rgb(out_base / "stack_rgb_gt_pred_gray.png", stacks["stack_rgb_gt_pred_gray"])
        save_png_rgb(out_base / "stack_rgb_gt_pred_color.png", stacks["stack_rgb_gt_pred_color"])
        save_png_rgb(out_base / "stack_full.png", stacks["stack_full"])


# -------------------------
# modes
# -------------------------
def run_folder_mode(args, model, device):
    in_dir = Path(args.in_rgb_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"--in-rgb-dir not found: {in_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise RuntimeError(f"No images found in {in_dir}")

    out_root = ensure_dir(Path(args.out_dir) / "da3_zeroshot" / args.model_id.replace("/", "__"))
    gt_resolver = FastGTResolverAcrossBags(Path(args.raw_root)) if args.try_gt else None

    print(f"[Folder mode] {len(files)} images")
    gt_found = 0

    # batch inference
    for start in range(0, len(files), args.batch_size):
        batch = files[start:start + args.batch_size]
        img_paths = [str(p) for p in batch]

        rgb_proc, pred_depth = da3_infer_paths(model, img_paths)  # rgb_proc uint8, pred_depth float
        N = pred_depth.shape[0]

        for i in range(N):
            rgb_u8 = rgb_proc[i]
            pr_m = pred_depth[i].astype(np.float32)

            gt_m = None
            if gt_resolver is not None:
                dp = gt_resolver.resolve(Path(img_paths[i]).name)
                if dp is not None:
                    gt_m = read_depth_mm_to_m_resize(dp, pr_m.shape[1], pr_m.shape[0])
                    gt_found += 1

            out_base = out_root / Path(img_paths[i]).stem
            save_prediction_pack(out_base, rgb_u8, pr_m, gt_m)

    print(f"[OK] saved to: {out_root.resolve()}")
    if gt_resolver is not None:
        print(f"[Folder mode] GT found for {gt_found}/{len(files)}")


def run_index_mode(args, model, device):
    raw_root = Path(args.raw_root)
    pairs = build_all_pairs(raw_root)
    idx = max(0, min(int(args.idx), len(pairs) - 1))
    img_path, depth_path = pairs[idx]

    rgb_proc, pred_depth = da3_infer_paths(model, [str(img_path)])
    rgb_u8 = rgb_proc[0]
    pr_m = pred_depth[0].astype(np.float32)

    # resize GT to pred resolution
    gt_m = read_depth_mm_to_m_resize(depth_path, pr_m.shape[1], pr_m.shape[0])

    out_root = ensure_dir(Path(args.out_dir) / "da3_zeroshot" / args.model_id.replace("/", "__"))
    out_base = out_root / img_path.stem
    save_prediction_pack(out_base, rgb_u8, pr_m, gt_m)

    print("[OK] saved:", out_base.resolve())
    print("img:", img_path)
    print("gt :", depth_path)


def run_bag_mode(args, model, device):
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

    out_root = ensure_dir(Path(args.out_dir) / "da3_zeroshot" / args.model_id.replace("/", "__") / "bags" / args.bag)
    clips_dir = ensure_dir(out_root / "clips")
    out_mp4 = clips_dir / f"{args.bag}.mp4"

    def depth_path_for_rectified(name: str) -> Optional[Path]:
        dn = rectified_name_to_depth_name(name)
        if dn is None:
            return None
        dp = depth_dir / dn
        return dp if dp.exists() else None

    # Prepare video writer after first frame
    vw = None

    print(f"[Bag mode] frames: {len(img_files)}")

    for start in range(0, len(img_files), args.batch_size):
        batch = img_files[start:start + args.batch_size]
        img_paths = [str(p) for p in batch]
        rgb_proc, pred_depth = da3_infer_paths(model, img_paths)

        for i in range(pred_depth.shape[0]):
            p_img = Path(img_paths[i])
            rgb_u8 = rgb_proc[i]
            pr_m = pred_depth[i].astype(np.float32)

            gt_m = None
            if args.try_gt:
                dp = depth_path_for_rectified(p_img.name)
                if dp is not None:
                    gt_m = read_depth_mm_to_m_resize(dp, pr_m.shape[1], pr_m.shape[0])

            # save per-frame pack (optional)
            if args.save_frames:
                save_prediction_pack(out_root / p_img.stem, rgb_u8, pr_m, gt_m)

            # video frame
            if args.make_clip:
                stacks = make_stacks(rgb_u8, pr_m, gt_m)
                if gt_m is None:
                    # RGB | Pred(gray) | Pred(color)
                    frame = np.concatenate([rgb_u8,
                                            cv2.cvtColor(stacks["pred_gray"], cv2.COLOR_GRAY2RGB),
                                            stacks["pred_color"]], axis=1)
                else:
                    # RGB | GT(gray) | Pred(gray) | GT(color) | Pred(color)
                    frame = stacks["stack_full"]

                if vw is None:
                    H, W = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(str(out_mp4), fourcc, float(args.fps), (W, H))
                    if not vw.isOpened():
                        raise RuntimeError("Failed to open VideoWriter (mp4v).")

                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if vw is not None:
        vw.release()
        print("[OK] clip saved:", out_mp4.resolve())

    print("[OK] outputs saved to:", out_root.resolve())


# -------------------------
# main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str, default="/path/to/dataset/raw_dataset_cpu_manual_1",
                   help="raw_dataset_cpu_manual_1 root (has rectified/ and depth_z16/)")
    p.add_argument("--da3-repo-root", type=str, default=str(DEFAULT_DA3_REPO_ROOT))
    p.add_argument("--model-id", type=str, default="depth-anything/DA3-LARGE")
    p.add_argument("--out-dir", type=str, default="image_prediction/out_vis_da3_zeroshot")

    # modes
    p.add_argument("--in-rgb-dir", type=str, default="image_prediction/input_img_to_pred/")
    p.add_argument("--idx", type=int, default=-1)
    p.add_argument("--bag", type=str, default="")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # GT
    p.add_argument("--try-gt", action="store_true",
                   help="try to locate GT depth_z16 and include GT views if found")

    # bag extras
    p.add_argument("--make-clip", action="store_true")
    p.add_argument("--save-frames", action="store_true",
                   help="save per-frame folders in bag mode (can be heavy)")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--max-frames", type=int, default=0)

    return p.parse_args()

def choose_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args()
    device = choose_device(args.device)
    print("Device:", device)

    model = load_da3(args.model_id, Path(args.da3_repo_root), device=device)

    # priority: bag > idx > folder
    if args.bag.strip():
        run_bag_mode(args, model, device)
        return

    if int(args.idx) >= 0:
        run_index_mode(args, model, device)
        return

    if not args.in_rgb_dir:
        raise RuntimeError("Provide either --in-rgb-dir, or --idx, or --bag.")

    run_folder_mode(args, model, device)

if __name__ == "__main__":
    main()
