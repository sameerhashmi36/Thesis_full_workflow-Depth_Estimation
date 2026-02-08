"""
predicting_depth_maps_monodepth2_zeroshot.py

Monodepth2 ZERO-SHOT predicting/visualization script for AGCO.

It does NOT compute metrics. It only:
- runs monodepth2 inference
- converts disp -> metric depth (with disp_to_depth)
- saves prediction outputs as images
- optionally finds GT depth and makes side-by-side stacks
- optionally makes an mp4 clip for a bag

Modes (priority):
1) --bag <bag_name> : predict for raw_root/rectified/<bag>
2) --idx N          : pick Nth paired sample from ALL common bags and visualize it
3) --in-rgb-dir DIR : predict for images inside a folder

Outputs for each image (out_dir/.../<image_stem>/):
  rgb.png
  pred_m.png (uint16 mm png)
  pred_gray.png (near=white)
  pred_color.png (JET)
  stack_rgb_pred_gray.png
  stack_rgb_pred_color.png

If GT exists (--try-gt):
  gt_m.png (uint16 mm png)
  gt_gray.png
  gt_color.png
  stack_rgb_gt_pred_gray.png  (RGB | GT gray | Pred gray)
  stack_rgb_gt_pred_color.png (RGB | GT color | Pred color)
  stack_full.png              (RGB | GT gray | Pred gray | GT color | Pred color)

Requirements:
- Run this from monodepth2 repo root OR make sure imports work:
    import networks
    from layers import disp_to_depth

"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


sys.path.append("path/to/repo/monodepth2")

import networks
from layers import disp_to_depth


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
def read_rgb_as_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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
# visualization
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
    return 255 - g  # near white


def colorize_depth_jet(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    d = depth_m.astype(np.float32).copy()
    d[~np.isfinite(d)] = vmin
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin + 1e-8)
    img8 = (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)
    cm_bgr = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    return cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)


def make_stacks(rgb_u8: np.ndarray, pred_m: np.ndarray, gt_m: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    # Use GT range if available else pred
    if gt_m is not None and np.any(gt_m > 0):
        vmin, vmax = robust_vmin_vmax(gt_m)
    else:
        vmin, vmax = robust_vmin_vmax(pred_m)

    pred_gray = gray_near_white(pred_m, vmin, vmax)
    pred_gray3 = cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2RGB)
    pred_col = colorize_depth_jet(pred_m, vmin, vmax)

    out = {
        "pred_gray": pred_gray,
        "pred_color": pred_col,
        "stack_rgb_pred_gray": np.concatenate([rgb_u8, pred_gray3], axis=1),
        "stack_rgb_pred_color": np.concatenate([rgb_u8, pred_col], axis=1),
    }

    if gt_m is not None:
        gt_gray = gray_near_white(gt_m, vmin, vmax)
        gt_gray3 = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2RGB)
        gt_col = colorize_depth_jet(gt_m, vmin, vmax)

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
    save_png_u16(out_base / "pred_m.png", depth_to_u16_mm(pred_m))

    stacks = make_stacks(rgb_u8, pred_m, gt_m)
    save_png_gray(out_base / "pred_gray.png", stacks["pred_gray"])
    save_png_rgb(out_base / "pred_color.png", stacks["pred_color"])
    save_png_rgb(out_base / "stack_rgb_pred_gray.png", stacks["stack_rgb_pred_gray"])
    save_png_rgb(out_base / "stack_rgb_pred_color.png", stacks["stack_rgb_pred_color"])

    if gt_m is not None:
        save_png_u16(out_base / "gt_m.png", depth_to_u16_mm(gt_m))
        save_png_gray(out_base / "gt_gray.png", stacks["gt_gray"])
        save_png_rgb(out_base / "gt_color.png", stacks["gt_color"])
        save_png_rgb(out_base / "stack_rgb_gt_pred_gray.png", stacks["stack_rgb_gt_pred_gray"])
        save_png_rgb(out_base / "stack_rgb_gt_pred_color.png", stacks["stack_rgb_gt_pred_color"])
        save_png_rgb(out_base / "stack_full.png", stacks["stack_full"])


# -------------------------
# GT resolving (folder mode)
# -------------------------
class FastGTResolverAcrossBags:
    """
    For --in-rgb-dir mode:
      rectified_idxX_t*.png -> depth_idx00000X_t*.png
      Search across raw_root/depth_z16/<any bag> and cache.
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
# monodepth2 model + inference
# -------------------------
def choose_device(req: str) -> torch.device:
    if req == "cuda":
        return torch.device("cuda")
    if req == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_monodepth2(weights_folder: Path, device: torch.device):
    weights_folder = Path(weights_folder)
    enc_p = weights_folder / "encoder.pth"
    dep_p = weights_folder / "depth.pth"
    if not enc_p.exists() or not dep_p.exists():
        raise FileNotFoundError(f"Missing encoder.pth/depth.pth in {weights_folder}")

    print(f"-> Loading monodepth2 weights from {weights_folder}")

    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    enc_state = torch.load(str(enc_p), map_location=device)
    enc_state = {k: v for k, v in enc_state.items() if k in encoder.state_dict()}
    encoder.load_state_dict(enc_state)

    dec_state = torch.load(str(dep_p), map_location=device)
    decoder.load_state_dict(dec_state)

    encoder.to(device).eval()
    decoder.to(device).eval()

    return encoder, decoder


@torch.no_grad()
def infer_monodepth2_depth_m(
    encoder,
    decoder,
    rgb_u8: np.ndarray,
    device: torch.device,
    in_w: int,
    in_h: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """
    rgb_u8: (H,W,3) uint8 (any size)
    returns: pred depth (in_h,in_w) float32 meters (monodepth2 output resolution)
    """
    rgb_rs = cv2.resize(rgb_u8, (in_w, in_h), interpolation=cv2.INTER_AREA)
    x = (rgb_rs.astype(np.float32) / 255.0)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

    feats = encoder(x)
    out = decoder(feats)
    disp = out[("disp", 0)]  # (1,1,H,W)

    _, depth = disp_to_depth(disp, min_depth, max_depth)  # meters
    depth = depth[0, 0].detach().cpu().numpy().astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    return depth


# -------------------------
# modes
# -------------------------
def run_folder_mode(args, encoder, decoder, device):
    in_dir = Path(args.in_rgb_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"--in-rgb-dir not found: {in_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise RuntimeError(f"No images found in {in_dir}")

    out_root = ensure_dir(Path(args.out_dir) / "monodepth2_zeroshot" / "mono_640x192")
    gt_resolver = FastGTResolverAcrossBags(Path(args.raw_root)) if args.try_gt else None
    gt_found = 0

    print(f"[Folder mode] {len(files)} images")

    for p in files:
        rgb_u8 = read_rgb_as_u8(p)
        pred_m = infer_monodepth2_depth_m(
            encoder, decoder, rgb_u8, device,
            in_w=args.in_w, in_h=args.in_h,
            min_depth=args.min_depth, max_depth=args.max_depth
        )

        gt_m = None
        if gt_resolver is not None:
            dp = gt_resolver.resolve(p.name)
            if dp is not None:
                gt_raw = read_depth_mm_to_m(dp)
                gt_m = cv2.resize(gt_raw, (pred_m.shape[1], pred_m.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt_found += 1

        save_prediction_pack(out_root / p.stem, rgb_u8=cv2.resize(rgb_u8, (pred_m.shape[1], pred_m.shape[0])), pred_m=pred_m, gt_m=gt_m)

    print(f"[OK] saved to: {out_root.resolve()}")
    if gt_resolver is not None:
        print(f"[Folder mode] GT found for {gt_found}/{len(files)}")


def run_index_mode(args, encoder, decoder, device):
    pairs = build_all_pairs(Path(args.raw_root))
    idx = max(0, min(int(args.idx), len(pairs) - 1))
    img_path, depth_path = pairs[idx]

    rgb_u8_full = read_rgb_as_u8(img_path)
    pred_m = infer_monodepth2_depth_m(
        encoder, decoder, rgb_u8_full, device,
        in_w=args.in_w, in_h=args.in_h,
        min_depth=args.min_depth, max_depth=args.max_depth
    )

    gt_raw = read_depth_mm_to_m(depth_path)
    gt_m = cv2.resize(gt_raw, (pred_m.shape[1], pred_m.shape[0]), interpolation=cv2.INTER_NEAREST)

    out_root = ensure_dir(Path(args.out_dir) / "monodepth2_zeroshot" / "mono_640x192")
    out_base = out_root / img_path.stem

    rgb_u8 = cv2.resize(rgb_u8_full, (pred_m.shape[1], pred_m.shape[0]), interpolation=cv2.INTER_AREA)
    save_prediction_pack(out_base, rgb_u8=rgb_u8, pred_m=pred_m, gt_m=gt_m)

    print("[OK] saved:", out_base.resolve())
    print("img:", img_path)
    print("gt :", depth_path)


def run_bag_mode(args, encoder, decoder, device):
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

    out_root = ensure_dir(Path(args.out_dir) / "monodepth2_zeroshot" / "mono_640x192" / "bags" / args.bag)
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
        rgb_u8_full = read_rgb_as_u8(p)
        pred_m = infer_monodepth2_depth_m(
            encoder, decoder, rgb_u8_full, device,
            in_w=args.in_w, in_h=args.in_h,
            min_depth=args.min_depth, max_depth=args.max_depth
        )
        H, W = pred_m.shape
        rgb_u8 = cv2.resize(rgb_u8_full, (W, H), interpolation=cv2.INTER_AREA)
        gt_m = gt_for_rectified(p.name, (W, H))

        if args.save_frames:
            save_prediction_pack(out_root / p.stem, rgb_u8, pred_m, gt_m)

        if args.make_clip:
            stacks = make_stacks(rgb_u8, pred_m, gt_m)

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

    p.add_argument("--raw-root", type=str, default="/path/to/dataset/raw_dataset_cpu_manual_1",
                   help="raw_dataset_cpu_manual_1 root (has rectified/ and depth_z16/)")
    p.add_argument("--weights-folder", type=str, default="models/mono_640x192",
                   help="monodepth2 weights folder containing encoder.pth and depth.pth")
    p.add_argument("--out-dir", type=str, default="image_prediction/out_vis_mono_zeroshot")

    # modes
    p.add_argument("--bag", type=str, default="")
    p.add_argument("--idx", type=int, default=-1)
    p.add_argument("--in-rgb-dir", type=str, default="image_prediction/input_img_to_pred")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # model input size (mono_640x192)
    p.add_argument("--in-w", type=int, default=640)
    p.add_argument("--in-h", type=int, default=192)

    # depth clamp (monodepth2 disp_to_depth)
    p.add_argument("--min-depth", type=float, default=1e-3)
    p.add_argument("--max-depth", type=float, default=80.0)

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

    encoder, decoder = load_monodepth2(Path(args.weights_folder), device=device)

    # Priority: bag > idx > folder
    if args.bag.strip():
        run_bag_mode(args, encoder, decoder, device)
        return

    if int(args.idx) >= 0:
        run_index_mode(args, encoder, decoder, device)
        return

    if not args.in_rgb_dir:
        raise RuntimeError("Provide one of: --bag, --idx, or --in-rgb-dir")

    run_folder_mode(args, encoder, decoder, device)


if __name__ == "__main__":
    main()
