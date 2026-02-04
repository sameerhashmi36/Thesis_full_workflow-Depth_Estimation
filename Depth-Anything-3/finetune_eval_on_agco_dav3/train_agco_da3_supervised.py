"""
Supervised finetuning of Depth Anything v3 (DA3) on AGCO rectified + LiDAR depth_z16.

Goals:
- Train DA3 (network = da3.model) using masked regression loss on sparse LiDAR depth
- Validate every epoch with Monodepth-style metrics:
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
- Keep DA3 input constraints stable by padding to a patch-size multiple, then cropping back

Why metrics differ across repos:
- Training losses often use L1/log-L1 (good for optimization)
- Reported metrics (abs_rel/a1/a2/a3) are evaluation conventions (Monodepth/KITTI-style)
- Here, we compute those in validation to match Monodepth2 / ZoeDepth reporting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from agco_da3_config import (
    DEFAULT_RAW_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_DA3_REPO_ROOT, DEFAULT_DA3_MODEL_ID,
    MIN_DEPTH_M, MAX_DEPTH_M, TRAIN_H, TRAIN_W, SPLIT_SEED
)
# from agco_da3_dataset import AGCODA3DepthDataset
from agco_da3_dataset_smooth import AGCODA3DepthDatasetSmooth as AGCODA3DepthDataset


# ----------------- helpers -----------------

def choose_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def pad_to_multiple_bschw(x: torch.Tensor, mult: int = 14):
    """
    Pad to make H and W divisible by mult.

    Accepts:
      - (B,C,H,W)
      - (B,S,C,H,W)

    Returns:
      x_pad, orig_hw
    """
    if x.dim() == 4:
        _, _, H, W = x.shape
        pad_h = (mult - (H % mult)) % mult
        pad_w = (mult - (W % mult)) % mult
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x_pad, (H, W)

    if x.dim() == 5:
        B, S, C, H, W = x.shape
        pad_h = (mult - (H % mult)) % mult
        pad_w = (mult - (W % mult)) % mult
        xs = x.reshape(B * S, C, H, W)
        xs_pad = F.pad(xs, (0, pad_w, 0, pad_h), mode="replicate")
        Hp, Wp = xs_pad.shape[-2], xs_pad.shape[-1]
        x_pad = xs_pad.reshape(B, S, C, Hp, Wp)
        return x_pad, (H, W)

    raise ValueError(f"Unexpected input rank: {x.dim()} (expected 4D or 5D)")


def crop_back(pred: torch.Tensor, orig_hw):
    H, W = orig_hw
    return pred[..., :H, :W]


def forward_da3_metric(da3_net, x_bchw: torch.Tensor, patch_mult: int = 14) -> torch.Tensor:
    """
    Forward for DA3 finetuning.

    - Convert (B,C,H,W) -> (B,1,C,H,W) because DA3 API often expects a temporal dim
    - Pad to multiple of patch size
    - Call da3_net(x, extrinsics, intrinsics, export_feat_layers, infer_gs)
      export_feat_layers must be iterable, so use empty tuple ()
    - Normalize output to (B,1,H,W)
    - Crop back to original size
    """
    if x_bchw.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(x_bchw.shape)}")

    x = x_bchw.unsqueeze(1)  # (B,1,C,H,W)
    x_pad, orig_hw = pad_to_multiple_bschw(x, mult=patch_mult)

    out = da3_net(x_pad, None, None, (), False)

    if isinstance(out, dict):
        for k in ["metric_depth", "depth", "pred", "out"]:
            if k in out:
                out = out[k]
                break
    if isinstance(out, (list, tuple)):
        out = out[0]

    # normalize to (B,1,h,w)
    if out.dim() == 5:          # (B,S,1,h,w)
        out = out[:, 0]
    elif out.dim() == 4:
        if out.shape[1] != 1:   # (B,S,h,w)
            out = out[:, 0].unsqueeze(1)
    elif out.dim() == 3:
        out = out.unsqueeze(1)

    out = crop_back(out, orig_hw)
    return out


def compute_errors(gt: np.ndarray, pred: np.ndarray):
    """
    Monodepth/KITTI-style metrics.

    gt, pred are 1D arrays of valid pixels in meters.
    """
    pred = np.clip(pred, 1e-8, None)
    gt = np.clip(gt, 1e-8, None)

    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# ----------------- args -----------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--da3-repo-root", type=str, default=str(DEFAULT_DA3_REPO_ROOT))
    p.add_argument("--model-id", type=str, default=DEFAULT_DA3_MODEL_ID)

    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.5)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--val-bag-fraction", type=float, default=0.2)

    p.add_argument("--img-height", type=int, default=TRAIN_H)
    p.add_argument("--img-width", type=int, default=TRAIN_W)

    p.add_argument("--min-depth", type=float, default=MIN_DEPTH_M)
    p.add_argument("--max-depth", type=float, default=MAX_DEPTH_M)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--w-l1", type=float, default=1.0)
    p.add_argument("--w-log", type=float, default=0.2)

    p.add_argument("--median-scaling", action="store_true")

    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--resume", type=str, default="")  # state_dict .pth

    return p.parse_args()


# ----------------- main -----------------

def main():
    args = parse_args()

    da3_repo_root = Path(args.da3_repo_root)
    if da3_repo_root.exists() and str(da3_repo_root) not in sys.path:
        sys.path.append(str(da3_repo_root))

    from depth_anything_3.api import DepthAnything3

    device = choose_device()
    print("Device:", device)
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"CUDA capability sm_{cap[0]}{cap[1]}")

    train_ds = AGCODA3DepthDataset(
        raw_root=args.raw_root,
        split="train",
        img_height=args.img_height,
        img_width=args.img_width,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        val_bag_fraction=args.val_bag_fraction,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        verbose=True,
    )
    val_ds = AGCODA3DepthDataset(
        raw_root=args.raw_root,
        split="val",
        img_height=args.img_height,
        img_width=args.img_width,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        val_bag_fraction=args.val_bag_fraction,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        verbose=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    if args.run_tag:
        tag = args.run_tag
    else:
        base = args.model_id.replace("/", "_").replace(":", "_")
        # fractions as percent
        tr = int(round(args.train_fraction * 100))
        va = int(round(args.val_fraction * 100))
        tag = f"{base}_e{args.epochs}_train{tr:02d}_val{va:02d}"
        
    run_dir = Path(args.output_root) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.txt"

    def log_line(s: str):
        print(s)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    log_line(f"[RUN] dir={run_dir}")
    log_line(f"[RUN] model_id={args.model_id}")
    log_line(f"[RUN] depth_mask=[{args.min_depth},{args.max_depth}] m")
    log_line(f"[RUN] median_scaling={args.median_scaling}")

    da3 = DepthAnything3.from_pretrained(args.model_id).to(device)
    net = da3.model.to(device)  # trainable network

    if args.resume:
        ckpt = Path(args.resume)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
        state = torch.load(str(ckpt), map_location=device)
        net.load_state_dict(state, strict=True)
        log_line(f"[RESUME] strict=True load OK: {ckpt}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_absrel = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # -------- train --------
        net.train()
        losses, l1s, logls = [], [], []

        pbar = tqdm(train_loader, desc=f"[Train e{epoch}/{args.epochs}]", ncols=110)
        for rgb, depth_gt, valid_mask, _meta in pbar:
            rgb = rgb.to(device, non_blocking=True)                  # (B,3,H,W)
            depth_gt = depth_gt.to(device, non_blocking=True)        # (B,H,W) meters
            valid_mask = valid_mask.to(device, non_blocking=True).bool()

            pred = forward_da3_metric(net, rgb, patch_mult=14)        # (B,1,h,w)
            Hp, Wp = pred.shape[-2:]

            gt = depth_gt.unsqueeze(1)
            vm = valid_mask.unsqueeze(1)

            if gt.shape[-2:] != (Hp, Wp):
                gt = F.interpolate(gt, size=(Hp, Wp), mode="nearest")
                vm = F.interpolate(vm.float(), size=(Hp, Wp), mode="nearest").bool()

            # mask = valid GT pixels within depth range + finite pred
            mask = vm & (gt > args.min_depth) & (gt < args.max_depth) & torch.isfinite(pred)
            if mask.sum().item() == 0:
                continue

            pred_c = torch.clamp(pred, args.min_depth, args.max_depth)
            gt_c = torch.clamp(gt, args.min_depth, args.max_depth)

            l1 = torch.abs(pred_c - gt_c)[mask].mean()
            log_l = torch.abs(torch.log(pred_c) - torch.log(gt_c))[mask].mean()
            loss = args.w_l1 * l1 + args.w_log * log_l

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()

            losses.append(float(loss.item()))
            l1s.append(float(l1.item()))
            logls.append(float(log_l.item()))
            pbar.set_postfix(loss=np.mean(losses), l1=np.mean(l1s), log=np.mean(logls))

        log_line(f"[Train e{epoch}] loss={np.mean(losses):.4f} l1={np.mean(l1s):.4f} log={np.mean(logls):.4f}")

        # -------- val (Monodepth-style metrics) --------
        net.eval()
        all_err = []

        with torch.no_grad():
            for rgb, depth_gt, valid_mask, _meta in tqdm(val_loader, desc="[Val]", ncols=110):
                rgb = rgb.to(device, non_blocking=True)
                depth_gt = depth_gt.to(device, non_blocking=True)
                valid_mask = valid_mask.to(device, non_blocking=True).bool()

                pred = forward_da3_metric(net, rgb, patch_mult=14)  # (1,1,h,w)
                Hp, Wp = pred.shape[-2:]

                gt = depth_gt.unsqueeze(1)
                vm = valid_mask.unsqueeze(1)
                if gt.shape[-2:] != (Hp, Wp):
                    gt = F.interpolate(gt, size=(Hp, Wp), mode="nearest")
                    vm = F.interpolate(vm.float(), size=(Hp, Wp), mode="nearest").bool()

                mask = vm & (gt > args.min_depth) & (gt < args.max_depth) & torch.isfinite(pred)
                if mask.sum().item() == 0:
                    continue

                gt_np = gt[0, 0].cpu().numpy().astype(np.float32)
                pr_np = pred[0, 0].cpu().numpy().astype(np.float32)
                m_np = mask[0, 0].cpu().numpy().astype(bool)

                gt_flat = gt_np[m_np]
                pr_flat = pr_np[m_np]

                if args.median_scaling:
                    scale = np.median(gt_flat) / (np.median(pr_flat) + 1e-8)
                else:
                    scale = 1.0

                pr_scaled = np.clip(pr_flat * scale, args.min_depth, args.max_depth)
                all_err.append(compute_errors(gt_flat, pr_scaled))

        if all_err:
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = np.mean(np.array(all_err), axis=0)
        else:
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = (float("inf"),) * 7

        log_line(
            f"[Val e{epoch}] abs_rel={abs_rel:.4f} sq_rel={sq_rel:.4f} rmse={rmse:.3f} rmse_log={rmse_log:.3f} "
            f"a1={a1:.3f} a2={a2:.3f} a3={a3:.3f}"
        )

        # checkpointing
        torch.save(net.state_dict(), str(run_dir / "last.pth"))

        if abs_rel < best_absrel:
            best_absrel = abs_rel
            best_epoch = epoch
            torch.save(net.state_dict(), str(run_dir / "best.pth"))
            log_line(f" New BEST -> best.pth (abs_rel={best_absrel:.4f})")

        if epoch % 10 == 0 and (run_dir / "best.pth").exists():
            snap = run_dir / f"best_e{epoch:03d}_absrel{best_absrel:.4f}.pth"
            torch.save(torch.load(str(run_dir / "best.pth"), map_location="cpu"), str(snap))
            log_line(f" Saved BEST snapshot -> {snap}")

    log_line(f"[DONE] best_abs_rel={best_absrel:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
