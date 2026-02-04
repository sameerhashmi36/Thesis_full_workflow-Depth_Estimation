"""
train_agco_zoe_strict.py

Purpose
-------
Supervised finetuning of ZoeDepth on the AGCO dataset using sparse LiDAR depth ground truth.

What it does
------------
- Loads a ZoeDepth model (ZoeD_K / ZoeD_N / ZoeD_NK) from the local ZoeDepth repository via torch.hub.
- Loads the pretrained base checkpoint STRICTLY (fails fast if weights do not match the model).
- Trains with masked supervised losses on valid LiDAR pixels:
    - masked L1 on metric depth
    - masked L1 on log(metric depth)
    - optional edge-aware smoothness regularization
- Validates after each epoch with Monodepth2-style metrics over valid pixels:
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
  Optional median scaling can be enabled for diagnostics.

Data assumptions
----------------
- Dataset root contains:
    raw_root/
      rectified/<bag_name>/rectified_idx*_t*.png
      depth_z16/<bag_name>/depth_idx******_t*.png
- Depth images are uint16 values representing millimeters; converted to meters in the dataset.
- Train/val/test are bag-level splits; test bags are fixed in agco_zoe_config.py.

Resolution handling
-------------------
- Training and validation use ZoeDepth input resolution (default 384x512) to reduce mismatch risk.
- Ground-truth depth and masks are resized to prediction resolution using nearest-neighbor interpolation.

Checkpoint loading (critical)
-----------------------------
- Base checkpoint is loaded using a strict state-dict extractor that:
    - finds the real tensor state_dict inside a .pt checkpoint
    - strips common prefixes (module., model., etc.)
    - loads with strict=True
  This prevents accidental "training from scratch" due to mismatched checkpoint structure.

Saving outputs
--------------
- Output root: models_finetuned_on_agco/ (default)
- Run directory name encodes model + train/val fractions:
    models_finetuned_on_agco/<MODEL>_trainXX_valYY/
- Files saved:
    - best.pth  : best model so far by lowest abs_rel (validation)
    - last.pth  : most recent epoch state_dict
    - best_eNNN_absrelX.XXXX.pth : snapshot of best-so-far saved every N epochs (debug)

Command examples
----------------
Train with 20% train/val fractions:
  python finetuning_and_eval_on_agco_dataset/train_agco_zoe_strict.py \
    --raw-root /path/to/raw_dataset_cpu_manual_1 \
    --model ZoeD_K \
    --base-ckpt checkpoints/ZoeD_M12_K.pt \
    --train-fraction 0.2 --val-fraction 0.2 \
    --epochs 20 --batch-size 2

Notes
-----
- For true metric-depth models (e.g., ZoeD_K), median scaling is typically not needed.
- For diagnostic comparison with monocular-style evaluation, enable --median-scaling.
"""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from agco_zoe_config import DEFAULT_RAW_ROOT, MIN_DEPTH, MAX_DEPTH
# from agco_zoe_dataset import AGCOZoeDepthDataset
from agco_zoe_dataset_smooth import AGCOZoeDepthDatasetSmooth as AGCOZoeDepthDataset
from zoe_ckpt_utils import load_state_dict_strict

ZOE_H, ZOE_W = 384, 512


def choose_device():
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            print(f"CUDA capability sm_{major}{minor}")
        except Exception:
            pass
        return torch.device("cuda")
    return torch.device("cpu")


def forward_zoe_metric(zoe: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = zoe(x)
    if isinstance(out, dict):
        for k in ["metric_depth", "depth", "pred", "out"]:
            if k in out:
                out = out[k]
                break
    if isinstance(out, (list, tuple)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"Unexpected Zoe output type: {type(out)}")
    if out.dim() == 3:
        out = out.unsqueeze(1)
    if out.dim() != 4:
        raise RuntimeError(f"Expected [B,1,H,W], got {tuple(out.shape)}")
    return out


def _to_b1hw(depth_gt, valid_mask):
    if depth_gt.dim() == 3:
        depth_gt = depth_gt.unsqueeze(1)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    return depth_gt, valid_mask.bool()


def resize_gt_to_pred(depth_gt, valid_mask, pred_hw):
    H, W = pred_hw
    if depth_gt.shape[-2:] != (H, W):
        depth_gt = F.interpolate(depth_gt, size=(H, W), mode="nearest")
        valid_mask = F.interpolate(valid_mask.float(), size=(H, W), mode="nearest").bool()
    return depth_gt, valid_mask


def masked_l1(pred, gt, mask, eps=1e-8):
    diff = torch.abs(pred - gt) * mask
    return diff.sum() / (mask.sum() + eps)


def masked_log_l1(pred, gt, mask, eps=1e-6):
    pred_c = torch.clamp(pred, min=eps)
    gt_c = torch.clamp(gt, min=eps)
    diff = torch.abs(torch.log(pred_c) - torch.log(gt_c)) * mask
    return diff.sum() / (mask.sum() + eps)


def edge_aware_smoothness(depth, image):
    if image.shape[-2:] != depth.shape[-2:]:
        image = F.interpolate(image, size=depth.shape[-2:], mode="bilinear", align_corners=False)

    depth_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    depth_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    img_dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    img_dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)

    weight_x = torch.exp(-img_dx)
    weight_y = torch.exp(-img_dy)
    return (depth_dx * weight_x).mean() + (depth_dy * weight_y).mean()


def build_loss(depth_pred, depth_gt, valid_mask, rgb, w_l1, w_log, w_smooth):
    range_mask = (depth_gt > MIN_DEPTH) & (depth_gt < MAX_DEPTH)
    mask = valid_mask & range_mask
    if mask.sum().item() == 0:
        zero = torch.zeros((), device=depth_pred.device)
        return zero, {"valid": 0.0, "l1": 0.0, "log": 0.0, "smooth": 0.0}

    l1 = masked_l1(depth_pred, depth_gt, mask)
    lg = masked_log_l1(depth_pred, depth_gt, mask)
    sm = edge_aware_smoothness(depth_pred, rgb) if w_smooth > 0 else torch.zeros((), device=depth_pred.device)
    loss = w_l1 * l1 + w_log * lg + w_smooth * sm

    return loss, {
        "valid": float(mask.sum().detach().cpu().item()),
        "l1": float(l1.detach().cpu().item()),
        "log": float(lg.detach().cpu().item()),
        "smooth": float(sm.detach().cpu().item()),
    }


def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def validate(model, loader, device, median_scaling: bool):
    model.eval()
    all_errors = []
    with torch.no_grad():
        for rgb, depth_gt, valid_mask, meta in tqdm(loader, desc="[Val]", ncols=110):
            rgb = rgb.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            depth_gt, valid_mask = _to_b1hw(depth_gt, valid_mask)
            depth_pred = forward_zoe_metric(model, rgb)

            Hp, Wp = depth_pred.shape[-2:]
            depth_gt, valid_mask = resize_gt_to_pred(depth_gt, valid_mask, (Hp, Wp))

            mask = valid_mask & (depth_gt > MIN_DEPTH) & (depth_gt < MAX_DEPTH)
            if mask.sum().item() == 0:
                continue

            gt_flat = depth_gt[mask].detach().cpu().numpy().astype(np.float32)
            pr_flat = depth_pred[mask].detach().cpu().numpy().astype(np.float32)

            if median_scaling:
                scale = np.median(gt_flat) / (np.median(pr_flat) + 1e-8)
            else:
                scale = 1.0

            pr_scaled = np.clip(pr_flat * scale, MIN_DEPTH, MAX_DEPTH)
            all_errors.append(compute_errors(gt_flat, pr_scaled))

    if not all_errors:
        return None
    return np.mean(np.array(all_errors), axis=0)


def _pct(x: float) -> int:
    return int(round(float(x) * 100))


def make_run_dir(output_root: Path, model: str, train_frac: float, val_frac: float) -> Path:
    # models_finetuned_on_agco/ZoeD_K_train20_val20/
    name = f"{model}_train{_pct(train_frac)}_val{_pct(val_frac)}"
    return output_root / name


def parse_args():
    p = argparse.ArgumentParser()

    # Paths (defaults stay)
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--output-root", type=str, default="models_finetuned_on_agco_smooth")

    # Split fractions (defaults stay)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--val-fraction", type=float, default=0.8)
    p.add_argument("--val-bag-fraction", type=float, default=0.2)

    # Model + pretrained
    p.add_argument("--model", type=str, default="ZoeD_K", choices=["ZoeD_K", "ZoeD_N", "ZoeD_NK"])
    p.add_argument("--base-ckpt", type=str, default="checkpoints/ZoeD_M12_K.pt")

    # Train params
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Loss weights
    p.add_argument("--w-l1", type=float, default=1.0)
    p.add_argument("--w-log", type=float, default=1.0)
    p.add_argument("--w-smooth", type=float, default=1e-4)

    # Validation
    p.add_argument("--median-scaling", action="store_true")

    # Debug saving
    p.add_argument("--save-best-every", type=int, default=10,
                   help="Every N epochs, save a snapshot of the BEST-so-far checkpoint (for debugging).")

    return p.parse_args()


def main():
    args = parse_args()

    # Repo root = ZoeDepth-1.0 (default style)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    device = choose_device()
    print("Device:", device)
    print("Repo root:", repo_root)

    # Output directories
    output_root = Path(args.output_root)
    run_dir = make_run_dir(output_root, args.model, args.train_fraction, args.val_fraction)
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Run dir:", run_dir)

    # Dataset at Zoe size for clean matching
    train_ds = AGCOZoeDepthDataset(
        raw_root=args.raw_root,
        split="train",
        train_fraction=args.train_fraction,
        img_width=ZOE_W,
        img_height=ZOE_H,
        val_bag_fraction=args.val_bag_fraction,
    )
    val_ds = AGCOZoeDepthDataset(
        raw_root=args.raw_root,
        split="val",
        val_fraction=args.val_fraction,
        img_width=ZOE_W,
        img_height=ZOE_H,
        val_bag_fraction=args.val_bag_fraction,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Build model (no weights) + strict load base
    print(f"[MODEL] torch.hub.load local: {repo_root}  name={args.model}")
    zoe = torch.hub.load(str(repo_root), args.model, source="local", pretrained=False)

    base_ckpt_path = args.base_ckpt if Path(args.base_ckpt).is_absolute() else str(repo_root / args.base_ckpt)
    print("[CKPT] Base ckpt:", base_ckpt_path)
    load_state_dict_strict(zoe, base_ckpt_path, device=device, verbose=True)

    zoe.to(device)

    optimizer = torch.optim.AdamW(
        zoe.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_abs_rel = float("inf")
    best_state_dict = None  # keep best in memory so snapshot saving is easy

    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"

    for epoch in range(1, args.epochs + 1):
        zoe.train()
        running = {"loss": 0.0, "l1": 0.0, "log": 0.0, "smooth": 0.0}
        n = 0

        pbar = tqdm(train_loader, desc=f"[Train e{epoch}/{args.epochs}]", ncols=120)
        for rgb, depth_gt, valid_mask, meta in pbar:
            rgb = rgb.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            depth_gt, valid_mask = _to_b1hw(depth_gt, valid_mask)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                depth_pred = forward_zoe_metric(zoe, rgb)
                Hp, Wp = depth_pred.shape[-2:]
                depth_gt_r, valid_mask_r = resize_gt_to_pred(depth_gt, valid_mask, (Hp, Wp))

                loss, stats = build_loss(
                    depth_pred, depth_gt_r, valid_mask_r, rgb,
                    args.w_l1, args.w_log, args.w_smooth
                )

            if stats["valid"] <= 0:
                pbar.set_postfix(loss="skip(no_valid)")
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(zoe.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running["loss"] += float(loss.detach().cpu().item())
            running["l1"] += stats["l1"]
            running["log"] += stats["log"]
            running["smooth"] += stats["smooth"]
            n += 1

            pbar.set_postfix(
                loss=f"{running['loss']/max(1,n):.3f}",
                l1=f"{running['l1']/max(1,n):.3f}",
                log=f"{running['log']/max(1,n):.3f}",
                sm=f"{running['smooth']/max(1,n):.6f}",
            )

        print(f"[Train] loss={running['loss']/max(1,n):.4f}  l1={running['l1']/max(1,n):.4f}  log={running['log']/max(1,n):.4f}")

        # Validate
        metrics = validate(zoe, val_loader, device, median_scaling=args.median_scaling)
        if metrics is None:
            print("[Val] No valid pixels")
        else:
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = metrics
            print(f"[Val] abs_rel={abs_rel:.3f} sq_rel={sq_rel:.3f} rmse={rmse:.3f} rmse_log={rmse_log:.3f} a1={a1:.3f}")

            # Update best
            if abs_rel < best_abs_rel:
                best_abs_rel = float(abs_rel)
                best_state_dict = {k: v.detach().cpu() for k, v in zoe.state_dict().items()}
                torch.save(best_state_dict, best_path)
                print(f" New BEST -> {best_path} (abs_rel={best_abs_rel:.4f})")

        # Always save last
        torch.save(zoe.state_dict(), last_path)

        # Debug snapshots: every N epochs, save BEST-so-far (if we have one)
        if args.save_best_every > 0 and (epoch % args.save_best_every == 0):
            if best_state_dict is not None:
                snap = run_dir / f"best_e{epoch:03d}_absrel{best_abs_rel:.4f}.pth"
                torch.save(best_state_dict, snap)
                print(f" Saved BEST snapshot -> {snap}")
            else:
                print(f"[Snapshot] epoch {epoch}: no best yet (no valid val metrics).")

    print("Done.")


if __name__ == "__main__":
    main()
