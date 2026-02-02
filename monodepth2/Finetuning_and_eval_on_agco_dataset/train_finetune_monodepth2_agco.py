"""
Supervised finetuning of monodepth2 on the AGCO dataset.

This script:
- loads pretrained mono_640x192 encoder and depth decoder
- builds AGCO train/val splits using AGCODepthDataset
- applies supervised depth loss using LiDAR-derived depth maps
- adds a smoothness loss term from monodepth2 layers
- saves best finetuned weights (encoder.pth, depth.pth) to a given folder

Usage example:

  cd /path/to/monodepth2

Train (normal dataset, with frame fractions):
python Finetuning_and_eval_on_agco_dataset/train_finetune_monodepth2_agco.py \
  --raw-root /home/sameer/Documents/Zoedepth_v1/raw_dataset_cpu_manual_1 \
  --pretrained-folder ./models/mono_640x192 \
  --output-folder ./models_finetuned_on_agco/agco_mono_640x192_finetuned_train20_val20 \
  --val-bag-fraction 0.2 \
  --train-fraction 0.2 --val-fraction 0.2 \
  --batch-size 4 --epochs 10

  
Train (smooth dataset):
python Finetuning_and_eval_on_agco_dataset/train_finetune_monodepth2_agco.py \
  --use-smooth \
  --raw-root /home/sameer/Documents/Zoedepth_v1/raw_dataset_cpu_manual_1 \
  --pretrained-folder ./models/mono_640x192 \
  --output-folder ./models_finetuned_on_agco/agco_mono_640x192_finetuned_smooth_train20_val20 \
  --val-bag-fraction 0.2 \
  --train-fraction 0.2 --val-fraction 0.2 \
  --batch-size 4 --epochs 10

  
"""


import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import networks
from layers import disp_to_depth, get_smooth_loss

from agco_config import (
    DEFAULT_RAW_ROOT,
    DEFAULT_VAL_BAG_FRACTION,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VAL_FRACTION,
    MIN_DEPTH,
    MAX_DEPTH,
    DEFAULT_SEED,
)


def compute_depth_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_pretrained_model(weights_folder: Path, device: torch.device):
    encoder_path = weights_folder / "encoder.pth"
    depth_path = weights_folder / "depth.pth"
    if not encoder_path.exists() or not depth_path.exists():
        raise FileNotFoundError(f"Cannot find encoder.pth/depth.pth in {weights_folder}")

    print(f"-> Loading pretrained weights from {weights_folder}")

    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    enc = torch.load(encoder_path, map_location=device)
    enc = {k: v for k, v in enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(enc)

    dec = torch.load(depth_path, map_location=device)
    depth_decoder.load_state_dict(dec)

    encoder.to(device)
    depth_decoder.to(device)
    print("-> Pretrained model loaded.")
    return encoder, depth_decoder


def save_model(encoder, depth_decoder, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), output_folder / "encoder.pth")
    torch.save(depth_decoder.state_dict(), output_folder / "depth.pth")
    print(f"-> Saved finetuned weights to {output_folder}")


def train_one_epoch(encoder, depth_decoder, loader, optimizer, device, smooth_weight, min_depth, max_depth):
    encoder.train()
    depth_decoder.train()

    sup_sum = 0.0
    smooth_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for rgb, depth_gt, valid_mask, meta in pbar:
        rgb = rgb.to(device)               # [B,3,H,W]
        depth_gt = depth_gt.to(device)     # [B,1,H,W]
        valid_mask = valid_mask.to(device) # bool or uint8

        optimizer.zero_grad()

        feats = encoder(rgb)
        outputs = depth_decoder(feats)
        disp = outputs[("disp", 0)]

        if disp.shape[-2:] != depth_gt.shape[-2:]:
            disp = F.interpolate(disp, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

        _, depth_pred = disp_to_depth(disp, min_depth, max_depth)

        vm = valid_mask.float()
        depth_diff = torch.abs(depth_pred - depth_gt)
        depth_loss = (depth_diff * vm).sum() / (vm.sum() + 1e-7)

        smooth_loss = get_smooth_loss(disp, rgb)

        loss = depth_loss + smooth_weight * smooth_loss
        loss.backward()
        optimizer.step()

        sup_sum += float(depth_loss.item())
        smooth_sum += float(smooth_loss.item())
        n += 1

        pbar.set_postfix({"sup": f"{sup_sum/max(1,n):.4f}", "sm": f"{smooth_sum/max(1,n):.6f}"})

    return sup_sum / max(1, n), smooth_sum / max(1, n)


def validate(encoder, depth_decoder, loader, device, min_depth, max_depth, median_scaling=True):
    encoder.eval()
    depth_decoder.eval()

    errs = []
    pbar = tqdm(loader, desc="val", leave=False)

    with torch.no_grad():
        for rgb, depth_gt, valid_mask, meta in pbar:
            rgb = rgb.to(device)
            depth_gt = depth_gt.to(device)
            valid_mask = valid_mask.to(device)

            feats = encoder(rgb)
            disp = depth_decoder(feats)[("disp", 0)]

            if disp.shape[-2:] != depth_gt.shape[-2:]:
                disp = F.interpolate(disp, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

            _, depth_pred = disp_to_depth(disp, min_depth, max_depth)

            mask = (valid_mask > 0) & (depth_gt > min_depth) & (depth_gt < max_depth)
            if mask.sum().item() == 0:
                continue

            gt_flat = depth_gt[mask].detach().cpu().numpy().astype(np.float32)
            pr_flat = depth_pred[mask].detach().cpu().numpy().astype(np.float32)

            if median_scaling:
                scale = np.median(gt_flat) / (np.median(pr_flat) + 1e-12)
                pr_flat = np.clip(pr_flat * scale, min_depth, max_depth)
            else:
                pr_flat = np.clip(pr_flat, min_depth, max_depth)

            errs.append(compute_depth_errors(gt_flat, pr_flat))

            if len(errs) > 0:
                m = np.array(errs).mean(axis=0)
                pbar.set_postfix({"abs_rel": f"{m[0]:.3f}", "rmse": f"{m[2]:.3f}"})

    if not errs:
        return None

    m = np.array(errs).mean(axis=0)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = m
    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "a1": float(a1),
        "a2": float(a2),
        "a3": float(a3),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--pretrained-folder", type=str, default="./models/mono_640x192")
    p.add_argument("--output-folder", type=str, default="./models_finetuned_on_agco/agco_mono_640x192_finetuned")

    p.add_argument("--use-smooth", action="store_true", help="Use agco_dataset_smooth instead of agco_dataset")

    p.add_argument("--img-width", type=int, default=640)
    p.add_argument("--img-height", type=int, default=192)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)

    # Zoe/DA3-like split params
    p.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION, help="Frame fraction in TRAIN split")
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION, help="Frame fraction in VAL split")
    p.add_argument("--val-bag-fraction", type=float, default=DEFAULT_VAL_BAG_FRACTION, help="Bag split fraction for VAL")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--smooth-weight", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--no-median-scaling", action="store_true", help="Disable median scaling in validation")
    return p.parse_args()


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    pretrained = Path(args.pretrained_folder)
    out_dir = Path(args.output_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Choose dataset
    if args.use_smooth:
        from agco_dataset_smooth import AGCODepthDataset
        print("Dataset: SMOOTH")
    else:
        from agco_dataset import AGCODepthDataset
        print("Dataset: NORMAL")

    train_ds = AGCODepthDataset(
        raw_root=str(raw_root),
        split="train",
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=1.0,
        val_bag_fraction=args.val_bag_fraction,
        seed=args.seed,
        img_width=args.img_width,
        img_height=args.img_height,
        verbose_bags=True,
    )

    val_ds = AGCODepthDataset(
        raw_root=str(raw_root),
        split="val",
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=1.0,
        val_bag_fraction=args.val_bag_fraction,
        seed=args.seed,
        img_width=args.img_width,
        img_height=args.img_height,
        verbose_bags=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    encoder, depth_decoder = load_pretrained_model(pretrained, device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(depth_decoder.parameters()), lr=args.lr)

    best_abs_rel = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n==== Epoch {epoch}/{args.epochs} ====")

        tr_sup, tr_sm = train_one_epoch(
            encoder, depth_decoder, train_loader, optimizer, device,
            smooth_weight=args.smooth_weight, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH
        )
        print(f"[Train] supervised_loss={tr_sup:.4f}, smooth_loss={tr_sm:.6f}")

        metrics = validate(
            encoder, depth_decoder, val_loader, device,
            min_depth=MIN_DEPTH, max_depth=MAX_DEPTH,
            median_scaling=(not args.no_median_scaling)
        )

        if metrics is None:
            print("[Val] No valid pixels for metrics.")
            continue

        print(
            "[Val] abs_rel={abs_rel:.3f}, sq_rel={sq_rel:.3f}, rmse={rmse:.3f}, rmse_log={rmse_log:.3f}, "
            "a1={a1:.3f}, a2={a2:.3f}, a3={a3:.3f}".format(**metrics)
        )

        if metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = metrics["abs_rel"]
            print(f"-> New best abs_rel={best_abs_rel:.3f}. Saving...")
            save_model(encoder, depth_decoder, out_dir)

    print("\nFinetuning finished.")


if __name__ == "__main__":
    main()

