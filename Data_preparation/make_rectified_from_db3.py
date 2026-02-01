"""
make_rectified_from_db3.py

Exports rectified images from all ROS2 bag folders under DB3_DIR.

Outputs:
  OUTPUT_ROOT/rectified/<bagfolder>/rectified_idx###_t<cam_time>.png
  OUTPUT_ROOT/summaries/rectified_summary.csv
"""

import os, csv, time
from pathlib import Path
from statistics import mean
import numpy as np
import cv2

from rosbags.typesys import Stores, get_typestore
from utils_ros2 import iter_cam_messages
from utils_vision import decode_image, rectify_rgb

# ---------- CONFIG ----------
DB3_DIR     = Path("/path/to/the/dataset/folder")
OUTPUT_ROOT = Path("raw_dataset_cpu_manual_1")
CAM_TOPIC   = "/cam_lucid_front/image_raw"

# Fisheye calibration (measured at CAL_W x CAL_H)
CAL_W, CAL_H = 2880, 1860
K_FLAT = [876.03473692, 0.0,           1418.31260146,
          0.0,          877.74642419,   936.26062054,
          0.0,          0.0,             1.0]
D_FISH = [-0.04833991, -0.00567162, 0.00099287, -0.00037649]

# Rectified output size
OUT_W, OUT_H = 1440, 928
ROTATE_180_AT_END = True

NUM_IMAGES_PER_BAG = None  # None = all frames
# ------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
TS = get_typestore(Stores.ROS2_FOXY)

def bag_folders(root: Path):
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "metadata.yaml").exists() and list(p.glob("*.db3")):
            yield p

def p95(vals):
    if not vals: return 0.0
    arr = np.array(vals, dtype=np.float64)
    return float(np.percentile(arr, 95))

def main():
    rect_dir = (OUTPUT_ROOT / "rectified"); rect_dir.mkdir(parents=True, exist_ok=True)
    sum_dir  = (OUTPUT_ROOT / "summaries"); sum_dir.mkdir(parents=True, exist_ok=True)
    sum_csv  = sum_dir / "rectified_summary.csv"

    bags = list(bag_folders(DB3_DIR))
    if not bags:
        print(f"No bag folders found in: {DB3_DIR}")
        return
    print(f"Found {len(bags)} bag folder(s). Rectified output → {rect_dir.resolve()}")

    if not sum_csv.exists():
        with sum_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["bag_name","frames_rectified","avg_rectify_ms","p95_rectify_ms","bag_elapsed_s","out_dir"])

    grand_frames = 0
    grand_start  = time.perf_counter()

    for bag_root in bags:
        bag_out_dir = rect_dir / bag_root.name
        bag_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Bag] {bag_root.name}")
        processed, timings = 0, []
        t_bag0 = time.perf_counter()

        for cam_idx, (t_cam, raw_cam, cam_type) in enumerate(iter_cam_messages(bag_root, CAM_TOPIC)):
            if NUM_IMAGES_PER_BAG and processed >= NUM_IMAGES_PER_BAG:
                break
            msg_cam = TS.deserialize_cdr(raw_cam, cam_type)
            rgb = decode_image(msg_cam)

            t0 = time.perf_counter()
            rect, K_rect = rectify_rgb(rgb, OUT_W, OUT_H, K_FLAT, D_FISH, CAL_W, CAL_H, rotate180=ROTATE_180_AT_END)
            timings.append((time.perf_counter() - t0) * 1000.0)

            out_name = f"rectified_idx{cam_idx:03d}_t{t_cam:.3f}.png"
            cv2.imwrite(str(bag_out_dir / out_name), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
            if processed % 50 == 0:
                print(f"[OK] {out_name} (frame {processed})")
            processed += 1

        bag_elapsed = time.perf_counter() - t_bag0
        grand_frames += processed
        avg_ms = mean(timings) if timings else 0.0
        p95_ms = p95(timings)

        print(f"[Bag Summary] {bag_root.name}: frames={processed}, avg_rect={avg_ms:.2f} ms, p95_rect={p95_ms:.2f} ms, elapsed={bag_elapsed:.2f} s")
        with sum_csv.open("a", newline="") as f:
            csv.writer(f).writerow([bag_root.name, processed, f"{avg_ms:.2f}", f"{p95_ms:.2f}", f"{bag_elapsed:.2f}", str(bag_out_dir.resolve())])

    print(f"\nDone. Total frames={grand_frames}, total time={time.perf_counter()-grand_start:.2f} s")
    print(f"Summary CSV → {sum_csv.resolve()}")

if __name__ == "__main__":
    main()
