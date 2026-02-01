"""
Check z16 LiDAR depth PNG.

Usage:
    python check_depth_png.py path/to/depth_idx000000_t....png
"""

import cv2
import numpy as np

def main():
    path = "./raw_dataset_cpu_manual_1/depth_z16/tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-37-57_7/depth_idx000000_t1713267477.937.png" # path to depth image
    print(f"Reading: {path}")

    # Load exactly as stored (uint16)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if d is None:
        print(" Could not read image, check the path!")
        return

    print("\n===== Image Info =====")
    print(f"dtype : {d.dtype}")
    print(f"shape : {d.shape}")  # (H, W)

    # Stats
    min_val = int(d.min())
    max_val = int(d.max())
    print(f"min raw value : {min_val}")
    print(f"max raw value : {max_val}")

    # Extract valid depth pixels
    nonzero = d[d > 0]
    print(f"valid (non-zero) pixels: {nonzero.size}")

    if nonzero.size == 0:
        print("No valid depth values found.")
        return

    # Show sample values
    sample = nonzero[:10].astype(int)
    print("\nSample depth values (mm):", sample.tolist())
    print("Sample depth values (meters):", (sample / 1000.0).tolist())

    # Show depth percentiles
    nz_m = nonzero.astype(np.float32) / 1000.0
    p = np.percentile(nz_m, [5, 50, 95])
    print("\nDepth percentiles (meters):")
    print(f"  5%  : {p[0]:.3f} m")
    print(f"  50% : {p[1]:.3f} m")
    print(f"  95% : {p[2]:.3f} m")

if __name__ == "__main__":
    main()

# Reading: ./raw_dataset_cpu_manual_1/depth_z16/tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-37-57_7/depth_idx000000_t1713267477.937.png

# ===== Image Info =====
# dtype : uint16
# shape : (928, 1440)
# min raw value : 0
# max raw value : 29292
# valid (non-zero) pixels: 482202

# Sample depth values (mm): [14464, 14464, 14464, 14464, 14464, 14464, 14464, 14464, 14464, 14464]
# Sample depth values (meters): [14.464, 14.464, 14.464, 14.464, 14.464, 14.464, 14.464, 14.464, 14.464, 14.464]

# Depth percentiles (meters):
#   5%  : 1.001 m
#   50% : 4.831 m
#   95% : 11.736 m