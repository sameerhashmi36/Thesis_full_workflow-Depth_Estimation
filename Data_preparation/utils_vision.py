"""
utils_vision.py
Decode + fisheye rectification + LiDAR unpack + point-splat Z-buffer (CPU).
"""

import math, struct
import numpy as np
import cv2
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)

# ====== Decoding ======
def decode_image(msg):
    """Return RGB uint8 from common encodings."""
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = msg.encoding.decode() if isinstance(msg.encoding,(bytes,bytearray)) else (msg.encoding or "")
    enc = enc.lower()
    buf = np.frombuffer(msg.data, np.uint8)

    if enc.startswith("bayer_rggb16"):
        arr16 = buf.view(np.uint16).reshape(h, step//2)[:, :w]
        rgb16 = cv2.cvtColor(arr16, cv2.COLOR_BayerRG2RGB)
        return (rgb16 >> 8).astype(np.uint8)

    if enc in ("rgb8","rgba8"):
        c = 4 if enc=="rgba8" else 3
        arr = buf.reshape(h, step)[:, :w*c].reshape(h, w, c)[..., :3]
        return arr

    if enc in ("bgr8","bgra8"):
        c = 4 if enc=="bgra8" else 3
        arr = buf.reshape(h, step)[:, :w*c].reshape(h, w, c)[..., :3]
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    if enc in ("mono8","8uc1"):
        g = buf.reshape(h, step)[:, :w].reshape(h, w)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    # best-effort 3ch fallback
    arr = buf.reshape(h, step)[:, :w*3].reshape(h, w, 3)
    return arr


# ====== Rectification ======
def scaled_KD_for_image(Kflat, Dflat, cal_w, cal_h, img_w, img_h):
    K = np.array(Kflat, dtype=np.float64).reshape(3,3)
    D = np.array(Dflat, dtype=np.float64).reshape(4,1)
    if (img_w, img_h) != (cal_w, cal_h):
        sx, sy = img_w/float(cal_w), img_h/float(cal_h)
        S = np.array([[sx,0,0],[0,sy,0],[0,0,1]], np.float64)
        K = S @ K
    return K, D


def rectify_rgb(rgb, out_w, out_h, K_flat, D_flat, cal_w, cal_h, rotate180=False):
    """Equidistant fisheye -> pinhole; returns (rect_rgb, K_rect)."""
    H0, W0 = rgb.shape[:2]
    if out_w is None or out_h is None:
        W, H = W0, H0; src = rgb
    else:
        W, H = int(out_w), int(out_h)
        src  = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    K_img, D_img = scaled_KD_for_image(K_flat, D_flat, cal_w, cal_h, W, H)
    R_rect = np.eye(3, dtype=np.float64)
    K_rect = K_img.copy()

    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K_img, D_img, R_rect, K_rect, (W, H), cv2.CV_16SC2)
    rect = cv2.remap(src, m1, m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if rotate180:
        rect = cv2.rotate(rect, cv2.ROTATE_180)
    return rect, K_rect


# ====== LiDAR unpack ======
def unpack_pc2_xyz(msg):
    """Extract (N,3) xyz from sensor_msgs/PointCloud2 (robust to field offsets)."""
    import math as _m
    step = int(msg.point_step)
    npts = int(msg.width) * int(msg.height)
    data = memoryview(msg.data)
    out = np.empty((npts, 3), np.float32)
    k = 0
    def off(field):
        for f in msg.fields:
            if f.name == field:
                return f.offset
        return None
    ox = off('x') or 0
    oy = off('y') or 4
    oz = off('z') or 8
    for i in range(npts):
        base = i * step
        x = struct.unpack_from('<f', data, base + ox)[0]
        y = struct.unpack_from('<f', data, base + oy)[0]
        z = struct.unpack_from('<f', data, base + oz)[0]
        if _m.isfinite(x) and _m.isfinite(y) and _m.isfinite(z):
            out[k] = (x, y, z); k += 1
    return out[:k].astype(np.float64)


# ====== Visual helpers ======
def colorize_mm(d_mm: np.ndarray) -> np.ndarray:
    d = d_mm.astype(np.float32)
    m = d > 0
    if not m.any():
        return np.zeros((*d.shape,3), np.uint8)
    lo, hi = np.percentile(d[m], [2,98]).astype(np.float32)
    hi = max(hi, lo+1.0)
    scaled = np.zeros_like(d, np.uint8)
    scaled[m] = np.clip((d - lo)/(hi - lo)*255.0, 0, 255).astype(np.uint8)[m]
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

def overlay_depth_on_rgb(rgb_rect: np.ndarray, d_mm: np.ndarray, alpha: float) -> np.ndarray:
    vis = colorize_mm(d_mm)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return (alpha * vis_rgb + (1 - alpha) * rgb_rect).astype(np.uint8)


# ====== Point-resolution-only: adaptive splats (no fusion, no hole-fill) ======
# Set these to exact Ouster model if needed (these defaults are sensible):
LIDAR_DTH_DEG = 0.2    # horizontal angular step (deg)
LIDAR_DTV_DEG = 0.35   # vertical angular step (deg)
MIN_SPLAT_PX  = 1
MAX_SPLAT_PX  = 8

def _splat_axes_px_from_angles(K_rect, Z_m, gain=1.5,
                               dth_deg=LIDAR_DTH_DEG, dtv_deg=LIDAR_DTV_DEG):
    """
    Convert LiDAR angular steps to pixel ellipse axes at depth Z.
    'gain' uniformly scales the ellipse size (user control).
    """
    fx, fy = float(K_rect[0,0]), float(K_rect[1,1])
    ax = fx * math.radians(dth_deg)
    ay = fy * math.radians(dtv_deg)
    # mild growth with range so far points don't vanish; keep conservative:
    rng_scale = 0.5 + 0.03 * max(0.0, Z_m)
    rx = int(np.clip(round(ax * gain * rng_scale), MIN_SPLAT_PX, MAX_SPLAT_PX))
    ry = int(np.clip(round(ay * gain * rng_scale), MIN_SPLAT_PX, MAX_SPLAT_PX))
    return rx, ry

def _ellipse_kernel(rx, ry):
    k = np.zeros((2*ry+1, 2*rx+1), np.uint8)
    cv2.ellipse(k, (rx, ry), (rx, ry), 0, 0, 360, 255, -1)
    return k

def zbuffer_numpy_splat_only(rect, K_rect, xyz, T, zmax_m, splat_gain=1.5, alpha=0.5):
    """
    Paint each LiDAR point as a small ellipse (nearest-Z). No hole fill, no fusion.
    Returns (depth_m float32 with NaNs, d_mm uint16, overlay RGB).
    """
    H, W = rect.shape[:2]
    fx, fy, cx, cy = float(K_rect[0,0]), float(K_rect[1,1]), float(K_rect[0,2]), float(K_rect[1,2])
    depth = np.full((H, W), np.inf, np.float32)

    if xyz is not None and xyz.size:
        pts_l = np.hstack([xyz, np.ones((xyz.shape[0],1), np.float64)])
        pts_c = (T @ pts_l.T).T[:, :3]
        X,Y,Z = pts_c[:,0], pts_c[:,1], pts_c[:,2]
        mpos = Z > 0
        X,Y,Z = X[mpos], Y[mpos], Z[mpos]
        if X.size:
            u = np.rint(fx*(X/Z) + cx).astype(np.int32)
            v = np.rint(fy*(Y/Z) + cy).astype(np.int32)
            inb = (u>=0)&(u<W)&(v>=0)&(v<H)
            u, v, Z = u[inb], v[inb], Z[inb]
            for ui, vi, zi in zip(u, v, Z):
                rx, ry = _splat_axes_px_from_angles(K_rect, float(zi), gain=splat_gain)
                x0, x1 = max(0, ui-rx), min(W-1, ui+rx)
                y0, y1 = max(0, vi-ry), min(H-1, vi+ry)
                if x0 >= x1 or y0 >= y1:
                    if zi < depth[vi, ui]: depth[vi, ui] = zi
                    continue
                patch = depth[y0:y1+1, x0:x1+1]
                kern  = _ellipse_kernel(rx, ry)
                ky0 = max(0, ry-(vi-y0)); ky1 = min(kern.shape[0], ry+(y1-vi)+1)
                kx0 = max(0, rx-(ui-x0)); kx1 = min(kern.shape[1], rx+(x1-ui)+1)
                kp = kern[ky0:ky1, kx0:kx1].astype(bool)
                if kp.any():
                    cur = patch[kp]
                    patch[kp] = np.where(np.isfinite(cur), np.minimum(cur, zi), zi)

    depth[np.isinf(depth)] = np.nan
    zmax_m = max(1.0, float(zmax_m))
    d_mm = (np.clip(np.nan_to_num(depth, nan=0.0), 0, zmax_m) * 1000.0).astype(np.uint16)
    overlay = overlay_depth_on_rgb(rect, d_mm, alpha=alpha)
    return depth, d_mm, overlay
