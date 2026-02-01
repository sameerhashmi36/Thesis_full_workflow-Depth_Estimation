"""
make_depth_manual_tweak_test_1.py (CPU-only)

Goal: Make LiDAR points more visible ("higher point resolution look")
without changing TF logic or fusing scans by default.

What this script does:
- The tweak LiDAR->camera extrinsics per bag (TF+delta, else tweak-only).
- Uses an adaptive ellipse "splat" per LiDAR point to cover a few pixels
  (nearest-Z), controlled by --splat-gain.
- Saves full-res z16 depth; optionally saves QA overlays.

Keys while previewing:
  s = save preview only
  c = confirm & process (or re-process) current bag
  u = update tweak json only (no processing)
  n = skip this bag
  r = same as c (redo/process now)
  q = quit (progress saved)
"""

import os, json, time, math, bisect, argparse, gc
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from rosbags.typesys import Stores, get_typestore
from utils_ros2   import iter_cam_messages, load_all_lidar, has_topic, count_messages
from utils_tf     import pick_tf_from_bag_robust, compose_T
from utils_vision import (
    decode_image, rectify_rgb, unpack_pc2_xyz,
    zbuffer_numpy_splat_only, overlay_depth_on_rgb, colorize_mm
)

# ---------- ROOTS ----------
DB3_DIR     = Path("path/to/ros2bags")
OUTPUT_ROOT = Path("raw_dataset_cpu_manual_1")
DEPTH_DIR   = OUTPUT_ROOT / "depth_z16"
SUMMARY_DIR = OUTPUT_ROOT / "summaries"
TWEAK_DIR   = SUMMARY_DIR / "tweaks"
PROGRESS_JSON = SUMMARY_DIR / "progress.json"

# ---------- TOPICS & CALIB ----------
CAM_TOPIC   = "/cam_lucid_front/image_raw"
LIDAR_TOPIC = "/ouster_front/ouster/points"

CAL_W, CAL_H = 2880, 1860
K_FLAT = [876.03473692, 0.0,           1418.31260146,
          0.0,          877.74642419,   936.26062054,
          0.0,          0.0,             1.0]
D_FISH = [-0.04833991, -0.00567162,  0.00099287, -0.00037649]

OUT_W, OUT_H = 1440, 928
ROTATE_180_AT_END = True

# ----- Initial tweak defaults -----
INIT_YAW_DEG, INIT_PITCH_DEG, INIT_ROLL_DEG = -90, +4, -89
INIT_TX_CM,   INIT_TY_CM,    INIT_TZ_CM    =  -2, +14,  -3

# ----- Params -----
MAX_LIDAR_DT_S     = 0.12
QA_EVERY           = 50
MAX_PTS_PREVIEW    = 60000
PREVIEW_REFRESH_MS = 15
CLAMP_M            = 655.35  # max depth clamp (m) when writing z16

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
TS = get_typestore(Stores.ROS2_FOXY)


# ---------- generic helpers ----------
def discover_bag_roots(root: Path) -> list[Path]:
    out = []
    if not root.exists():
        return out
    out += sorted(root.glob("*.db3"))
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "metadata.yaml").exists() and list(p.glob("*.db3")):
            out.append(p)
    return out

def Rx(deg):
    r = math.radians(deg); c,s = math.cos(r), math.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], np.float64)

def Ry(deg):
    r = math.radians(deg); c,s = math.cos(r), math.sin(r)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float64)

def Rz(deg):
    r = math.radians(deg); c,s = math.cos(r), math.sin(r)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float64)

def delta_from_tweaks(yaw,pitch,roll,tx_cm,ty_cm,tz_cm) -> np.ndarray:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    t = np.array([tx_cm, ty_cm, tz_cm], np.float64) / 100.0
    return compose_T(R, t)

def find_nearest_idx(times: List[float], t: float) -> Optional[int]:
    if not times:
        return None
    j = bisect.bisect_left(times, t)
    cand = []
    if j-1 >= 0: cand.append(j-1)
    if j   < len(times): cand.append(j)
    if not cand:
        return None
    return min(cand, key=lambda i: abs(times[i]-t))

def load_progress(bag_order: List[str]) -> Dict[str, Any]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS_JSON.exists():
        try:
            data = json.loads(PROGRESS_JSON.read_text())
            data.setdefault("bag_order", bag_order)
            data.setdefault("done", [])
            data.setdefault("last_index", 0)
            return data
        except Exception:
            pass
    data = {"bag_order": bag_order, "done": [], "last_index": 0}
    PROGRESS_JSON.write_text(json.dumps(data, indent=2))
    return data

def save_progress(progress: Dict[str, Any]):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_JSON.write_text(json.dumps(progress, indent=2))

def load_tweak_json(bagstem: str) -> Optional[Dict[str, float]]:
    p = TWEAK_DIR / f"{bagstem}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

def save_tweak_json(bagstem: str, tweak: Dict[str, float]):
    TWEAK_DIR.mkdir(parents=True, exist_ok=True)
    (TWEAK_DIR / f"{bagstem}.json").write_text(json.dumps(tweak, indent=2))

def tf_or_tweak_extrinsic(bag_root: Path, yaw,pitch,roll,tx,ty,tz) -> Tuple[np.ndarray, str]:
    try:
        T_base, tf_path, dbg = pick_tf_from_bag_robust(
            bag_root,
            cam_topic=CAM_TOPIC,
            lidar_topic=LIDAR_TOPIC,
            camera_name_hint="cam_lucid_front",
            lidar_hint="ouster"
        )
        T_camOpt = delta_from_tweaks(yaw,pitch,roll,tx,ty,tz) @ T_base
        return T_camOpt, "tf+delta"
    except Exception:
        # No TF → tweak-only (full extrinsic)
        return delta_from_tweaks(yaw,pitch,roll,tx,ty,tz), "tweak_only"

def rectified_first_frame_and_nearest_lidar(bag_root: Path):
    lid_times, lid_raws, lid_types = load_all_lidar(bag_root, LIDAR_TOPIC, offset=0.0)
    if not lid_times:
        return None, None, None, None, None, None, None
    for t_cam, raw_cam, cam_type in iter_cam_messages(bag_root, CAM_TOPIC):
        msg_cam = TS.deserialize_cdr(raw_cam, cam_type)
        rgb = decode_image(msg_cam)
        rect, K_rect = rectify_rgb(
            rgb, OUT_W, OUT_H,
            K_FLAT, D_FISH, CAL_W, CAL_H,
            rotate180=ROTATE_180_AT_END
        )
        k = find_nearest_idx(lid_times, t_cam)
        if k is None or abs(lid_times[k]-t_cam) > MAX_LIDAR_DT_S:
            continue
        msg_lid = TS.deserialize_cdr(lid_raws[k], lid_types[k])
        xyz = unpack_pc2_xyz(msg_lid)
        return rect, K_rect, t_cam, xyz, lid_times, lid_raws, lid_types
    return None, None, None, None, None, None, None


# ---------- UI helpers ----------
def ensure_dirs():
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    TWEAK_DIR.mkdir(parents=True, exist_ok=True)

def init_trackbar_window(win="tune"):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.createTrackbar("yaw(deg)",   win, 180 + INIT_YAW_DEG,   360, lambda x: None)
    cv2.createTrackbar("pitch(deg)", win, 180 + INIT_PITCH_DEG, 360, lambda x: None)
    cv2.createTrackbar("roll(deg)",  win, 180 + INIT_ROLL_DEG,  360, lambda x: None)
    cv2.createTrackbar("tx(cm)",     win, 100 + INIT_TX_CM,     200, lambda x: None)
    cv2.createTrackbar("ty(cm)",     win, 100 + INIT_TY_CM,     200, lambda x: None)
    cv2.createTrackbar("tz(cm)",     win, 100 + INIT_TZ_CM,     200, lambda x: None)
    cv2.createTrackbar("zmax(m)",    win, 80,                   1000, lambda x: None)
    cv2.createTrackbar("autoZ(0/1)", win, 1,                     1,   lambda x: None)
    cv2.createTrackbar("alpha(x100)",win, 50,                   100,  lambda x: None)

def set_trackbar_from_tweak(tw: Dict[str, float], win="tune"):
    cv2.setTrackbarPos("yaw(deg)",   win, int(180 + tw.get("yaw",   INIT_YAW_DEG)))
    cv2.setTrackbarPos("pitch(deg)", win, int(180 + tw.get("pitch", INIT_PITCH_DEG)))
    cv2.setTrackbarPos("roll(deg)",  win, int(180 + tw.get("roll",  INIT_ROLL_DEG)))
    cv2.setTrackbarPos("tx(cm)",     win, int(100 + tw.get("tx_cm", INIT_TX_CM)))
    cv2.setTrackbarPos("ty(cm)",     win, int(100 + tw.get("ty_cm", INIT_TY_CM)))
    cv2.setTrackbarPos("tz(cm)",     win, int(100 + tw.get("tz_cm", INIT_TZ_CM)))

def get_trackbar_values(win="tune") -> Dict[str, float]:
    yaw   = cv2.getTrackbarPos("yaw(deg)",   win) - 180
    pitch = cv2.getTrackbarPos("pitch(deg)", win) - 180
    roll  = cv2.getTrackbarPos("roll(deg)",  win) - 180
    txcm  = cv2.getTrackbarPos("tx(cm)",     win) - 100
    tycm  = cv2.getTrackbarPos("ty(cm)",     win) - 100
    tzcm  = cv2.getTrackbarPos("tz(cm)",     win) - 100
    zmax  = max(1, cv2.getTrackbarPos("zmax(m)", "tune"))
    autoZ = cv2.getTrackbarPos("autoZ(0/1)", "tune")
    alpha = cv2.getTrackbarPos("alpha(x100)", "tune")/100.0
    return dict(yaw=yaw, pitch=pitch, roll=roll,
                tx_cm=txcm, ty_cm=tycm, tz_cm=tzcm,
                zmax=zmax, autoZ=autoZ, alpha=alpha)


# ---------- bag processing ----------
def process_whole_bag(bag_root: Path, tweak: Dict[str,float],
                      mode_hint: str,
                      splat_gain: float,
                      qa_height: Optional[int]) -> int:
    bagstem = bag_root.stem
    bag_out_dir = DEPTH_DIR / bagstem
    qa_out_dir  = bag_out_dir / "qa"
    bag_out_dir.mkdir(parents=True, exist_ok=True)
    qa_out_dir.mkdir(parents=True, exist_ok=True)

    yaw, pitch, roll = tweak["yaw"], tweak["pitch"], tweak["roll"]
    tx, ty, tz       = tweak["tx_cm"], tweak["ty_cm"], tweak["tz_cm"]
    zmax_m           = float(tweak.get("zmax", CLAMP_M))
    alpha            = float(tweak.get("alpha", 0.5))

    T_camOpt_lidar, base_mode = tf_or_tweak_extrinsic(bag_root, yaw,pitch,roll,tx,ty,tz)
    (bag_out_dir / "extrinsic_mode.txt").write_text(f"{base_mode}\n")

    lid_times, lid_raws, lid_types = load_all_lidar(bag_root, LIDAR_TOPIC, offset=0.0)
    if not lid_times:
        print("  [WARN] No LiDAR scans.")
        return 0

    processed = 0
    per_ms = []

    for cam_idx, (t_cam, raw_cam, cam_type) in enumerate(iter_cam_messages(bag_root, CAM_TOPIC)):
        k = find_nearest_idx(lid_times, t_cam)
        if k is None or abs(lid_times[k]-t_cam) > MAX_LIDAR_DT_S:
            continue

        msg_cam = TS.deserialize_cdr(raw_cam, cam_type)
        rgb = decode_image(msg_cam)
        rect, K_rect = rectify_rgb(
            rgb, OUT_W, OUT_H,
            K_FLAT, D_FISH, CAL_W, CAL_H,
            rotate180=ROTATE_180_AT_END
        )

        # Single-scan only (point-resolution look). No temporal fusion.
        msg_lid = TS.deserialize_cdr(lid_raws[k], lid_types[k])
        xyz = unpack_pc2_xyz(msg_lid)

        t0 = time.perf_counter()
        depth_m, d_mm, overlay_full = zbuffer_numpy_splat_only(
            rect, K_rect, xyz, T_camOpt_lidar, zmax_m,
            splat_gain=splat_gain, alpha=alpha
        )
        out_name = f"depth_idx{cam_idx:06d}_t{t_cam:.3f}.png"
        cv2.imwrite(str(bag_out_dir / out_name), d_mm)

        if QA_EVERY and (processed % QA_EVERY == 0):
            overlay_vis = overlay_full
            if qa_height and qa_height > 0:
                scale = qa_height / overlay_full.shape[0]
                overlay_vis = cv2.resize(overlay_full, (int(overlay_full.shape[1]*scale), qa_height),
                                         interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(qa_out_dir / f"overlay_idx{cam_idx:06d}.png"),
                        cv2.cvtColor(overlay_vis, cv2.COLOR_RGB2BGR))

        per_ms.append((time.perf_counter() - t0) * 1000.0)
        processed += 1

        # free
        del msg_cam, rgb, rect, K_rect, msg_lid, xyz, depth_m, d_mm, overlay_full
        if processed % 50 == 0:
            print(f"  [OK] {out_name}")

    if processed > 0:
        avg = float(np.mean(per_ms))
        p95 = float(np.percentile(np.array(per_ms, np.float32), 95))
        print(f"  [Bag Summary] frames={processed}, avg={avg:.2f} ms, p95={p95:.2f} ms, device=cpu")
    else:
        print("  [Bag Summary] frames=0 (no matched timestamps)")

    gc.collect()
    return processed


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume from last index")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--bag", type=str, default=None, help="substring to pick a specific bag")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--splat-gain", type=float, default=1.8,
                        help="scales ellipse size per point (try 1.2..2.8)")
    parser.add_argument("--qa-height", type=int, default=960,
                        help="resize QA overlays to this height (visual only). Set 0 to disable resize.")
    args = parser.parse_args()

    ensure_dirs()
    bag_roots = discover_bag_roots(DB3_DIR)
    bag_order = [str(p) for p in bag_roots]
    progress = load_progress(bag_order)
    done_set = set(progress.get("done", []))
    last_index = progress.get("last_index", 0)

    # choose start index
    if args.resume:
        start_idx = last_index
    elif args.start_index is not None:
        start_idx = max(0, min(args.start_index, len(bag_order)-1))
    elif args.bag:
        matches = [i for i, s in enumerate(bag_order) if args.bag in s]
        if not matches:
            print(f" Bag name '{args.bag}' not found.")
            return
        start_idx = matches[0]
    else:
        start_idx = last_index

    print(f"Found {len(bag_order)} bag(s). Done so far: {len(done_set)}. Starting at index {start_idx}.")
    print("Keys: [s]=save preview, [c]=confirm & process, [u]=update tweak, [n]=next bag, [r]=redo/process now, [q]=quit")

    last_used_tweak: Optional[Dict[str,float]] = None
    i = start_idx

    while i < len(bag_order):
        bag_path = Path(bag_order[i])
        bagstem = bag_path.stem
        print(f"\n=== [{i+1}/{len(bag_order)}] {bagstem} ===")

        # LiDAR sanity
        if not has_topic(bag_path, LIDAR_TOPIC) or count_messages(bag_path, LIDAR_TOPIC, max_stop=1) == 0:
            print("  [SKIP] No LiDAR data.")
            i += 1
            progress["last_index"] = i
            save_progress(progress)
            continue

        rect, K_rect, t_cam0, xyz0, lid_times, lid_raws, lid_types = rectified_first_frame_and_nearest_lidar(bag_path)
        if rect is None:
            print("  [SKIP] No camera frame with nearby LiDAR.")
            i += 1
            progress["last_index"] = i
            save_progress(progress)
            continue

        # downsample preview cloud (for speed only)
        if xyz0 is not None and xyz0.shape[0] > MAX_PTS_PREVIEW:
            stride = max(1, xyz0.shape[0] // MAX_PTS_PREVIEW)
            xyz0 = xyz0[::stride]

        init_trackbar_window("tune")

        bag_tweak = load_tweak_json(bagstem)
        if bag_tweak:
            print("  [INFO] Loaded existing tweak for this bag.")
            set_trackbar_from_tweak(bag_tweak)
            cur_tweak = bag_tweak.copy()
        elif last_used_tweak:
            print("  [INFO] Initialized sliders from last bag's tweak.")
            set_trackbar_from_tweak(last_used_tweak)
            cur_tweak = last_used_tweak.copy()
        else:
            cur_tweak = dict(
                yaw=INIT_YAW_DEG, pitch=INIT_PITCH_DEG, roll=INIT_ROLL_DEG,
                tx_cm=INIT_TX_CM, ty_cm=INIT_TY_CM, tz_cm=INIT_TZ_CM,
                zmax=80.0, autoZ=1, alpha=0.5
            )

        # ---- interactive loop ----
        while True:
            vals = get_trackbar_values("tune")
            cur_tweak.update(vals)

            T_camOpt_lidar, _ = tf_or_tweak_extrinsic(
                bag_path,
                cur_tweak["yaw"],cur_tweak["pitch"],cur_tweak["roll"],
                cur_tweak["tx_cm"],cur_tweak["ty_cm"],cur_tweak["tz_cm"]
            )

            zmax_preview = 120.0 if int(cur_tweak["autoZ"]) == 1 else cur_tweak["zmax"]

            # Single-scan preview (no fusion). Use splat-only projector.
            depth_m, d_mm, overlay = zbuffer_numpy_splat_only(
                rect, K_rect, xyz0, T_camOpt_lidar, zmax_preview,
                splat_gain=float(args.splat_gain), alpha=float(cur_tweak["alpha"])
            )

            depth_color = colorize_mm(d_mm)
            depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            side = np.hstack([overlay,
                              np.pad(depth_color_rgb, ((0,0),(10,10),(0,0)), constant_values=0)])
            cv2.imshow("tune", cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

            k = cv2.waitKey(PREVIEW_REFRESH_MS) & 0xFF

            if k == ord('q'):
                cv2.destroyAllWindows()
                progress["last_index"] = i
                save_progress(progress)
                print(f"\n[QUIT] Progress saved at index {i}.")
                return

            if k == ord('s'):
                qa_dir = (DEPTH_DIR / bagstem / "qa" / "preview")
                qa_dir.mkdir(parents=True, exist_ok=True)
                d_mm_save = (np.clip(np.nan_to_num(depth_m, nan=0.0),
                                     0, cur_tweak["zmax"]) * 1000.0).astype(np.uint16)
                cv2.imwrite(str(qa_dir / "preview_depth_z16.png"), d_mm_save)
                cv2.imwrite(str(qa_dir / "preview_depth_color.png"), colorize_mm(d_mm_save))
                cv2.imwrite(str(qa_dir / "preview_overlay.png"),
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                np.savez_compressed(
                    qa_dir / "preview_raw.npz",
                    depth_m=depth_m.astype(np.float32),
                    fx=K_rect[0,0], fy=K_rect[1,1], cx=K_rect[0,2], cy=K_rect[1,2],
                    yaw=cur_tweak["yaw"], pitch=cur_tweak["pitch"], roll=cur_tweak["roll"],
                    tx_cm=cur_tweak["tx_cm"], ty_cm=cur_tweak["ty_cm"], tz_cm=cur_tweak["tz_cm"],
                    splat_gain=float(args.splat_gain)
                )
                print("Saved preview snapshots.")

            if k == ord('u'):
                save_tweak_json(bagstem, cur_tweak)
                last_used_tweak = cur_tweak.copy()
                print("Tweak updated for this bag (no batch).")

            if k == ord('n'):
                cv2.destroyAllWindows()
                i += 1
                progress["last_index"] = i
                save_progress(progress)
                print("Skipped bag.")
                break

            if k == ord('r') or k == ord('c'):
                cv2.destroyAllWindows()
                save_tweak_json(bagstem, cur_tweak)
                last_used_tweak = cur_tweak.copy()

                print("Processing whole bag with current tweak …")
                frames = process_whole_bag(
                    bag_path, cur_tweak, mode_hint="manual_tweak",
                    splat_gain=float(args.splat_gain),
                    qa_height=int(args.qa_height) if args.qa_height else None
                )
                if frames > 0:
                    done_set.add(bagstem)
                    progress["done"] = sorted(done_set)
                i += 1
                progress["last_index"] = i
                save_progress(progress)
                print(f"!! Done. Next start index: {i}")
                break

    print("\nAll bags visited. Use --start-index or --bag to revisit specific bags if needed.")

if __name__ == "__main__":
    main()


# How to run
# # pip install rosbags opencv-python

# python make_depth_manual_tweak_test_1.py --splat-gain 2.0 --qa-height 960 --resume


# --splat-gain (default 1.8): raise to 2.2–2.6 for denser look; >3.0 can start to over-smear.

# --qa-height (visual only): set 0 to save overlays at full resolution.