"""
utils_tf.py
- Normalizes all frame ids (strips leading '/')
- Builds a bidirectional TF graph from /tf_static and /tf
- Resolves transforms and returns both matrix and the path used
- Robust autodetection of lidar + camera frames
"""

import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rosbags.typesys import Stores, get_typestore
from utils_ros2 import iter_topic_messages

TS = get_typestore(Stores.ROS2_FOXY)

# ---------- math helpers ----------
def quat_xyzw_to_R(x,y,z,w):
    n = math.sqrt(x*x+y*y+z*z+w*w)
    if n > 0: x,y,z,w = x/n,y/n,z/n,w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], dtype=np.float64)

def compose_T(R,t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = np.asarray(t, np.float64)
    return T

def inv_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

# ---------- helpers ----------
def _norm(fid: str) -> str:
    """Normalize frame id: strip any leading '/'."""
    fid = fid or ""
    return fid[1:] if fid.startswith("/") else fid

def _collect_tf_list(bag_root: Path, topic: str) -> List[Dict]:
    out: List[Dict] = []
    for _, raw, typ in iter_topic_messages(bag_root, topic):
        msg = TS.deserialize_cdr(raw, typ)  # tf2_msgs/msg/TFMessage
        for tr in getattr(msg, "transforms", []):
            out.append(dict(
                parent = _norm(tr.header.frame_id),
                child  = _norm(tr.child_frame_id),
                tx     = float(tr.transform.translation.x),
                ty     = float(tr.transform.translation.y),
                tz     = float(tr.transform.translation.z),
                qx     = float(tr.transform.rotation.x),
                qy     = float(tr.transform.rotation.y),
                qz     = float(tr.transform.rotation.z),
                qw     = float(tr.transform.rotation.w),
            ))
    return out

def _build_edges(tf_list: List[Dict]):
    """Bidirectional adjacency: edges[src] -> list of (dst, T_dst_src)."""
    edges: Dict[str, List[Tuple[str, np.ndarray]]] = {}

    def add_edge(a: str, b: str, T_b_a: np.ndarray):
        edges.setdefault(a, []).append((b, T_b_a))

    for r in tf_list:
        R = quat_xyzw_to_R(r["qx"], r["qy"], r["qz"], r["qw"])
        t = np.array([r["tx"], r["ty"], r["tz"]], np.float64)
        T_parent_child = compose_T(R, t)
        # Store both directions
        add_edge(r["parent"], r["child"], T_parent_child)        # child <- parent
        add_edge(r["child"],  r["parent"], inv_T(T_parent_child))# parent <- child
    return edges

def _resolve_T_with_path(edges, src: str, dst: str) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Return (T_dst_src, path_nodes). If src==dst -> (I, [src]).
    BFS over edges; each edge transform maps current->neighbor in dst's frame composition.
    """
    src, dst = _norm(src), _norm(dst)
    if src == dst:
        return np.eye(4, dtype=np.float64), [src]

    from collections import deque
    q = deque()
    q.append((src, np.eye(4, dtype=np.float64), [src]))
    visited = {src}

    while q:
        cur, T_dst_cur, path = q.popleft()
        for (nbr, T_cur_nbr) in edges.get(cur, []):
            if nbr in visited:
                continue
            T_dst_nbr = T_dst_cur @ T_cur_nbr
            if nbr == dst:
                return T_dst_nbr, path + [nbr]
            visited.add(nbr)
            q.append((nbr, T_dst_nbr, path + [nbr]))
    return None, []

def _first_header_frame_id(bag_root: Path, topic: str) -> Optional[str]:
    for _, raw, typ in iter_topic_messages(bag_root, topic):
        msg = TS.deserialize_cdr(raw, typ)
        hdr = getattr(msg, "header", None)
        if hdr and getattr(hdr, "frame_id", ""):
            return _norm(hdr.frame_id)
        break
    return None

def _neighbors(edges, node: str) -> List[str]:
    return [n for (n, _) in edges.get(_norm(node), [])]

# ---------- main ----------
def pick_tf_from_bag_robust(bag_root: Path,
                            cam_topic: str = "/cam_lucid_front/image_raw",
                            lidar_topic: str = "/ouster_front/ouster/points",
                            camera_name_hint: str = "cam_lucid_front",
                            lidar_hint: str = "ouster") -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    """
    Returns (T_camOpt_lidar, path_camBase<-lidar, debug_neighbors)
      - debug_neighbors: dict of near-neighbors to aid debug prints
    Strategy:
      * Normalize all frames.
      * Pick lidar frame:
          - prefer /tf_static child with lidar_hint and 'os_lidar'/'os_sensor'
          - else use header.frame_id from lidar_topic
      * Pick camera base:
          - prefer /tf_static child containing camera_name_hint
          - else derive from image header (strip '_optical')
      * cam_base -> cam_optical:
          - if explicit TF exists, resolve it; else REP-103
      * Resolve cam_base <- lidar via graph; compose.
    """
    tf_static = _collect_tf_list(bag_root, "/tf_static")
    tf_dyn    = _collect_tf_list(bag_root, "/tf")
    tf_all    = tf_static + tf_dyn
    edges     = _build_edges(tf_all)

    # Candidate list of frames for prints
    frames = sorted({r["parent"] for r in tf_all} | {r["child"] for r in tf_all})

    # --- LiDAR frame
    lidar_frame = None
    cand_l = [r for r in tf_static
              if (lidar_hint in r["child"]) and ("os_lidar" in r["child"] or "os_sensor" in r["child"])]
    if cand_l:
        lidar_frame = cand_l[0]["child"]
    if lidar_frame is None:
        lf = _first_header_frame_id(bag_root, lidar_topic)
        if lf:
            lidar_frame = lf
    if not lidar_frame:
        raise RuntimeError("Could not determine LiDAR frame (no TF and no header.frame_id).")

    # --- Camera base
    cam_base = None
    cand_c = [r for r in tf_static if (camera_name_hint in r["child"])]
    if cand_c:
        cam_base = cand_c[0]["child"]
    else:
        img_f = _first_header_frame_id(bag_root, cam_topic)
        if not img_f:
            raise RuntimeError("Could not determine camera base frame (no TF child and no image header.frame_id).")
        cam_base = img_f[:-len("_optical")] if img_f.endswith("_optical") else img_f

    # --- cam_base -> cam_optical
    cam_opt = f"{cam_base}_optical"
    explicit_camopt = any((r["parent"] == cam_base and r["child"] == cam_opt) for r in tf_all)
    if explicit_camopt:
        T_camOpt_camBase, path_cb_co = _resolve_T_with_path(edges, cam_base, cam_opt)
        if T_camOpt_camBase is None:
            raise RuntimeError(f"Found {cam_base}->{cam_opt} in TF but could not resolve it.")
        used_opt = f"explicit: {' -> '.join(path_cb_co)}"
    else:
        # REP-103
        R = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=np.float64)
        T_camOpt_camBase = compose_T(R, np.zeros(3, np.float64))
        used_opt = "REP-103 default"

    # --- resolve cam_base <- lidar
    T_camBase_lidar, path_lb_cb = _resolve_T_with_path(edges, lidar_frame, cam_base)
    if T_camBase_lidar is None:
        # Try common alias: if lidar_frame has '/os_sensor' but graph only has '/os_lidar'
        if lidar_frame.endswith("/os_sensor"):
            alt = lidar_frame[:-len("/os_sensor")] + "/os_lidar"
            T_camBase_lidar, path_lb_cb = _resolve_T_with_path(edges, alt, cam_base)
            if T_camBase_lidar is not None:
                lidar_frame = alt
        # Or if cam_base has leading slash variant
    if T_camBase_lidar is None:
        # give detailed diag
        dbg = {
            "lidar_frame": lidar_frame,
            "cam_base": cam_base,
            "neighbors_lidar": _neighbors(edges, lidar_frame),
            "neighbors_cam_base": _neighbors(edges, cam_base),
            "all_frames": frames[:],
        }
        raise RuntimeError(
            "TF resolve failed: No path from LiDAR to camera base.\n"
            f"  lidar_frame     : {dbg['lidar_frame']}\n"
            f"  cam_base        : {dbg['cam_base']}\n"
            f"  neighbors(lidar): {dbg['neighbors_lidar']}\n"
            # f"  neighbors(cam)  : {dbg['neighbors_cam_base']}\n"
            f"  frames_in_bag   : {dbg['all_frames']}\n"
            "Likely causes: bag has no LiDAR TF edges, LiDAR topic absent, or mismatched frame names."
        )

    T_camOpt_lidar = T_camOpt_camBase @ T_camBase_lidar

    debug_neighbors = {
        "neighbors(lidar)": _neighbors(edges, lidar_frame),
        "neighbors(cam_base)": _neighbors(edges, cam_base),
    }

    print(f"[TF] lidar_frame        : {lidar_frame}")
    print(f"[TF] camera_base_frame  : {cam_base}")
    print(f"[TF] cam_base->optical  : {used_opt}")
    print(f"[TF] path(lidar->camBase): {' -> '.join(path_lb_cb)}")

    return T_camOpt_lidar, path_lb_cb, debug_neighbors
