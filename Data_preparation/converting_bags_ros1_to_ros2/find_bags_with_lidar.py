"""
Keep-only PointCloud2 bags:
- KEEP a bag folder if it has LIDAR_TOPIC with type == sensor_msgs/msg/PointCloud2 and >=1 message.
- Copy kept folders SRC_DIR/<bagfolder> → DST_DIR/<bagfolder>
- (Optional) delete skipped folders from SRC_DIR.
- Also summarizes presence of camera and TF topics (informational only).

Reports: prune_summary.txt / prune_summary.json (written to DST_DIR)

Requires: pip install rosbags
"""

import json, shutil, sys
from pathlib import Path
from typing import Dict, Set

from rosbags.rosbag2 import Reader as Rosbag2Reader

# ---------- CONFIG ----------
SRC_DIR     = Path("src/path")
DST_DIR     = Path("dstination/path")

LIDAR_TOPIC = "/ouster_front/ouster/points"
REQUIRED_LIDAR_TYPE = "sensor_msgs/msg/PointCloud2"

# Optional summaries (not used for keep/skip)
CAM_TOPIC   = "/cam_lucid_front/image_raw"
# ------------------------------------------

DELETE_SKIPPED_FROM_SRC = False   # True → remove skipped folders from SRC
DRY_RUN                  = False  # True → only print/report, no copy/delete


def bag_folders(root: Path):
    """Yield bag folder paths (must have metadata.yaml and at least one *.db3)."""
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "metadata.yaml").exists() and list(p.glob("*.db3")):
            yield p


def topics_and_types(bag_root: Path) -> Dict[str, Set[str]]:
    """Return {topic: {msgtypes}} or {} if open fails."""
    try:
        r = Rosbag2Reader(bag_root); r.open()
        try:
            out: Dict[str, Set[str]] = {}
            for c in r.connections:
                out.setdefault(c.topic, set()).add(c.msgtype)
            return out
        finally:
            r.close()
    except Exception:
        return {}


def topic_has_any_message(bag_root: Path, topic: str, max_peek: int = 1) -> bool:
    """True if topic has at least one message."""
    try:
        r = Rosbag2Reader(bag_root); r.open()
        try:
            conns = [c for c in r.connections if c.topic == topic]
            if not conns:
                return False
            n = 0
            for _cid, _t, _raw in r.messages(connections=conns):
                n += 1
                if n >= max_peek:
                    return True
            return False
        finally:
            r.close()
    except Exception:
        return False


def decide(bag_root: Path):
    """Return info dict with keep/skip decision based ONLY on PointCloud2 topic."""
    ttypes = topics_and_types(bag_root)
    topics = set(ttypes.keys())

    has_lidar_topic = LIDAR_TOPIC in topics
    lidar_type_ok = False
    lidar_has_data = False
    if has_lidar_topic:
        types_here = ttypes.get(LIDAR_TOPIC, set())
        lidar_type_ok = (REQUIRED_LIDAR_TYPE in types_here) or any(t.endswith("PointCloud2") for t in types_here)
        lidar_has_data = topic_has_any_message(bag_root, LIDAR_TOPIC)

    # Informational summaries
    has_cam       = CAM_TOPIC in topics
    has_tf        = "/tf" in topics
    has_tf_static = "/tf_static" in topics
    db3_count     = len(list(bag_root.glob("*.db3")))

    if has_lidar_topic and lidar_type_ok and lidar_has_data:
        status, reason = "keep", "pointcloud2_present_with_data"
    else:
        if not has_lidar_topic:
            reason = "missing_lidar_topic"
        elif not lidar_type_ok:
            reason = "wrong_lidar_type"
        else:
            reason = "empty_lidar_topic"
        status = "skip"

    return dict(
        bag=bag_root.name,
        status=status,
        reason=reason,
        has_lidar_topic=has_lidar_topic,
        lidar_type_ok=lidar_type_ok,
        lidar_has_data=lidar_has_data,
        # extras for summary only:
        has_camera=has_cam,
        has_tf=has_tf,
        has_tf_static=has_tf_static,
        db3_files=db3_count,
    )


def copy_folder(src: Path, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst_root / src.name, dirs_exist_ok=True)


def main():
    if not SRC_DIR.exists():
        print(f"SRC_DIR not found: {SRC_DIR}"); sys.exit(1)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    keep_paths, skip_paths = [], []

    print(f"Scanning bag folders under: {SRC_DIR}\n")
    for bag in bag_folders(SRC_DIR):
        info = decide(bag)
        results.append(info)

        tag = " keep" if info["status"] == "keep" else " skip"
        print(f"- {info['bag']}: {tag} "
              f"[pc2_topic={info['has_lidar_topic']}, type_ok={info['lidar_type_ok']}, data={info['lidar_has_data']}; "
              f"cam={info['has_camera']}, tf={info['has_tf']}, tf_static={info['has_tf_static']}] → {info['reason']}")

        (keep_paths if info["status"] == "keep" else skip_paths).append(bag)

    print("\nSummary:")
    print(f"  Total: {len(results)}")
    print(f"  Keep : {len(keep_paths)}")
    print(f"  Skip : {len(skip_paths)}")

    if DRY_RUN:
        print("\nDRY_RUN=True → no copy/delete.")
    else:
        # Copy kept
        copied = 0
        for bag in keep_paths:
            try:
                copy_folder(bag, DST_DIR)
                copied += 1
            except Exception as e:
                print(f"[WARN] failed to copy {bag.name}: {e}")
        print(f"Copied {copied}/{len(keep_paths)} kept folders → {DST_DIR}")

        # Optional delete skipped from SRC
        if DELETE_SKIPPED_FROM_SRC:
            deleted = 0
            for bag in skip_paths:
                try:
                    shutil.rmtree(bag, ignore_errors=False)
                    deleted += 1
                except Exception as e:
                    print(f"[WARN] failed to delete {bag.name}: {e}")
            print(f"Deleted {deleted}/{len(skip_paths)} skipped folders from SRC_DIR.")

    # Reports (to DST_DIR)
    txt = [
        f"SRC_DIR: {SRC_DIR}",
        f"DST_DIR: {DST_DIR}",
        f"Total: {len(results)}",
        f"Keep : {len(keep_paths)}",
        f"Skip : {len(skip_paths)}",
        "",
        "Per-bag:"
    ]
    for r in results:
        txt.append(
            f"- {r['bag']}: {r['status']} [{r['reason']}] "
            f"(pc2_topic={r['has_lidar_topic']}, type_ok={r['lidar_type_ok']}, data={r['lidar_has_data']}, "
            f"cam={r['has_camera']}, tf={r['has_tf']}, tf_static={r['has_tf_static']}, db3={r['db3_files']})"
        )
    (DST_DIR / "prune_summary.txt").write_text("\n".join(txt), encoding="utf-8")
    with open(DST_DIR / "prune_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nReports written to:\n  {DST_DIR}/prune_summary.txt\n  {DST_DIR}/prune_summary.json")


if __name__ == "__main__":
    main()
