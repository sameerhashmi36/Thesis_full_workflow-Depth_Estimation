"""
Convert ROS1 .bag files to ROS2 rosbag2 folders (with metadata.yaml kept).

For each <name>.bag in SRC_DIR:
  - remove DST_DIR/<name> if it exists (clean re-run)
  - run: rosbags-convert --src <bag> --dst DST_DIR/<name>
  - verify: metadata.yaml & at least one *.db3 present
  - parse metadata.yaml → write topics.txt + info.txt (has_tf, has_tf_static, has_cam, has_lidar)

At end, write DST_DIR/summary_report.txt with a concise overview.

Requires:
  pip install rosbags pyyaml
"""

import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml  # pip install pyyaml

# -------- CONFIG --------
SRC_DIR = Path("path/to/ros1bag/dir")
DST_DIR = Path("destination/path/2024_04_16_db3_files")

ROSBAGS_CONVERT = "rosbags-convert"

# (Adjust to the naming if needed to these convenience checks)
CAM_TOPIC_HINTS   = ["/cam_lucid_front/image_raw", "/cam_lucid_front/image"]
LIDAR_TOPIC_HINTS = ["/ouster_front/ouster/points", "/ouster/points", "/os_cloud_node/points"]
# ------------------------


def run(cmd: List[str]) -> int:
    print(">>>", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, check=False).returncode


def clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def parse_topics_from_metadata(meta_path: Path) -> List[Tuple[str, str]]:
    """
    Return list of (name, type) from metadata.yaml (rosbag2 format).
    """
    data = yaml.safe_load(meta_path.read_text())
    topics = []
    # Robust parse across rosbags versions:
    # Most use: data['rosbag2_bagfile_information']['topics_with_message_count']
    root = data.get("rosbag2_bagfile_information") or data
    tlist = root.get("topics_with_message_count") or root.get("topics") or []
    for entry in tlist:
        name = entry.get("topic_metadata", {}).get("name") or entry.get("name")
        typ  = entry.get("topic_metadata", {}).get("type") or entry.get("type")
        if name and typ:
            topics.append((name, typ))
    return topics


def bag_has_any(topics: List[Tuple[str, str]], names: List[str]) -> bool:
    tnames = {t[0] for t in topics}
    return any(n in tnames for n in names)


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    bags = sorted(SRC_DIR.glob("*.bag"))
    if not bags:
        print(f" No .bag files in {SRC_DIR}")
        sys.exit(1)

    print(f"Found {len(bags)} bag(s). Output → {DST_DIR.resolve()}")

    results: Dict[str, Dict[str, str]] = {}
    ok, failed = 0, 0

    for bag in bags:
        name = bag.stem
        out_dir = DST_DIR / name
        meta = out_dir / "metadata.yaml"

        print(f"\n[Bag] {bag.name}")
        try:
            # Clean previous attempt
            clean_dir(out_dir)

            # Convert directly into final folder
            rc = run([ROSBAGS_CONVERT, "--src", str(bag), "--dst", str(out_dir)])
            if rc != 0:
                results[name] = {"status": "convert_error", "note": "rosbags-convert rc != 0"}
                print(f"[ERR] conversion failed for {bag.name}")
                failed += 1
                continue

            # Verify outputs
            if not meta.exists():
                results[name] = {"status": "missing_metadata", "note": "metadata.yaml not found"}
                print("[ERR] metadata.yaml missing")
                failed += 1
                continue

            db3s = sorted(out_dir.rglob("*.db3"))
            if not db3s:
                results[name] = {"status": "missing_db3", "note": "no .db3 produced"}
                print("[ERR] no .db3 produced")
                failed += 1
                continue

            # Parse topics → topics.txt + info.txt
            topics = parse_topics_from_metadata(meta)
            (out_dir / "topics.txt").write_text(
                "\n".join([f"{n}  [{t}]" for (n, t) in topics]) + ("\n" if topics else "")
            )
            has_tf        = any(n == "/tf" for (n, _) in topics)
            has_tf_static = any(n == "/tf_static" for (n, _) in topics)
            has_cam       = bag_has_any(topics, CAM_TOPIC_HINTS)
            has_lidar     = bag_has_any(topics, LIDAR_TOPIC_HINTS)

            info_lines = [
                f"bag: {bag.name}",
                f"db3_files: {len(db3s)}",
                f"has_tf: {has_tf}",
                f"has_tf_static: {has_tf_static}",
                f"has_camera: {has_cam}",
                f"has_lidar: {has_lidar}",
            ]
            (out_dir / "info.txt").write_text("\n".join(info_lines) + "\n")

            results[name] = {
                "status": "ok",
                "db3_files": str(len(db3s)),
                "has_tf": str(has_tf),
                "has_tf_static": str(has_tf_static),
                "has_camera": str(has_cam),
                "has_lidar": str(has_lidar),
            }
            print(f"[OK] {bag.name} → {out_dir.name} (db3={len(db3s)}, tf={has_tf}, tf_static={has_tf_static})")
            ok += 1

        except Exception as e:
            results[name] = {"status": "exception", "note": f"{type(e).__name__}: {e}"}
            print(f"[ERR] {bag.name} → {type(e).__name__}: {e}")
            failed += 1

    # Write final summary
    lines = [
        f"Total bags: {len(bags)}",
        f"Converted OK: {ok}",
        f"Failed: {failed}",
        "",
        "Per-bag status:",
    ]
    for name in sorted(results.keys()):
        r = results[name]
        line = f"- {name}: {r.get('status')}"
        extra = []
        for k in ("db3_files", "has_tf", "has_tf_static", "has_camera", "has_lidar", "note"):
            if k in r:
                extra.append(f"{k}={r[k]}")
        if extra:
            line += "  [" + ", ".join(extra) + "]"
        lines.append(line)

    summary_path = DST_DIR / "summary_report.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n===== SUMMARY =====")
    print("\n".join(lines))
    print(f"\nReport saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
