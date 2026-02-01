"""
utils_ros2.py

Convenience helpers for reading ROS2 (.db3) bags with rosbags:
- iter_topic_messages(bag_root, topic)
- iter_cam_messages(bag_root, cam_topic)
- load_all_lidar(bag_root, lidar_topic, offset=0.0)
- list_topics(bag_root) -> set[str]
- has_topic(bag_root, topic) -> bool
- count_messages(bag_root, topic) -> int

"""

from pathlib import Path
from typing import Iterator, Tuple, List, Set, Optional
import numpy as np

from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)


def _open_reader(bag_root: Path) -> Rosbag2Reader:
    reader = Rosbag2Reader(bag_root)
    reader.open()
    return reader


def list_topics(bag_root: Path) -> Set[str]:
    """Return a set of topic names present in the bag."""
    reader = _open_reader(bag_root)
    try:
        return {c.topic for c in reader.connections}
    finally:
        reader.close()


def has_topic(bag_root: Path, topic: str) -> bool:
    """True if a topic exists in the bag (exact match)."""
    reader = _open_reader(bag_root)
    try:
        for c in reader.connections:
            if c.topic == topic:
                return True
        return False
    finally:
        reader.close()


def count_messages(bag_root: Path, topic: str, max_stop: Optional[int] = None) -> int:
    """Count messages on a topic. If max_stop is set, stop after that many for speed."""
    reader = _open_reader(bag_root)
    try:
        cnt = 0
        conns = [c for c in reader.connections if c.topic == topic]
        for c in conns:
            for _seq, _t, _raw in reader.messages(connections=[c]):
                cnt += 1
                if max_stop is not None and cnt >= max_stop:
                    return cnt
        return cnt
    finally:
        reader.close()


def iter_topic_messages(bag_root: Path, topic: str) -> Iterator[Tuple[float, bytes, str]]:
    """
    Yield (t_sec, raw, msgtype) for all messages on a given topic in time order.
    """
    reader = _open_reader(bag_root)
    try:
        msgs: List[Tuple[float, bytes, str]] = []
        conns = [c for c in reader.connections if c.topic == topic]
        for c in conns:
            for _seq, t, raw in reader.messages(connections=[c]):
                msgs.append((t * 1e-9, raw, c.msgtype))
        # sort by time (bags can have multiple connections)
        msgs.sort(key=lambda x: x[0])
        for item in msgs:
            yield item
    finally:
        reader.close()


def iter_cam_messages(bag_root: Path, cam_topic: str) -> Iterator[Tuple[float, bytes, str]]:
    """
    Yield (t_sec, raw, msgtype) for camera messages (wrapper around iter_topic_messages).
    """
    yield from iter_topic_messages(bag_root, cam_topic)


def load_all_lidar(bag_root: Path, lidar_topic: str, offset: float = 0.0):
    """
    Return (times_sec_list, raws_list, types_list),
    times already offset by `offset`.
    """
    reader = _open_reader(bag_root)
    try:
        times, raws, types = [], [], []
        conns = [c for c in reader.connections if c.topic == lidar_topic]
        for c in conns:
            for _seq, t, raw in reader.messages(connections=[c]):
                times.append(t * 1e-9 + offset)
                raws.append(raw)
                types.append(c.msgtype)
        # sort together by time
        order = np.argsort(times)
        times = [times[i] for i in order]
        raws  = [raws[i]  for i in order]
        types = [types[i] for i in order]
        return times, raws, types
    finally:
        reader.close()
