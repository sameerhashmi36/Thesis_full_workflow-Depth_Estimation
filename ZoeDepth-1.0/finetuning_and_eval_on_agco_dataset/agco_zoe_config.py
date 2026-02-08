"""
Config + helpers for finetuning ZoeDepth on the AGCO dataset.

- Defines DEFAULT_RAW_ROOT (where rectified/ and depth_z16/ live)
- Defines AGCO_TEST_BAG_NAMES (fixed test set, same 7 bags as before)
- Provides get_agco_bag_split(...) to split non-test bags into train/val.
"""

from pathlib import Path
from typing import List, Tuple
import random

# Adjust if needed, or override via --raw-root CLI arg in the trainer
DEFAULT_RAW_ROOT = Path(
    "path/to/dataset/"
    "raw_dataset_cpu_manual_1"
)

# 7 test bags
AGCO_TEST_BAG_NAMES = [
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-54_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-29-24_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-35-57_3",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-36-57_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-45-53_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-24_1",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-31-54_10",
]

MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0


def get_agco_bag_split(
    rectified_root: Path,
    val_bag_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """
    Discover all bag folders under rectified_root and split them into:
      - train_bags (non-test)
      - val_bags   (non-test)
      - test_bags  (fixed list AGCO_TEST_BAG_NAMES)

    val_bag_fraction controls how many non-test bags go to validation.

    Returns:
      train_bags, val_bags, test_bags  (each is a sorted list of bag names)
    """
    assert rectified_root.is_dir(), f"rectified root not found: {rectified_root}"

    all_bags = sorted([p.name for p in rectified_root.iterdir() if p.is_dir()])

    test_bags = [b for b in all_bags if b in AGCO_TEST_BAG_NAMES]
    non_test_bags = [b for b in all_bags if b not in AGCO_TEST_BAG_NAMES]

    rng = random.Random(seed)
    rng.shuffle(non_test_bags)

    n_non_test = len(non_test_bags)
    n_val = max(1, int(round(n_non_test * val_bag_fraction)))
    val_bags = sorted(non_test_bags[:n_val])
    train_bags = sorted(non_test_bags[n_val:])

    print(f"[AGCO SPLIT] non-test bags : {len(non_test_bags)}")
    print(f"[AGCO SPLIT] train bags    : {len(train_bags)}")
    print(f"[AGCO SPLIT] val bags      : {len(val_bags)}")

    return train_bags, val_bags, sorted(test_bags)