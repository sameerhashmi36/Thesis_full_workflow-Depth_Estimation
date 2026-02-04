"""
AGCO + DA3 configuration.

This module centralizes:
- dataset roots
- fixed test bag list
- depth range used for masking metrics/loss
- default train resolution
- bag-level split logic (fixed test set + train/val from remaining bags)

Goal:
Keep split logic and constants in one place so training/eval always matches.
"""

from pathlib import Path
from typing import Tuple, List
import random

# --- Paths ---
DEFAULT_RAW_ROOT = Path("/media/sameer/ran_epav_disk/Thesis/bags_from_smb/data_preparation/raw_dataset_cpu_manual_1")
DEFAULT_OUTPUT_ROOT = Path("./models_finetuned_on_agco_smooth")

# Repo path is only needed if importing DA3 from a local clone via sys.path.
DEFAULT_DA3_REPO_ROOT = Path("/media/sameer/ran_epav_disk/Thesis/public_dataset_and_models/Depth-Anything-3")

# HuggingFace / local model id used by DepthAnything3.from_pretrained(...)
DEFAULT_DA3_MODEL_ID = "depth-anything/DA3-LARGE"

# --- Fixed test bags (AGCO) ---
AGCO_TEST_BAG_NAMES = [
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-54_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-29-24_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-35-57_3",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-36-57_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-45-53_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-24_1",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-31-54_10",
]

# --- Depth range used in training/eval masks ---
MIN_DEPTH_M = 1e-3
MAX_DEPTH_M = 80.0

# --- Default train resolution (override by CLI) ---
TRAIN_H = 224
TRAIN_W = 448

# --- Split seed (bag-level) ---
SPLIT_SEED = 42


def get_agco_bag_split(
    rectified_root: Path,
    val_bag_fraction: float = 0.2,
    seed: int = SPLIT_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Discover bag folders under rectified_root and split them into:
      - test_bags: fixed list (AGCO_TEST_BAG_NAMES)
      - train_bags/val_bags: bag-level split on remaining bags

    Returns:
      train_bags, val_bags, test_bags
    """
    assert rectified_root.is_dir(), f"rectified root not found: {rectified_root}"

    all_bags = sorted([p.name for p in rectified_root.iterdir() if p.is_dir()])

    test_bags = [b for b in all_bags if b in AGCO_TEST_BAG_NAMES]
    non_test_bags = [b for b in all_bags if b not in AGCO_TEST_BAG_NAMES]

    rng = random.Random(seed)
    rng.shuffle(non_test_bags)

    n_non_test = len(non_test_bags)
    if n_non_test == 0:
        return [], [], sorted(test_bags)

    n_val = max(1, int(round(n_non_test * float(val_bag_fraction))))
    val_bags = sorted(non_test_bags[:n_val])
    train_bags = sorted(non_test_bags[n_val:])

    print(f"[AGCO SPLIT] non-test bags : {len(non_test_bags)}")
    print(f"[AGCO SPLIT] train bags    : {len(train_bags)}")
    print(f"[AGCO SPLIT] val bags      : {len(val_bags)}")
    print(f"[AGCO SPLIT] test bags     : {len(test_bags)}")

    return train_bags, val_bags, sorted(test_bags)
