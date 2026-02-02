"""
Configuration helpers for AGCO finetuning and evaluation.

This module centralizes:
- default raw dataset root
- AGCO test bag names (fixed test set)
- default train/val split fraction
- default random seed

The idea is to keep these consistent across training and evaluation scripts.
"""

from pathlib import Path

DEFAULT_RAW_ROOT = Path(
    "/path/to/dataset/raw_dataset_cpu_manual_1"
)

AGCO_TEST_BAG_NAMES = [
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-54_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-29-24_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-35-57_3",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-36-57_5",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-45-53_2",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-27-24_1",
    "tractor_fendt_1038_961_23_0008_syslogic_orin_nx_17187910_log_2024-04-16-13-31-54_10",
]

# Bag-level split among NON-test bags
DEFAULT_VAL_BAG_FRACTION = 0.2

# Frame-level subsampling defaults
DEFAULT_TRAIN_FRACTION = 1.0
DEFAULT_VAL_FRACTION = 1.0
DEFAULT_TEST_FRACTION = 1.0

DEFAULT_SEED = 42

# Depth range used for training/metrics (meters)
MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0
