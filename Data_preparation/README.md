# Data Preparation

This folder contains all scripts used to prepare the dataset for this thesis. The goal is to convert raw ROS recordings into **rectified RGB images** and **LiDAR-based depth maps** that can be used for training and evaluating monocular depth estimation models.

The pipeline is mostly automatic, but includes a **small manual alignment step** to fix real-world sensor inaccuracies.

---

## What is the overall idea?

```
ROS1 .bag files
   ↓
Convert to ROS2 (.db3 + metadata)
   ↓
Keep only bags with LiDAR
   ↓
Rectify camera images
   ↓
Project LiDAR to image → depth maps
   (manual tweaks if needed)
```

Final output is saved in:

```
raw_dataset_cpu_manual_1/
  ├─ rectified/   (RGB images)
  ├─ depth_z16/   (depth maps, 16‑bit PNG)
  └─ summaries/   (logs, tweaks, progress)
```

---

## Step 1 – Convert ROS1 bags to ROS2

**File:** `convert_bags_to_db3.py`

Converts each ROS1 `.bag` file into a ROS2 bag folder containing `.db3` files and `metadata.yaml`.

Run:

```bash
python convert_bags_to_db3.py
```

This makes the data readable using modern Python tools.

---

## Step 2 – Keep only bags with LiDAR

**File:** `find_bags_with_lidar.py`

Some recordings do not contain usable LiDAR data. This script keeps only bags that:

* have a PointCloud2 LiDAR topic
* contain at least one LiDAR message

Run:

```bash
python find_bags_with_lidar.py
```

After this step, all remaining bags are guaranteed to have LiDAR.

---

## Step 3 – Rectify fisheye camera images

**File:** `make_rectified_from_db3.py`

Camera images are rectified using known fisheye calibration parameters.

Run:

```bash
python make_rectified_from_db3.py
```

Output:

```
raw_dataset_cpu_manual_1/rectified/<bag_name>/
```

---

## Step 4 – Generate depth maps (with manual tweaks)

**File:** `make_depth_manual_tweak_test_1.py`

This script projects LiDAR points onto the rectified images to create depth maps.

In practice, small sensor misalignments exist even when TF data is available. To handle this, the script allows **small manual adjustments** to the LiDAR–camera alignment.

Run:

```bash
python make_depth_manual_tweak_test_1.py
```

---

## How do the manual tweaks work?

For each bag, it can slightly adjust:

* yaw, pitch, roll (degrees)
* x, y, z translation (centimeters)

An interactive window shows the LiDAR projection over the image. Adjust sliders until the alignment looks correct.

Key controls:

* `c` → confirm and process the bag
* `s` → save preview only
* `n` → skip bag
* `q` → quit (progress is saved)

The chosen values are stored automatically in:

```
raw_dataset_cpu_manual_1/summaries/tweaks/<bag_name>.json
```

These tweaks are reused if the script is run again.

---

## Depth map details

* Format: 16‑bit PNG
* Units: millimeters
* Invalid pixels: value = 0
* Method: nearest LiDAR point (Z‑buffer)

Depth maps are saved in:

```
raw_dataset_cpu_manual_1/depth_z16/<bag_name>/
```

---

## Helper files

* `utils_ros2.py` – reading ROS2 bags safely
* `utils_tf.py` – resolving TF transforms
* `utils_vision.py` – image decoding, rectification, LiDAR projection

