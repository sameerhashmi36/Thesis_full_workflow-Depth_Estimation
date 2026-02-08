
# Monocular Depth Estimation for Agricultural Machinery
### LiDAR-Based Ground Truth Generation and Model Benchmarking

This repository contains the **complete end-to-end workflow** developed for the Master’s thesis:

The project focuses on building a **reproducible dataset preparation pipeline** from raw ROS bag recordings and benchmarking **state-of-the-art monocular depth estimation models** on real agricultural data.

---

## What This Repository Covers

This repository reflects the *entire research lifecycle*:

- Raw ROS bag processing (camera + LiDAR)
- LiDAR–camera projection and depth-map generation
- Dataset engineering and quality control
- Supervised fine-tuning of monocular depth models
- Quantitative and qualitative benchmarking
- Analysis of supervision strategies and evaluation protocols

---

## Models Evaluated

The following monocular depth estimation models are benchmarked under a **unified training and evaluation setup**:

- **Monodepth2** – self-supervised baseline
- **ZoeDepth** – metric depth estimation with adaptive bins
- **Depth Anything v3 (DA-v3)** – large-scale pretrained transformer-based model

Each model is evaluated in:
- Zero-shot (Baseline models)
- Sparse LiDAR-supervised finetuned models
- Smooth LiDAR-supervised finetuned models

---

## Repository Structure

```text
Thesis_full_workflow-Depth_Estimation/
├── Data_preparation/          # ROS bag processing & depth
├── monodepth2/                # Monodepth2 training & eval
├── ZoeDepth-1.0/              # ZoeDepth training & eval
├── Depth-Anything-3/          # Depth Anything v3 & eval
├── plotting_logs/             # Logs, plots
├── predicted_clips/           # Qualitative prediction 
└── README.md                  # Main project overview
```

Each submodule contains its **own detailed README** with usage instructions and dependencies.

---

## Dataset Preparation Pipeline

The dataset pipeline converts **raw ROS bag recordings** into training-ready RGB–depth pairs:

1. ROS1 → ROS2 bag conversion  
2. Camera image decoding and fisheye rectification  
3. LiDAR point cloud filtering and transformation  
4. LiDAR–camera projection with Z-buffering  
5. Sparse and smooth depth-map generation (16-bit PNG, metric scale)  
6. Bag-level train / validation / test splitting  

Key design choices:
- Bag-level splitting to avoid temporal leakage
- Sparse vs. smooth depth variants from the same sensor data
- Explicit validity masking for reliable supervision

Detailed documentation is available in `data_preparation/`.

---

## Training and Evaluation

All models are fine-tuned using:

- Identical bag-level dataset splits
- Masked depth losses (L1 + log + smoothness)
- Standard depth metrics:
  - Abs Rel
  - Sq Rel
  - RMSE
  - RMSE Log
  - δ₁ / δ₂ / δ₃

A **fixed set of 7 bags** is reserved for final testing across *all* experiments to ensure fair comparison.

---

## Qualitative Results (Video Predictions)

Each clip shows (left → right):
RGB input | Sparse GT (gray) | Predicted depth (gray) | Sparse GT (colored) | Predicted depth (colored)

### Monodepth2
https://github.com/sameerhashmi36/Thesis_full_workflow-Depth_Estimation/blob/main/predicted_clips/monodepth2_clip.mp4

### ZoeDepth
https://github.com/sameerhashmi36/Thesis_full_workflow-Depth_Estimation/blob/main/predicted_clips/zoedepth_clip.mp4

### Depth Anything v3
https://github.com/sameerhashmi36/Thesis_full_workflow-Depth_Estimation/blob/main/predicted_clips/DA-v3_clip.mp4


---

## Key Findings (High-Level)

- Sparse LiDAR supervision **significantly improves metric depth accuracy**
- Smooth supervision improves training stability but can **inflate relative-error metrics**
- Depth Anything v3 provides the **strongest zero-shot and fine-tuned performance**
- Evaluation protocol design (masking, cropping, GT choice) strongly affects reported results

<!-- --- -->

<!-- ## Thesis & Citation

If use or reference this work, please cite:

```
Sameer Aqib Hashmi,
Monocular Depth Estimation for Agricultural Machinery:
LiDAR-Based Ground Truth Generation and Model Benchmarking,
Master’s Thesis, Aalborg University, 2026.
``` -->

---

## Acknowledgments

This work was carried out in collaboration with **AGCO Corporation** and **Aalborg University**.  
Special thanks to academic and industrial supervisors for their guidance and support.
