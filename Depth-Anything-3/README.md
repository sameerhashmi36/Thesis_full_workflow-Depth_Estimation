# Depth Anything v3 (DA3) – Evaluation and Fine-tuning

This directory contains custom scripts for zero-shot evaluation, LiDAR-supervised fine-tuning, quantitative evaluation, and visualization of Depth Anything v3 (DA3) on the AGCO agricultural dataset.

All scripts are built on top of the official Depth Anything v3 repository and are intended to be executed from the repository root.

Original repository:
https://github.com/ByteDance-Seed/Depth-Anything-3

The conda environment was created following the instructions provided in the official repository.

---

## Repository Requirement

All commands must be executed from the root of the Depth Anything v3 repository:

```bash
cd Depth-Anything-3
```

The following folders are custom additions placed directly in the repo root:

```
Depth-Anything-3/
├── evaluate_zeroshot/
└── finetune_eval_on_agco_dav3/
```

No core DA3 model code was modified.

---

## How to Run (Recommended Order)

### 1. Zero-shot Evaluation

```bash
python evaluate_zeroshot/eval_agco_DA3.py \
  --raw-root /path/to/raw_dataset_cpu_manual_1 \
  --model-id depth-anything/DA3-LARGE
```

Zero-shot visualization:

```bash
python evaluate_zeroshot/predicting_depth_map_zeroshot.py \
  --raw-root /path/to/raw_dataset_cpu_manual_1 \
  --model-id depth-anything/DA3-LARGE \
  --out-dir ./out_da3_zeroshot_vis
```

---

### 2. Fine-tuning on AGCO Dataset

```bash
python finetune_eval_on_agco_dav3/train_agco_da3_supervised.py \
  --raw-root /path/to/raw_dataset_cpu_manual_1 \
  --model-id depth-anything/DA3-LARGE \
  --train-fraction 0.5 \
  --val-fraction 0.2 \
  --output-root ./models_finetuned_on_agco_da3
```

---

### 3. Evaluation of Fine-tuned Model

Standard evaluation:

```bash
python finetune_eval_on_agco_dav3/eval_agco_da3_finetuned.py \
  --raw-root /path/to/raw_dataset_cpu_manual_1 \
  --model-id depth-anything/DA3-LARGE \
  --finetuned-weights ./models_finetuned_on_agco_da3/<run_name>/best.pth
```

ROI-based evaluation:

```bash
python finetune_eval_on_agco_dav3/eval_agco_da3_roi.py \
  --raw-root /path/to/raw_dataset_cpu_manual_1 \
  --model-id depth-anything/DA3-LARGE \
  --finetuned-weights ./models_finetuned_on_agco_da3/<run_name>/best.pth \
  --roi-frac 0.333
```

---

### 4. Prediction and Visualization (Fine-tuned)

```bash
python finetune_eval_on_agco_dav3/predicting_depth_maps_da3.py \
  --raw-root /path/to/raw_dataset_cpu_manual_1 \
  --model-id depth-anything/DA3-LARGE \
  --finetuned-weights ./models_finetuned_on_agco_da3/<run_name>/best.pth \
  --out-dir ./out_da3_finetuned_vis
```

---

## Dataset and Configuration

- agco_da3_config.py  
  Central configuration for dataset paths, fixed test bags, depth range, and bag-level split logic.

- agco_da3_dataset.py  
  Sparse LiDAR depth dataset.

- agco_da3_dataset_smooth.py  
  Smoothed / filled depth dataset (Telea-limited filling + optional bilateral filtering).

---

## Reference

Thesis:
Monocular Depth Estimation with LiDAR Supervision: Automated Dataset Preparation and Model Evaluation for Agricultural Machinery
