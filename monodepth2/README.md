# Monodepth2 – AGCO Fine-tuning and Evaluation

This directory contains custom scripts for training, evaluation, and visualization of Monodepth2 using LiDAR-supervised depth data from agricultural machinery. The implementation is built on top of the official Monodepth2 repository and should be run from its root directory.

---

##  Repository Requirement

All scripts must be executed from the **root of the official Monodepth2 repository**:

```bash
cd monodepth2
```

This ensures imports like the following work properly:
```python
import networks
from layers import disp_to_depth
```

Only custom folders and scripts were added. The original Monodepth2 model code remains untouched.

---

##  Environment

The Python environment was created using the official Monodepth2 setup. No architectural changes were made to the base model.

---

##  Folder Overview

```
monodepth2/
├── networks/                          # Original model
├── layers.py                         # Original layers
├── monodepth2_eval_zeroshot/         # Baseline zero-shot scripts
└── Finetuning_and_eval_on_agco_dataset/
    ├── train_finetune_monodepth2_agco.py
    ├── eval_finetuned_monodepth2_agco.py
    ├── eval_finetuned_monodepth2_agco_roi.py
    ├── predicting_depth_maps.py
    ├── agco_dataset.py
    ├── agco_dataset_smooth.py
    └── agco_config.py
```

---

##  How to Run

### 1️ Zero-shot Evaluation (Baseline)

**Folder:** `monodepth2_eval_zeroshot/`

**Metric evaluation (zero-shot):**
```bash
python monodepth2_eval_zeroshot/eval_agco_monodepth2.py   --raw-root /path/to/raw_dataset_cpu_manual_1
```

**Visualization (prediction only):**
```bash
python monodepth2_eval_zeroshot/predicting_depth_maps_monodepth2_zeroshot.py   --raw-root /path/to/raw_dataset_cpu_manual_1   --out-dir ./out_zeroshot_vis
```

---

### 2️ Fine-tune Monodepth2 on AGCO Dataset

**Folder:** `Finetuning_and_eval_on_agco_dataset/`

```bash
python Finetuning_and_eval_on_agco_dataset/train_finetune_monodepth2_agco.py   --raw-root /path/to/raw_dataset_cpu_manual_1   --output-folder ./models_finetuned_on_agco   --train-fraction 0.5   --val-fraction 0.2
```

This script performs LiDAR-supervised training and saves:
```
models_finetuned_on_agco/
├── encoder.pth
└── depth.pth
```

---

### 3️ Evaluate Fine-tuned Model

**Standard LiDAR evaluation:**
```bash
python Finetuning_and_eval_on_agco_dataset/eval_finetuned_monodepth2_agco.py   --raw-root /path/to/raw_dataset_cpu_manual_1   --load-weights-folder ./models_finetuned_on_agco
```

**ROI-based evaluation (bottom image region):**
```bash
python Finetuning_and_eval_on_agco_dataset/eval_finetuned_monodepth2_agco_roi.py   --raw-root /path/to/raw_dataset_cpu_manual_1   --load-weights-folder ./models_finetuned_on_agco
```

---

### 4️ Predict and Visualize Depth Maps

```bash
python Finetuning_and_eval_on_agco_dataset/predicting_depth_maps.py   --raw-root /path/to/raw_dataset_cpu_manual_1   --load-weights-folder ./models_finetuned_on_agco   --out-dir ./out_finetuned_vis
```

Outputs stacked RGB–GT–Prediction images or videos depending on the mode used.

---

## Dataset and Config Files

- `agco_config.py`: Holds paths, bag list, train/val splits, and depth ranges
- `agco_dataset.py`: Dataset using sparse LiDAR depth
- `agco_dataset_smooth.py`: Dataset using dense smoothed depth (filled using Telea, optionally bilateral-filtered)

---

## Recommended Order

1. Run `eval_agco_monodepth2.py` to get baseline metrics
2. Run `train_finetune_monodepth2_agco.py` to fine-tune
3. Run evaluation:
   - `eval_finetuned_monodepth2_agco.py`
   - `eval_finetuned_monodepth2_agco_roi.py`
4. Run `predicting_depth_maps.py` for visual results

---

## Reference

This pipeline was developed as part of a Master’s thesis at Aalborg University in collaboration with AGCO:

**“Monocular Depth Estimation with LiDAR Supervision:  
Automated Dataset Preparation and Model Evaluation for Agricultural Machinery”**
