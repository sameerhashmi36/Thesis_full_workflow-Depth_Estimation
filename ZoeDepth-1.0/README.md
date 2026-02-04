# ZoeDepth Evaluation and Finetuning

This README provides an overview of how ZoeDepth was used for both **zero-shot evaluation** and **supervised finetuning** on the AGCO dataset. All the experiments are based on the original ZoeDepth repo:  
 https://github.com/isl-org/ZoeDepth

The pretrained model used:  
 https://github.com/isl-org/ZoeDepth/releases

## Folder Structure

```
ZoeDepth/
├── testing_zeroshot_evaluation/
│   ├── eval_agco_zoedepth.py
│   └── predicting_depth_maps_zoe_zeroshot.py
├── finetuning_and_eval_on_agco_dataset/
│   ├── train_agco_zoe_strict.py
│   ├── eval_agco_zoe_strict.py
│   ├── eval_agco_zoe_strict_roi.py
│   ├── predicting_depth_maps_zoe.py
│   ├── agco_zoe_dataset.py
│   ├── agco_zoe_dataset_smooth.py
│   ├── agco_zoe_config.py
│   └── zoe_ckpt_utils.py
```

All custom files live inside the original repo but follow its conventions. The conda environment was created based on the ZoeDepth installation instructions.

##  Zero-Shot Evaluation Scripts

### 1. `eval_agco_zoedepth.py`

Used for running zero-shot evaluation on AGCO test data.

**Command:**
```bash
python testing_zeroshot_evaluation/eval_agco_zoedepth.py \
  --raw-root /path/to/dataset \
  --model ZoeD_K \
  --base-ckpt checkpoints/ZoeD_M12_K.pt
```

---

### 2. `predicting_depth_maps_zoe_zeroshot.py`

Used to generate and save depth predictions on AGCO RGB images using ZoeDepth zero-shot.

**Command:**
```bash
python testing_zeroshot_evaluation/predicting_depth_maps_zoe_zeroshot.py \
  --raw-root /path/to/dataset \
  --model ZoeD_K \
  --base-ckpt checkpoints/ZoeD_M12_K.pt \
  --out-dir ./output_depths_zoe_zeroshot
```

##  Finetuning and Evaluation Scripts

### 3. `train_agco_zoe_strict.py`

Finetunes ZoeDepth using AGCO sparse ground-truth depth (LiDAR). Also supports optional smoothness regularization.

**Command:**
```bash
python finetuning_and_eval_on_agco_dataset/train_agco_zoe_strict.py \
  --raw-root /path/to/dataset \
  --model ZoeD_K \
  --base-ckpt checkpoints/ZoeD_M12_K.pt \
  --train-fraction 0.2 --val-fraction 0.2 \
  --epochs 20 --batch-size 2
```

---

### 4. `eval_agco_zoe_strict.py`

Evaluates the finetuned model on the full image using Monodepth2-style metrics.

**Command:**
```bash
python finetuning_and_eval_on_agco_dataset/eval_agco_zoe_strict.py \
  --raw-root /path/to/dataset \
  --model ZoeD_K \
  --base-ckpt checkpoints/ZoeD_M12_K.pt \
  --finetuned-weights models_finetuned_on_agco/ZoeD_K_train20_val20/best.pth
```

---

### 5. `eval_agco_zoe_strict_roi.py`

Performs evaluation only on the **bottom fraction** (e.g., bottom 1/3) of the image—important for evaluating tractor-front scenes with denser GT.

**Command:**
```bash
python finetuning_and_eval_on_agco_dataset/eval_agco_zoe_strict_roi.py \
  --raw-root /path/to/dataset \
  --model ZoeD_K \
  --base-ckpt checkpoints/ZoeD_M12_K.pt \
  --finetuned-weights models_finetuned_on_agco/ZoeD_K_train20_val20/best.pth \
  --roi-frac 0.333 \
  --vis-dir roi_eval_vis --vis-prob 0.05
```

---

### 6. `predicting_depth_maps_zoe.py`

Used to generate and save depth prediction images using the **finetuned** ZoeDepth model.

**Command:**
```bash
python finetuning_and_eval_on_agco_dataset/predicting_depth_maps_zoe.py \
  --raw-root /path/to/dataset \
  --model ZoeD_K \
  --base-ckpt checkpoints/ZoeD_M12_K.pt \
  --finetuned-weights models_finetuned_on_agco/ZoeD_K_train20_val20/best.pth \
  --out-dir ./output_depths_zoe_finetuned
```

---

##  Note

All training/validation/test splits, depth thresholds, and resolution settings are controlled via:

- `agco_zoe_config.py`
- `agco_zoe_dataset.py` or `agco_zoe_dataset_smooth.py`
- `zoe_ckpt_utils.py` (for utility)
