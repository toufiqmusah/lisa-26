# Notes: Radiomics QC Improvements

## Problem
Baseline radiomics submission scored F1_micro=0.798 but F1_macro=0.463 — the model is good on class 0 (no artifact) but poor on minority classes 1 and 2.

## Changes Made

### 1. 80/20 train/val split (was 85/15)
`config.py: TEST_SIZE = 0.2` — more validation data for reliable metric estimation.

### 2. Hyperparameter search per label
`config.py: XGB_SEARCH_N_ITER = 30` — `RandomizedSearchCV` per label per view (21 searches), 3-fold CV, searching:
- `n_estimators`: 200, 500, 1000
- `max_depth`: 3, 6, 10, 15
- `learning_rate`: 0.01, 0.03, 0.05, 0.1
- `subsample`, `colsample_bytree`, `gamma`, `reg_lambda`, `reg_alpha`, `min_child_weight`

Each of the 21 label models gets its own optimal hyperparameters based on its class distribution.

### 3. Class-balanced sample weights
`config.py: USE_CLASS_WEIGHTS = True` — uses `sklearn.utils.class_weight.compute_class_weight('balanced')` so minority classes (1, 2) contribute more to the loss. This directly addresses the macro-F1 gap.

### 4. Probability calibration
`config.py: USE_CALIBRATION = True` — wraps each model with `CalibratedClassifierCV(cv=3, method='sigmoid')` on a held-out calibration split (`CAL_SPLIT=0.15`). Better probability estimates → better argmax decisions for minority classes.

### 5. Robust class handling
- Labels with only 1 class: predict that class (skip training)
- Labels with 2 classes: use `binary:logistic`, remap to 3 output slots at predict time
- Labels with 3 classes: use `multi:softprob` as before
- `label_meta.json` tracks which classes were present per label per view

### 6. Stratified splits
`_stratified_view_split()` uses `StratifiedShuffleSplit` on the multi-label sum — ensures calibration and validation splits preserve class distribution.

### 7. Calibration hold-out
15% of each view's data is held out from hyperparameter search and used exclusively for calibrating probabilities. This prevents overconfident probability estimates.

## To retrain
```bash
# Remove old models first
rm -rf checkpoints/radiomics/xgb_*.pkl checkpoints/radiomics/selector_*.pkl

# Train with all improvements
python3 train_radiomics.py

# Predict
python3 predict.py radiomics
```

## Expected impact
The class weighting + per-label hyperparameter tuning should improve macro-F1 by lifting recall on classes 1 and 2. Calibration should make probability-based decisions more reliable. Target: F1_macro from 0.463 → 0.55+ and F1_micro from 0.798 → 0.84+.
