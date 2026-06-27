# LISA-26

QC classification (Task 1a) & enhancement (Task 1b) for the LISA Low-Field MRI challenge.

## Structure

```
lisa-26/
├── config.py              # Paths, hyperparams, HF checkpoint config
├── data/task1a.py         # Dataset loaders, resize, radiomics
├── models/
│   ├── radiomics_model.py # FeatureSelector + per-view XGBoost (21 models)
│   └── encoder_qc.py      # PrimusV3S backbone + QCHead (needs GPU)
├── train_radiomics.py     # Extract features, train XGBoost
├── train_encoder_qc.py    # Train the deep QC head
├── predict.py             # Generate LISA_LF_QC_predictions.csv
├── predict_task1b.py      # ULFSynth enhancement + zip
├── ensemble.py            # Weighted ensemble (radiomics + encoder)
├── cache/radiomics/       # Pre-extracted feature caches
├── checkpoints/radiomics/ # Trained XGBoost models + selectors
├── nnUNet/                # Submodule for Task 2 segmentation
├── requirements.txt
└── setup.sh
```

## Task 1a — QC Classification

```bash
# 1. Setup
git clone --recurse-submodules https://github.com/toufiqmusah/lisa-26.git
cd lisa-26
bash setup.sh

# 2. Point to data
export LISA_DATA_ROOT=/path/to/DATASET/task1a

# 3. Generate submission CSV (uses pre-trained radiomics models + feature cache)
python3 predict.py radiomics
# → outputs/LISA_LF_QC_predictions.csv
```

To retrain from scratch:
```bash
python3 train_radiomics.py
python3 predict.py radiomics
```

Encoder QC (requires GPU):

Three backbones:
- `primus` (default): Frozen PrimusV3S EVA (300M) + trainable head. Needs Task 2 checkpoint.
- `conv3d`: Fully trainable 3D residual ConvNet (8.3M). No checkpoint needed.
- `recon_feat`: Frozen conv stage from PrimusV3S reconstruction encoder (72M frozen) + 300K trainable head. Uses the same checkpoint as primus.

```bash
# Primus backbone (frozen EVA + head)
python3 train_encoder_qc.py train --backbone primus --checkpoint /path/to/checkpoint_best.pth
python3 train_encoder_qc.py predict --backbone primus
python3 predict.py ensemble

# Conv3D backbone (fully trainable)
python3 train_encoder_qc.py train --backbone conv3d
python3 train_encoder_qc.py predict --backbone conv3d

# Recon feature backbone (lightweight, recommended)
python3 train_encoder_qc.py train --backbone recon_feat
python3 train_encoder_qc.py predict --backbone recon_feat
```

## Task 1b — Enhancement (already submitted)

```bash
python3 predict_task1b.py
# → outputs/LISA_enhanced_predictions.zip
```

## Data layout (not in repo)

```
DATASET/
├── task1a/
│   ├── train/images/*.nii.gz   # 532 images
│   ├── train/labels.csv
│   └── val/images/*.nii.gz     # 114 images
└── task1b/
    └── val/images/*.nii.gz
```
