# LISA-26

QC classification (Task 1a) & enhancement (Task 1b) for the LISA Low-Field MRI challenge.

## Structure

```
lisa-26/
├── config.py              # Paths, hyperparams, HF checkpoint config
├── data/task1a.py         # Dataset loaders, resize, radiomics
├── models/
│   ├── dinov3_qc.py       # DINOv3 + LoRA + VisionBlocks + AttentionPool
│   ├── radiomics_model.py # FeatureSelector + per-view XGBoost (21 models)
│   └── encoder_qc.py      # PrimusV3S backbone + QCHead (needs GPU)
├── train_dinov3_qc.py     # Train & predict DINOv3 QC (multi-view, TTA)
├── train_radiomics.py     # Extract features, train XGBoost
├── train_encoder_qc.py    # Train the deep QC head
├── predict.py             # Generate LISA_LF_QC_predictions.csv
├── predict_task1b.py      # ULFSynth enhancement + zip
├── ensemble.py            # Weighted ensemble (radiomics + encoder)
├── cache/radiomics/       # Pre-extracted feature caches
├── checkpoints/
│   ├── dinov3_qc/         # DINOv3 checkpoints + predictions
│   └── radiomics/         # Trained XGBoost models + selectors
├── nnUNet/                # Submodule for Task 2 segmentation
├── requirements.txt
└── setup.sh
```

## Task 1a — QC Classification

### DINOv3 (current best: F1-micro 0.828)

Deep learning QC using a frozen DINOv3 backbone (ViT-B or ViT-S+) with LoRA adapters, VisionBlocks, and attention pooling over slices.

```bash
export LISA_DATA_ROOT=/path/to/DATASET/task1a

# Train (multi-view, ViT-S+, 250 epochs)
python3 train_dinov3_qc.py train \
  --model vits16plus \
  --view all \
  --num_epochs 250 \
  --hf_token hf_<YOUR_TOKEN>

# Predict (TTA enabled)
python3 train_dinov3_qc.py predict \
  --model vits16plus \
  --view all \
  --tta true \
  --hf_token hf_<YOUR_TOKEN>
# → checkpoints/dinov3_qc/LISA_LF_QC_predictions.csv
```

**Args:**
| Arg | Options | Default | Description |
|---|---|---|---|
| `--model` | `vits16plus`, `vitb16` | `vits16plus` | DINOv3 backbone variant |
| `--view` | `axial`, `coronal`, `sagittal`, `all` | `all` | View mode. `all` trains on all 3 orientations per subject |
| `--tta` | `true`, `false` | `true` | Test-time augmentation (4 flips) |
| `--num_epochs` | int | 250 | Training epochs |

### Radiomics (F1-micro 0.798)

Handcrafted whole-image features (PyRadiomics) + per-view XGBoost.

```bash
python3 train_radiomics.py
python3 predict.py radiomics
# → outputs/LISA_LF_QC_predictions.csv
```

### Encoder QC (needs GPU)

Three backbones: `primus` (frozen EVA 300M), `conv3d` (trainable 8.3M), `recon_feat` (frozen 72M + 300K head).

```bash
python3 train_encoder_qc.py train --backbone recon_feat
python3 train_encoder_qc.py predict --backbone recon_feat
python3 predict.py ensemble
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
│   ├── train/images/*.nii.gz   # 532 images (3 orientations × 176-182 each)
│   ├── train/labels.csv        # 532 rows × 7 artifact labels (0/1/2)
│   └── val/images/*.nii.gz     # 114 images (38 subjects × 3 orientations)
└── task1b/
    └── val/images/*.nii.gz
```
