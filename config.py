import os
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "cache"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Data root: set LISA_DATA_ROOT env var, or place DATASET/ next to this repo
_env_data = os.environ.get("LISA_DATA_ROOT")
if _env_data:
    DATA_ROOT = Path(_env_data)
else:
    DATA_ROOT = PROJECT_DIR.parent / "DATASET" / "task1a"

QC_LABELS = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
N_QC_CLASSES = 3

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
CAL_SPLIT = 0.15       # held-out fraction for probability calibration

# --- Radiomics + XGBoost ---
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "random_state": RANDOM_SEED,
    "eval_metric": "mlogloss",
}

# Hyperparameter search
XGB_SEARCH_N_ITER = 15
XGB_SEARCH_CV = 3
XGB_SEARCH_PARAMS = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5],
    "reg_lambda": [1.0, 5.0, 10.0],
    "reg_alpha": [0, 0.1, 0.5],
    "min_child_weight": [1, 3, 5],
}

# Training refinements
USE_CLASS_WEIGHTS = True
USE_CALIBRATION = True
USE_ORDINAL = False

# --- Encoder QC checkpoint (downloaded from HuggingFace) ---
ENCODER_CHECKPOINT_URL = "toufiqmusah/LISA-26"
ENCODER_CHECKPOINT_FILENAME = "QC-Encoder/checkpoint_best.pth"
ENCODER_CHECKPOINT_DIR = CHECKPOINT_DIR / "encoder_qc"
ENCODER_CHECKPOINT_PATH = ENCODER_CHECKPOINT_DIR / "QC-Encoder" / "checkpoint_best.pth"


def ensure_encoder_checkpoint() -> Path:
    """Download the encoder QC checkpoint from HuggingFace if not present."""
    if ENCODER_CHECKPOINT_PATH.exists():
        return ENCODER_CHECKPOINT_PATH

    ENCODER_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading encoder checkpoint from {ENCODER_CHECKPOINT_URL}...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=ENCODER_CHECKPOINT_URL,
            filename=ENCODER_CHECKPOINT_FILENAME,
            repo_type="dataset",
            local_dir=ENCODER_CHECKPOINT_DIR,
        )
        return Path(path)
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        raise


# --- Deep learning (encoder + head) ---
ENCODER_ARCH_KWARGS = {
    "input_channels": 1,
    "n_stages": 6,
    "features_per_stage": (32, 64, 128, 256, 320, 320),
    "conv_op": "torch.nn.Conv3d",
    "kernel_sizes": 3,
    "strides": (1, 2, 2, 2, 2, 2),
    "n_blocks_per_stage": (1, 3, 4, 6, 6, 6),
    "conv_bias": True,
    "norm_op": "torch.nn.InstanceNorm3d",
    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    "nonlin": "torch.nn.LeakyReLU",
    "nonlin_kwargs": {"inplace": True},
    "stem_channels": None,
    "bottleneck_channels": None,
    "block": "dynamic_network_architectures.building_blocks.residual.BasicBlockD",
}
ENCODER_REQUIRED_IMPORTS = ("conv_op", "norm_op", "nonlin", "block")

QC_HEAD_HIDDEN = [256, 128]
QC_HEAD_DROPOUT = 0.3
QC_HEAD_LR = 1e-4
QC_HEAD_WEIGHT_DECAY = 1e-5
QC_HEAD_BATCH_SIZE = 2
QC_HEAD_EPOCHS = 25
QC_HEAD_PATIENCE = 10

# Finetune: unfreeze last N encoder blocks with separate LR
FINETUNE_N_BLOCKS = 4
FINETUNE_LR = 1e-6

CROP_SIZE = (112, 160, 128)

# --- Ensemble ---
ENSEMBLE_WEIGHTS = {"radiomics": 0.3, "encoder_qc": 0.7}
