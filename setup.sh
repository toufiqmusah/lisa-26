#!/usr/bin/env bash
set -e

echo "Installing LISA-26 dependencies..."

# PyTorch (Colab comes with it; this is a fallback)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true

# Core dependencies
pip install nibabel pandas scipy scikit-learn xgboost joblib pyyaml tqdm \
  huggingface_hub opencv-python-headless

# PyRadiomics from source (avoids numpy 2 conflicts)
pip install --no-cache-dir \
  "numpy>=1.26,<1.27" \
  "SimpleITK>=2.2,<2.3"

pip install --no-cache-dir \
  "cython>=0.29" \
  "pykwalify>=0.5" \
  "scipy>=1.5"

git clone https://github.com/AIM-Harvard/pyradiomics.git /tmp/pyradiomics
cd /tmp/pyradiomics
pip install --no-cache-dir -e .
cd -
rm -rf /tmp/pyradiomics

# nnUNet submodule
git submodule update --init --recursive
pip install -e nnUNet

# Dynamic network architectures (PrimusV3S)
pip install --no-cache-dir git+https://github.com/MIC-DKFZ/dynamic-network-architectures.git

echo "Done. Set LISA_DATA_ROOT=/path/to/DATASET/task1a before running."
