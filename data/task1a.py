import random
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import zoom

from config import DATA_ROOT, QC_LABELS, CROP_SIZE


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1).astype(np.float32)


class RandomFlip3D(nn.Module):
    """Randomly flip along the last two (in-plane) dimensions."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = x.flip(-1)
        if random.random() < 0.5:
            x = x.flip(-2)
        return x


class RandomIntensityAug(nn.Module):
    """Random intensity shift and scale."""
    def __init__(self, shift=0.1, scale=0.1):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = x + random.uniform(-self.shift, self.shift)
        if random.random() < 0.5:
            x = x * random.uniform(1.0 - self.scale, 1.0 + self.scale)
        return x


def get_train_transform():
    return torch.nn.Sequential(
        RandomFlip3D(),
        RandomIntensityAug(),
    )


class QCImageDataset(Dataset):
    """Loads NIfTI volumes and their QC labels for Task 1a.

    view: "axial", "coronal", "sagittal", or "all"
      - Specific view: loads only that orientation (one volume per sample)
      - "all": groups by subject, returns all orientations as a list (one subject per sample)
    """

    _ORI_MAP = {"axi": "axial", "cor": "coronal", "sag": "sagittal"}
    _ORI_SUFFIX = {"axial": "_axi", "coronal": "_cor", "sagittal": "_sag"}

    def __init__(
        self,
        root: Path = DATA_ROOT,
        split: str = "train",
        transform: Optional[callable] = None,
        resize: Optional[Tuple[int, ...]] = CROP_SIZE,
        view: str = "axial",
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.resize = resize
        self.view = view

        self.image_dir = self.root / split / "images"
        self.labels_path = self.root / split / "labels.csv"

        all_files = sorted(
            f for f in self.image_dir.glob("*.nii.gz")
            if ".mask." not in f.name
        )

        if self.labels_path.exists():
            import pandas as pd
            self.df = pd.read_csv(self.labels_path)
            self.df.index = self.df["filename"]
        else:
            self.df = None

        if view == "all" and split == "train":
            self.subjects, self.files = self._group_by_subject(all_files)
        elif view == "all":
            self.files = all_files
            self.subjects = None
        else:
            suffix = self._ORI_SUFFIX.get(view)
            if suffix is None:
                raise ValueError(f"Unknown view: {view!r}. Use 'axial','coronal','sagittal',or 'all'.")
            self.files = [f for f in all_files if suffix in f.name]
            self.subjects = None

    def _group_by_subject(self, files):
        groups = defaultdict(list)
        for f in files:
            m = re.match(r"(LISA_(?:VALIDATION_)?\d+)_LF_(axi|cor|sag)\.nii\.gz", f.name)
            if m:
                groups[m.group(1)].append(f)
        subjects = sorted(groups.keys())
        return subjects, [groups[s] for s in subjects]

    def __len__(self) -> int:
        return len(self.files)

    def _load_volume(self, path: Path) -> torch.Tensor:
        img = nib.load(str(path))
        volume = img.get_fdata().astype(np.float32)
        if self.resize:
            volume = resize_volume(volume, self.resize)
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        tensor = torch.from_numpy(volume).unsqueeze(0)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    def __getitem__(self, idx: int):
        if self.view == "all" and self.subjects is not None:
            orientations = self.files[idx]
            volumes = [self._load_volume(p) for p in orientations]
            labels = None
            if self.df is not None:
                row = self.df.loc[orientations[0].name]
                labels = row[QC_LABELS].values.astype(np.int64)
            return volumes, labels
        else:
            path = self.files[idx]
            tensor = self._load_volume(path)
            labels = None
            if self.df is not None:
                row = self.df.loc[path.name]
                labels = row[QC_LABELS].values.astype(np.int64)
            return tensor, labels


class RadiomicsDataset(Dataset):
    """Thin wrapper: yields (nifti_path, labels) for PyRadiomics extraction."""

    def __init__(self, root: Path = DATA_ROOT, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / "images"
        self.labels_path = self.root / split / "labels.csv"
        self.files = sorted(
            f for f in self.image_dir.glob("*.nii.gz")
            if ".mask." not in f.name
        )
        if self.labels_path.exists():
            import pandas as pd
            self.df = pd.read_csv(self.labels_path)
            self.df.index = self.df["filename"]
        else:
            self.df = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Path, Optional[np.ndarray]]:
        path = self.files[idx]
        labels = None
        if self.df is not None:
            row = self.df.loc[path.name]
            labels = row[QC_LABELS].values.astype(np.int64)
        return path, labels
