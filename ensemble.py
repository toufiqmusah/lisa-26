from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import QC_LABELS, ENSEMBLE_WEIGHTS, CHECKPOINT_DIR
from models.radiomics_model import RadiomicsXGBoost


class EnsembleQC:
    """Weighted ensemble of radiomics XGBoost and encoder-based models."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        radiomics_model: Optional[RadiomicsXGBoost] = None,
        encoder_model: Optional["EncoderQC"] = None,
    ):
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.radiomics = radiomics_model
        self.encoder = encoder_model

    @classmethod
    def from_checkpoints(cls, radiomics_path: Path = None) -> "EnsembleQC":
        model = cls()
        if radiomics_path is None:
            radiomics_path = CHECKPOINT_DIR / "radiomics"
        if radiomics_path.exists():
            rm = RadiomicsXGBoost()
            rm.load(radiomics_path)
            model.radiomics = rm
        return model

    def predict_proba(self, file_paths: list) -> np.ndarray:
        probs = np.zeros((len(file_paths), len(QC_LABELS), 3))
        total_w = 0.0

        if self.radiomics is not None:
            rp = self.radiomics.predict_proba(file_paths)
            probs += self.weights.get("radiomics", 0.0) * rp
            total_w += self.weights.get("radiomics", 0.0)

        if self.encoder is not None:
            import torch
            from data.task1a import QCImageDataset
            ds = QCImageDataset(split="val")  # placeholder
            eps = []
            for i in range(len(ds)):
                x, _ = ds[i]
                eps.append(self.encoder.predict_proba(x.unsqueeze(0)))
            ep = torch.cat(eps).cpu().numpy()
            probs += self.weights.get("encoder_qc", 0.0) * ep
            total_w += self.weights.get("encoder_qc", 0.0)

        if total_w > 0:
            probs /= total_w
        return probs

    def predict(self, file_paths: list) -> np.ndarray:
        return self.predict_proba(file_paths).argmax(axis=-1)
