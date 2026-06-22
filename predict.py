import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from config import QC_LABELS, CHECKPOINT_DIR, DATA_ROOT, OUTPUT_DIR
from data.task1a import RadiomicsDataset
from models.radiomics_model import RadiomicsXGBoost
from ensemble import EnsembleQC


def predict_radiomics(args):
    model = RadiomicsXGBoost()
    model.load(CHECKPOINT_DIR / "radiomics")
    print("Loaded radiomics models")

    dataset = RadiomicsDataset(root=DATA_ROOT, split="val")
    file_paths = [dataset[i][0] for i in range(len(dataset))]
    print(f"Predicting on {len(file_paths)} validation images")

    preds = model.predict(file_paths)

    rows = []
    for path, pred in zip(file_paths, preds):
        rows.append({"filename": path.name, **{lbl: int(p) for lbl, p in zip(QC_LABELS, pred)}})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "LISA_LF_QC_predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    for c in QC_LABELS:
        vc = df[c].value_counts().sort_index()
        print(f"  {c:15s} {dict(vc)}")

    print(f"Saved to {out_path}")


def predict_ensemble(args):
    ensemble = EnsembleQC.from_checkpoints()
    print(f"Loaded ensemble with weights: {ensemble.weights}")

    dataset = RadiomicsDataset(root=DATA_ROOT, split="val")
    file_paths = [dataset[i][0] for i in range(len(dataset))]
    print(f"Predicting on {len(file_paths)} validation images")

    preds = ensemble.predict(file_paths)

    rows = []
    for path, pred in zip(file_paths, preds):
        rows.append({"filename": path.name, **{lbl: int(p) for lbl, p in zip(QC_LABELS, pred)}})

    out_path = OUTPUT_DIR / "LISA_LF_QC_predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["radiomics", "ensemble"])
    args = parser.parse_args()

    if args.method == "radiomics":
        predict_radiomics(args)
    else:
        predict_ensemble(args)
