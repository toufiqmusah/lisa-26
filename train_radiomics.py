import argparse
from pathlib import Path

import numpy as np
import yaml

from config import CACHE_DIR, CHECKPOINT_DIR, QC_LABELS, RANDOM_SEED
from models.radiomics_model import RadiomicsXGBoost, parse_view


def load_cache(cache_dir: Path):
    import joblib

    cache = joblib.load(cache_dir / "features_cache.pkl")
    feature_cache = cache["features"]
    all_keys = cache["keys"]

    labels = np.load(cache_dir / "labels.npy")

    with open(cache_dir / "file_paths.txt") as f:
        file_paths = [Path(line.strip()) for line in f]

    return feature_cache, all_keys, labels, file_paths


def build_view_data(feature_cache, all_keys, labels, file_paths):
    views = set(parse_view(p.name) for p in file_paths)
    view_data = {}

    for view in sorted(views):
        idx = [i for i, p in enumerate(file_paths) if parse_view(p.name) == view]
        view_paths = [file_paths[i] for i in idx]
        view_y = labels[idx]

        matrix = np.array(
            [[feature_cache[str(p)][k] for k in all_keys] for p in view_paths],
            dtype=np.float32,
        )
        matrix = np.nan_to_num(matrix, nan=0.0)

        view_data[view] = {
            "paths": view_paths,
            "X": matrix,
            "feature_names": all_keys,
            "y": view_y,
        }
        print(f"  [{view}] {len(view_paths)} samples, {matrix.shape[1]} features")

    return view_data


def train(args):
    print("Loading feature cache...")
    feature_cache, all_keys, labels, file_paths = load_cache(Path(args.cache_dir))
    print(f"Loaded {len(feature_cache)} images, {len(all_keys)} features\n")

    print("Organising by view...")
    view_data = build_view_data(feature_cache, all_keys, labels, file_paths)

    print("\nTraining models (with hyperparameter search + class weights)...")
    model = RadiomicsXGBoost()
    model.fit(
        view_paths_dict={v: d["paths"] for v, d in view_data.items()},
        view_matrices={v: d["X"] for v, d in view_data.items()},
        view_feature_names={v: d["feature_names"] for v, d in view_data.items()},
        view_labels={v: d["y"] for v, d in view_data.items()},
        cal_idx=None,
    )

    save_dir = Path(args.out_dir)
    model.save(save_dir)

    selected_path = save_dir / "selected_features.yaml"
    with open(selected_path, "w") as f:
        yaml.dump(model.selected_features, f, default_flow_style=False)
    print(f"Selected features saved to {selected_path}")

    train_preds = model.predict(file_paths)
    acc = (train_preds == labels).mean()
    print(f"\nTrain accuracy: {acc:.4f}")

    per_label = []
    for j, lbl in enumerate(QC_LABELS):
        la = (train_preds[:, j] == labels[:, j]).mean()
        per_label.append(f"{lbl}={la:.4f}")
    print(f"Per-label: {', '.join(per_label)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(CACHE_DIR / "radiomics"))
    parser.add_argument("--out-dir", default=str(CHECKPOINT_DIR / "radiomics"))
    args = parser.parse_args()
    train(args)
