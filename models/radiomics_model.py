import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold

from config import QC_LABELS, N_QC_CLASSES, XGB_PARAMS


VIEW_PATTERN = re.compile(r".*_([a-z]+)\.nii\.gz$")


def parse_view(filename: str) -> str:
    m = VIEW_PATTERN.search(filename)
    if m is None:
        raise ValueError(f"Cannot parse view from {filename}")
    return m.group(1)


def _make_mask(image_path: Path) -> Path:
    import nibabel as nib
    img = nib.load(str(image_path))
    data = img.get_fdata().astype(np.float64)
    threshold = data.mean() * 0.1
    mask_data = np.where(data > threshold, np.uint8(1), np.uint8(0))
    mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
    mask_path = image_path.with_suffix(".mask.nii.gz")
    nib.save(mask_img, str(mask_path))
    return mask_path


def extract_radiomics_features(image_path: Path) -> Dict[str, float]:
    from radiomics import featureextractor

    params = {
        "normalize": True,
        "normalizeScale": 100,
        "binWidth": 25,
        "label": 1,
        "resampledPixelSpacing": (1.5, 1.5, 1.5),
        "force2D": False,
        "force2Ddimension": 0,
        "additionalInfo": False,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    for cls in ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
        extractor.enableFeatureClassByName(cls)

    mask_path = _make_mask(image_path)
    result = extractor.execute(str(image_path), str(mask_path))
    mask_path.unlink(missing_ok=True)

    features = {}
    for key, val in result.items():
        if key.startswith("diagnostics"):
            continue
        if isinstance(val, np.ndarray) and val.ndim == 0:
            features[key] = float(val)
        elif isinstance(val, (int, float, np.integer, np.floating)):
            features[key] = float(val)
    return features


def build_radiomics_matrix(
    file_paths: List[Path],
    feature_cache: Optional[Dict[str, Dict[str, float]]] = None,
    feature_mask: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if feature_cache is None:
        feature_cache = {}

    all_features = []
    feature_keys = None

    for path in file_paths:
        key = str(path)
        if key not in feature_cache:
            feature_cache[key] = extract_radiomics_features(path)
        feats = feature_cache[key]
        if feature_keys is None:
            feature_keys = sorted(feats.keys())
        all_features.append([feats[k] for k in feature_keys])

    matrix = np.array(all_features, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0)

    if feature_mask is not None:
        mask_set = set(feature_mask)
        if feature_keys is None:
            return matrix, feature_mask
        kept = [i for i, k in enumerate(feature_keys) if k in mask_set]
        matrix = matrix[:, kept]
        feature_keys = [feature_keys[i] for i in kept]

    return matrix, feature_keys or []


class FeatureSelector:
    """Select features by dropping low-variance then high-correlation ones.

    Artefacts saved alongside models so inference uses the exact same features.
    """

    def __init__(self, var_threshold: float = 0.0, corr_threshold: float = 0.95):
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.selected_features_: List[str] = []

    def fit(self, X: np.ndarray, feature_names: List[str]) -> "FeatureSelector":
        vt = VarianceThreshold(threshold=self.var_threshold)
        vt.fit(X)
        keep_var = vt.get_support()
        names = [n for n, k in zip(feature_names, keep_var) if k]

        if keep_var.sum() < 2:
            self.selected_features_ = names
            return self

        X_reduced = X[:, keep_var]
        corr = np.corrcoef(X_reduced.T)
        keep_corr = np.ones(X_reduced.shape[1], dtype=bool)

        for i in range(X_reduced.shape[1]):
            if not keep_corr[i]:
                continue
            for j in range(i + 1, X_reduced.shape[1]):
                if not keep_corr[j]:
                    continue
                if abs(corr[i, j]) > self.corr_threshold:
                    keep_corr[j] = False

        self.selected_features_ = [n for n, k in zip(names, keep_corr) if k]
        print(f"  Feature selection: {len(feature_names)} -> {len(self.selected_features_)}")
        return self

    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        mask_set = set(self.selected_features_)
        kept = [i for i, k in enumerate(feature_names) if k in mask_set]
        return X[:, kept], [feature_names[i] for i in kept]


class RadiomicsXGBoost:
    """
    Multi-label XGBoost on PyRadiomics features.

    Trains one model per view (axi, cor, sag) with per-view feature selection.
    Saves all artefacts (models + feature selectors) for reproducible inference.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or XGB_PARAMS
        self.models: Dict[str, Dict[str, xgb.XGBClassifier]] = {}
        self.selectors: Dict[str, FeatureSelector] = {}
        self.selected_features: Dict[str, List[str]] = {}

    def fit(
        self,
        view_paths_dict: Dict[str, List[Path]],
        view_matrices: Dict[str, np.ndarray],
        view_feature_names: Dict[str, List[str]],
        view_labels: Dict[str, np.ndarray],
    ) -> "RadiomicsXGBoost":
        for view in sorted(view_matrices.keys()):
            X_raw = view_matrices[view]
            feat_names = view_feature_names[view]
            view_y = view_labels[view]
            n = len(view_paths_dict[view])
            print(f"  [{view}] {n} samples, {X_raw.shape[1]} raw features")

            selector = FeatureSelector(var_threshold=0.0, corr_threshold=0.95)
            selector.fit(X_raw, feat_names)
            self.selectors[view] = selector
            self.selected_features[view] = selector.selected_features_

            mask = [feat_names.index(n) for n in selector.selected_features_]
            X = X_raw[:, mask]
            print(f"  [{view}] {X.shape[1]} features after selection")

            self.models[view] = {}
            for j, label in enumerate(QC_LABELS):
                clf = xgb.XGBClassifier(
                    **self.params,
                    objective="multi:softprob",
                    num_class=N_QC_CLASSES,
                )
                clf.fit(X, view_y[:, j])
                self.models[view][label] = clf

        return self

    def predict_from_cache(
        self,
        view_matrices: Dict[str, np.ndarray],
        view_feature_names: Dict[str, List[str]],
        global_idx: Dict[str, np.ndarray],
        n_total: int,
    ) -> np.ndarray:
        probs = np.zeros((n_total, len(QC_LABELS), N_QC_CLASSES))
        for view in self.models:
            X_raw = view_matrices[view]
            feat_names = view_feature_names[view]
            selector = self.selectors[view]
            mask = [feat_names.index(n) for n in selector.selected_features_]
            X = X_raw[:, mask]
            for j, label in enumerate(QC_LABELS):
                probs[global_idx[view], j, :] = self.models[view][label].predict_proba(X)
        return probs

    def predict_proba(self, file_paths: List[Path]) -> np.ndarray:
        from config import CACHE_DIR

        cache_path = CACHE_DIR / "radiomics" / "val_features_cache.pkl"
        feat_cache = None
        if cache_path.exists():
            import joblib
            feat_cache = joblib.load(cache_path)

        view_paths = {v: [] for v in self.models}
        for p in file_paths:
            v = parse_view(p.name)
            view_paths[v].append(p)

        matrices = {}
        feature_names = {}
        global_idx = {}
        total = len(file_paths)

        idx_map = {p: i for i, p in enumerate(file_paths)}
        for view, paths in view_paths.items():
            if not paths:
                continue
            feat_dict = {}
            for p in paths:
                key = str(p)
                if feat_cache and key in feat_cache:
                    feat_dict[key] = feat_cache[key]
                else:
                    feat_dict[key] = extract_radiomics_features(p)
            mat, fnames = build_radiomics_matrix(paths, feature_cache=feat_dict)
            matrices[view] = mat
            feature_names[view] = fnames
            global_idx[view] = np.array([idx_map[p] for p in paths])

        return self.predict_from_cache(matrices, feature_names, global_idx, total)

    def predict(self, file_paths: List[Path]) -> np.ndarray:
        return self.predict_proba(file_paths).argmax(axis=-1)

    def save(self, path: Path):
        import joblib
        path.mkdir(parents=True, exist_ok=True)

        for view, view_models in self.models.items():
            for label, model in view_models.items():
                joblib.dump(model, path / f"xgb_{view}_{label}.pkl")

        for view, selector in self.selectors.items():
            joblib.dump(selector, path / f"selector_{view}.pkl")

        with open(path / "selected_features.json", "w") as f:
            json.dump(self.selected_features, f, indent=2)

        print(f"Model artefacts saved to {path}")

    def load(self, path: Path):
        import joblib
        self.models = {}
        self.selectors = {}

        for p in path.glob("xgb_*.pkl"):
            parts = p.stem.split("_")
            _, view, label = parts[0], parts[1], "_".join(parts[2:])
            self.models.setdefault(view, {})[label] = joblib.load(p)

        for p in path.glob("selector_*.pkl"):
            view = p.stem.split("_", 1)[1]
            self.selectors[view] = joblib.load(p)

        json_path = path / "selected_features.json"
        if json_path.exists():
            with open(json_path) as f:
                self.selected_features = json.load(f)

        print(f"Loaded {sum(len(v) for v in self.models.values())} models from {path}")
