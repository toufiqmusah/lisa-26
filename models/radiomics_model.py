import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

from config import (
    QC_LABELS,
    N_QC_CLASSES,
    XGB_PARAMS,
    XGB_SEARCH_N_ITER,
    XGB_SEARCH_CV,
    XGB_SEARCH_PARAMS,
    USE_CLASS_WEIGHTS,
    USE_CALIBRATION,
    USE_ORDINAL,
    RANDOM_SEED,
)


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


def _stratified_view_split(view_y: np.ndarray, test_size: float, seed: int):
    """Stratified split for multi-label data using label sum as proxy."""
    strat = view_y.sum(axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(sss.split(view_y, strat))


def _build_xgb(ordinal: bool, n_classes: int, **params) -> xgb.XGBModel:
    base = dict(params)
    if ordinal:
        return xgb.XGBRegressor(
            **{k: v for k, v in base.items() if k not in ("eval_metric", "random_state")},
            random_state=base.get("random_state", RANDOM_SEED),
        )
    return xgb.XGBClassifier(
        **base,
        objective="multi:softprob",
        num_class=n_classes,
    )


def _select_objective(y: np.ndarray, ordinal: bool):
    """Return the correct objective and num_class for the given labels."""
    unique = np.unique(y)
    n = len(unique)
    if n < 2:
        return None, None, 0
    if ordinal:
        return "reg:squarederror", None, 0
    if n == 2:
        return "binary:logistic", None, 0
    return "multi:softprob", N_QC_CLASSES, n


class RadiomicsXGBoost:
    """
    Multi-label XGBoost on PyRadiomics features.

    Per-view feature selection + per-label hyperparameter search +
    class weighting + probability calibration.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or XGB_PARAMS
        self.models: Dict[str, Dict[str, xgb.XGBModel]] = {}
        self.calibrators: Dict[str, Dict[str, CalibratedClassifierCV]] = {}
        self.selectors: Dict[str, FeatureSelector] = {}
        self.selected_features: Dict[str, List[str]] = {}
        self.label_meta: Dict[str, Dict[str, dict]] = {}

    def _compute_sample_weight(self, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        if len(classes) < 2:
            return np.ones(len(y))
        w = compute_class_weight("balanced", classes=classes, y=y)
        return w[y]

    def _search_and_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label: str,
        view: str,
    ) -> xgb.XGBModel:
        unique = np.unique(y)
        n_unique = len(unique)

        if n_unique < 2:
            print(f"    [{view}/{label}] only 1 class ({unique[0]}), skipping")
            return None

        sample_weight = self._compute_sample_weight(y) if USE_CLASS_WEIGHTS else None
        n_per_class = np.bincount(y.astype(int), minlength=N_QC_CLASSES)

        if n_unique == 2:
            obj = "binary:logistic"
            num_c = 0
            y_map = {c: i for i, c in enumerate(sorted(unique))}
            y_fit = np.array([y_map[v] for v in y])
        else:
            obj = "multi:softprob"
            num_c = N_QC_CLASSES
            y_fit = y

        base = dict(self.params)
        base.pop("eval_metric", None)
        base["random_state"] = RANDOM_SEED

        clf = xgb.XGBClassifier(**base, objective=obj, num_class=num_c)

        min_class_count = n_per_class[n_per_class > 0].min()
        cv = min(XGB_SEARCH_CV, min_class_count)
        if cv < 2:
            print(f"    [{view}/{label}] too few samples per class ({min_class_count}), skipping search")
            fit_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
            clf.fit(X, y_fit, **fit_kw)
            return clf

        search = RandomizedSearchCV(
            clf,
            XGB_SEARCH_PARAMS,
            n_iter=XGB_SEARCH_N_ITER,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=3,
            random_state=RANDOM_SEED,
            verbose=0,
        )

        fit_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        search.fit(X, y_fit, **fit_kw)

        best = search.best_estimator_
        print(f"    [{view}/{label}] best: depth={best.max_depth} lr={best.learning_rate} "
              f"est={best.n_estimators} subsample={best.subsample} "
              f"cv_logloss={-search.best_score_:.4f}")

        if sample_weight is not None:
            best.fit(X, y_fit, sample_weight=sample_weight)
        else:
            best.fit(X, y_fit)

        return best

    def _calibrate(
        self,
        clf: xgb.XGBModel,
        X: np.ndarray,
        y: np.ndarray,
        label: str,
        view: str,
    ) -> xgb.XGBModel:
        if clf is None:
            return None
        unique = np.unique(y)
        if len(unique) < 2:
            print(f"    [{view}/{label}] only 1 class in cal set, skipping calibration")
            return clf

        cal = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid")
        cal.fit(X, y)
        print(f"    [{view}/{label}] calibrated")
        return cal

    def fit(
        self,
        view_paths_dict: Dict[str, List[Path]],
        view_matrices: Dict[str, np.ndarray],
        view_feature_names: Dict[str, List[str]],
        view_labels: Dict[str, np.ndarray],
        cal_idx: Optional[Dict[str, np.ndarray]] = None,
    ) -> "RadiomicsXGBoost":
        for view in sorted(view_matrices.keys()):
            X_raw = view_matrices[view]
            feat_names = view_feature_names[view]
            view_y = view_labels[view]
            n = len(view_paths_dict[view])
            print(f"\n  [{view}] {n} samples, {X_raw.shape[1]} raw features")

            selector = FeatureSelector(var_threshold=0.0, corr_threshold=0.95)
            selector.fit(X_raw, feat_names)
            self.selectors[view] = selector
            self.selected_features[view] = selector.selected_features_

            mask = [feat_names.index(n) for n in selector.selected_features_]
            X = X_raw[:, mask]
            print(f"  [{view}] {X.shape[1]} features after selection")

            self.models[view] = {}
            self.calibrators[view] = {}
            self.label_meta[view] = {}

            for j, label in enumerate(QC_LABELS):
                y_view = view_y[:, j]

                if cal_idx is not None and USE_CALIBRATION:
                    train_idx = np.setdiff1d(np.arange(len(y_view)), cal_idx[view])
                    X_train, y_train = X[train_idx], y_view[train_idx]
                    X_cal, y_cal = X[cal_idx[view]], y_view[cal_idx[view]]
                else:
                    X_train, y_train = X, y_view
                    X_cal, y_cal = X, y_view

                self.label_meta[view][label] = {
                    "classes_present": sorted(int(c) for c in np.unique(y_view)),
                    "n_classes": int(len(np.unique(y_view))),
                }

                clf = self._search_and_fit(X_train, y_train, label, view)

                if clf is not None and USE_CALIBRATION:
                    calibrated = self._calibrate(clf, X_cal, y_cal, label, view)
                    self.models[view][label] = calibrated
                    self.calibrators[view][label] = calibrated if calibrated is not None else clf
                else:
                    self.models[view][label] = clf
                    self.calibrators[view][label] = clf

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
                clf = self.calibrators[view][label]
                if clf is None:
                    probs[global_idx[view], j, 0] = 1.0
                    continue
                try:
                    p = clf.predict_proba(X)
                except Exception:
                    probs[global_idx[view], j, 0] = 1.0
                    continue
                n_pred = p.shape[1]
                meta = self.label_meta.get(view, {}).get(label, {})
                present = meta.get("classes_present", [0, 1, 2])
                if n_pred == 2:
                    full = np.zeros((p.shape[0], 3))
                    c0, c1 = present
                    full[:, c0] = p[:, 0]
                    full[:, c1] = p[:, 1]
                    probs[global_idx[view], j, :] = full
                elif n_pred == 3:
                    probs[global_idx[view], j, :] = p
                else:
                    probs[global_idx[view], j, 0] = 1.0
        return probs

    def _load_feature_cache(self) -> dict:
        from config import CACHE_DIR
        import joblib

        cache_dir = CACHE_DIR / "radiomics"
        combined = {}

        val_path = cache_dir / "val_features_cache.pkl"
        if val_path.exists():
            combined.update(joblib.load(val_path))

        train_path = cache_dir / "features_cache.pkl"
        if train_path.exists():
            train_cache = joblib.load(train_path)
            combined.update(train_cache["features"])

        return combined

    def predict_proba(self, file_paths: List[Path]) -> np.ndarray:
        feat_cache = self._load_feature_cache()

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

        if self.label_meta:
            with open(path / "label_meta.json", "w") as f:
                json.dump(self.label_meta, f, indent=2)

        with open(path / "selected_features.json", "w") as f:
            json.dump(self.selected_features, f, indent=2)

        print(f"Model artefacts saved to {path}")

    def load(self, path: Path):
        import joblib
        self.models = {}
        self.selectors = {}
        self.calibrators = {}

        for p in path.glob("xgb_*.pkl"):
            parts = p.stem.split("_")
            _, view, label = parts[0], parts[1], "_".join(parts[2:])
            model = joblib.load(p)
            self.models.setdefault(view, {})[label] = model
            self.calibrators.setdefault(view, {})[label] = model

        for p in path.glob("selector_*.pkl"):
            view = p.stem.split("_", 1)[1]
            self.selectors[view] = joblib.load(p)

        json_path = path / "selected_features.json"
        if json_path.exists():
            with open(json_path) as f:
                self.selected_features = json.load(f)

        meta_path = path / "label_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.label_meta = json.load(f)

        print(f"Loaded {sum(len(v) for v in self.models.values())} models from {path}")
