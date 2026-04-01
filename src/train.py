from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from .config import MODEL_PARAMS
    from .data_loader import load_all_datasets
    from .features import FEATURE_MAP
    from .preprocessing import build_preprocessor, make_xy
except Exception:
    from config import MODEL_PARAMS
    from data_loader import load_all_datasets
    from features import FEATURE_MAP
    from preprocessing import build_preprocessor, make_xy

try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


@dataclass
class TrainResult:
    dataset: str
    model_key: str
    model: str
    accuracy: float
    f1_macro: float
    roc_auc: float
    labels: List[str]
    estimator: Any
    preprocessor: Any
    label_encoder: LabelEncoder
    artifact_dir: Optional[str] = None


AVAILABLE_MODELS = ["rf", "xgb", "lr", "et", "mlp", "lgbm"]


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def save_train_result(train_result: TrainResult, base_dir: str = "data/models") -> str:
    """
    Save a TrainResult to disk and return artifact directory path.
    """
    dataset_safe = _sanitize_name(train_result.dataset)
    model_safe = _sanitize_name(train_result.model_key)
    artifact_dir = os.path.join(base_dir, model_safe, dataset_safe)
    os.makedirs(artifact_dir, exist_ok=True)

    payload = {
        "dataset": train_result.dataset,
        "model_key": train_result.model_key,
        "model_name": train_result.model,
        "metrics": {
            "accuracy": train_result.accuracy,
            "f1_macro": train_result.f1_macro,
            "roc_auc": train_result.roc_auc,
        },
        "labels": train_result.labels,
        "estimator": train_result.estimator,
        "preprocessor": train_result.preprocessor,
        "label_encoder": train_result.label_encoder,
    }

    with open(os.path.join(artifact_dir, "artifact.pkl"), "wb") as f:
        pickle.dump(payload, f)

    train_result.artifact_dir = artifact_dir
    return artifact_dir


def load_train_result(artifact_dir: str) -> TrainResult:
    """
    Load a TrainResult from artifact directory.
    """
    artifact_path = os.path.join(artifact_dir, "artifact.pkl")
    with open(artifact_path, "rb") as f:
        payload = pickle.load(f)

    return TrainResult(
        dataset=payload["dataset"],
        model_key=payload["model_key"],
        model=payload["model_name"],
        accuracy=float(payload["metrics"]["accuracy"]),
        f1_macro=float(payload["metrics"]["f1_macro"]),
        roc_auc=float(payload["metrics"]["roc_auc"]),
        labels=[str(x) for x in payload.get("labels", [])],
        estimator=payload["estimator"],
        preprocessor=payload["preprocessor"],
        label_encoder=payload["label_encoder"],
        artifact_dir=artifact_dir,
    )


def _build_model(model_name: str, params: Dict[str, Any], class_ratio: float) -> Tuple[Any, str]:
    model_name = model_name.lower()

    if model_name == "rf":
        return RandomForestClassifier(**params), "RandomForest"

    if model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )
        xgb_params = dict(params)
        xgb_params.setdefault("scale_pos_weight", class_ratio)
        return XGBClassifier(**xgb_params), "XGBoost"

    if model_name == "lr":
        return LogisticRegression(**params), "LogisticRegression"

    if model_name == "et":
        return ExtraTreesClassifier(**params), "ExtraTrees"

    if model_name == "mlp":
        return MLPClassifier(**params), "MLP"

    if model_name == "lgbm":
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed. Install it with: pip install lightgbm"
            )
        lgbm_params = dict(params)
        lgbm_params.setdefault("scale_pos_weight", class_ratio)
        return LGBMClassifier(**lgbm_params), "LightGBM"

    raise ValueError(
        f"Unknown model '{model_name}'. Use one of: {', '.join(AVAILABLE_MODELS)}."
    )


def _safe_binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique_classes = np.unique(y_true)
    if len(unique_classes) != 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def train(
    model_name: str,
    dataset_name: Optional[str] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    use_smote: bool = True,
    smote_threshold: float = 2.0,
    smote_sampling_strategy: float = 0.5,
    custom_params: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
    artifacts_base_dir: str = "data/models",
    verbose: bool = True,
) -> List[TrainResult]:
    """
    Train one selected model on one or all datasets.

    Examples:
        train("xgb")
        train("rf", dataset_name="TON_IoT", custom_params={"n_estimators": 400})
    """
    model_key = model_name.lower()
    if model_key not in MODEL_PARAMS:
        raise ValueError(f"'{model_name}' not found in MODEL_PARAMS.")

    datasets = load_all_datasets()
    if not datasets:
        raise RuntimeError("No datasets loaded. Check paths in data_loader.py.")

    if dataset_name is not None:
        if dataset_name not in datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}"
            )
        datasets = {dataset_name: datasets[dataset_name]}

    results: List[TrainResult] = []

    for name, df in datasets.items():
        if name not in FEATURE_MAP:
            if verbose:
                print(f"[SKIP] No feature list configured for dataset: {name}")
            continue

        X, y = make_xy(df)
        keep_cols = [col for col in FEATURE_MAP[name] if col in X.columns]
        if not keep_cols:
            if verbose:
                print(f"[SKIP] No matching features for dataset: {name}")
            continue

        X = X[keep_cols]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )

        preprocessor = build_preprocessor(X_train)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        counts = pd.Series(y_train).value_counts()
        class_ratio = float(counts.max() / counts.min()) if counts.min() > 0 else 1.0

        did_smote = False
        if (
            use_smote
            and HAS_SMOTE
            and name != "TON_IoT"
            and class_ratio > smote_threshold
            and len(np.unique(y_train)) == 2
        ):
            smote = SMOTE(
                sampling_strategy=smote_sampling_strategy,
                random_state=random_state,
            )
            X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
            class_ratio = 1.0
            did_smote = True

        params = dict(MODEL_PARAMS.get(model_key, {}))
        params["random_state"] = random_state
        if custom_params:
            params.update(custom_params)

        model, model_display_name = _build_model(model_key, params, class_ratio)
        model.fit(X_train_proc, y_train)

        y_pred = model.predict(X_test_proc)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_proc)[:, 1]
        else:
            y_score = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        auc = _safe_binary_roc_auc(y_test, y_score)

        if verbose:
            print("\n==============================")
            print(f"Dataset: {name}")
            print(f"Model  : {model_display_name}")
            print("==============================")
            print(f"Features used : {len(keep_cols)}")
            print(f"SMOTE applied : {did_smote} (available={HAS_SMOTE})")
            print(f"Class ratio   : {class_ratio:.4f}")
            print(f"Accuracy      : {acc:.4f}")
            print(f"F1 macro      : {f1:.4f}")
            print(f"ROC-AUC       : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC       : nan")

        results.append(
            TrainResult(
                dataset=name,
                model_key=model_key,
                model=model_display_name,
                accuracy=acc,
                f1_macro=f1,
                roc_auc=auc,
                labels=[str(c) for c in le.classes_],
                estimator=model,
                preprocessor=preprocessor,
                label_encoder=le,
            )
        )

        if save_artifacts:
            artifact_dir = save_train_result(results[-1], base_dir=artifacts_base_dir)
            if verbose:
                print(f"Artifacts saved: {artifact_dir}")

    return results


def train_all_models(
    dataset_name: Optional[str] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    use_smote: bool = True,
    smote_threshold: float = 2.0,
    smote_sampling_strategy: float = 0.5,
    model_params_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    save_artifacts: bool = True,
    artifacts_base_dir: str = "data/models",
    skip_failed_models: bool = True,
    verbose: bool = True,
) -> Dict[str, List[TrainResult]]:
    """
    Train all model families and return grouped results.

    Example:
        all_results = train_all_models(dataset_name="TON_IoT")
        rf_results = all_results["rf"]
    """
    all_results: Dict[str, List[TrainResult]] = {}
    model_params_overrides = model_params_overrides or {}

    for model_name in AVAILABLE_MODELS:
        if verbose:
            print("\n############################################")
            print(f"Training model family: {model_name}")
            print("############################################")

        try:
            all_results[model_name] = train(
                model_name=model_name,
                dataset_name=dataset_name,
                test_size=test_size,
                random_state=random_state,
                use_smote=use_smote,
                smote_threshold=smote_threshold,
                smote_sampling_strategy=smote_sampling_strategy,
                custom_params=model_params_overrides.get(model_name),
                save_artifacts=save_artifacts,
                artifacts_base_dir=artifacts_base_dir,
                verbose=verbose,
            )
        except Exception as exc:
            if not skip_failed_models:
                raise
            all_results[model_name] = []
            if verbose:
                print(f"[WARN] Skipping model '{model_name}': {exc}")

    return all_results


if __name__ == "__main__":
    # Quick local run example
    train("xgb")
