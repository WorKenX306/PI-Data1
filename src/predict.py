from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from .train import TrainResult, load_train_result
except Exception:
    from train import TrainResult, load_train_result


def _ensure_dataframe(X_raw: Any) -> pd.DataFrame:
    if isinstance(X_raw, pd.DataFrame):
        return X_raw.copy()
    if isinstance(X_raw, dict):
        return pd.DataFrame([X_raw])
    if isinstance(X_raw, list):
        return pd.DataFrame(X_raw)
    raise TypeError("X_raw must be a pandas DataFrame, dict, or list of dicts.")


def predict_with_components(
    model: Any,
    preprocessor: Any,
    label_encoder: Optional[Any],
    X_raw: Any,
) -> Dict[str, Any]:
    """
    Predict using a fitted model + preprocessor (+ optional label encoder).
    """
    X_df = _ensure_dataframe(X_raw)
    X_proc = preprocessor.transform(X_df)

    y_pred_encoded = model.predict(X_proc)
    y_pred_encoded = np.asarray(y_pred_encoded)

    if label_encoder is not None:
        y_pred_labels = label_encoder.inverse_transform(y_pred_encoded.astype(int))
    else:
        y_pred_labels = y_pred_encoded

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_proc)

    return {
        "pred_encoded": y_pred_encoded,
        "pred_labels": np.asarray(y_pred_labels),
        "proba": y_proba,
    }


def predict_with_result(train_result: TrainResult, X_raw: Any) -> Dict[str, Any]:
    """
    Predict using a TrainResult returned by train()/train_all_models().
    """
    return predict_with_components(
        model=train_result.estimator,
        preprocessor=train_result.preprocessor,
        label_encoder=train_result.label_encoder,
        X_raw=X_raw,
    )


def predict_from_artifact_dir(artifact_dir: str, X_raw: Any) -> Dict[str, Any]:
    """
    Load TrainResult from disk and predict in one step.
    """
    train_result = load_train_result(artifact_dir)
    return predict_with_result(train_result, X_raw)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict using a saved artifact (data/models/<model>/<dataset>/artifact.pkl)."
    )
    parser.add_argument(
        "--artifact",
        default="data/models/rf/eMBB",
        help="Directory containing artifact.pkl",
    )
    parser.add_argument(
        "--dataset",
        default="eMBB",
        help="Dataset name to take a sample row from when --input is not set",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional CSV file with feature columns (same as training). If omitted, uses head(1) from the dataset.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of rows to predict when using default dataset sample (default: 1)",
    )
    args = parser.parse_args()

    try:
        from .data_loader import load_all_datasets
        from .preprocessing import make_xy
    except Exception:
        from data_loader import load_all_datasets
        from preprocessing import make_xy

    if args.input:
        df = pd.read_csv(args.input)
        X_raw = df
    else:
        datasets = load_all_datasets()
        if args.dataset not in datasets:
            raise SystemExit(
                f"Dataset '{args.dataset}' not loaded. Available: {list(datasets.keys())}"
            )
        X_df, _ = make_xy(datasets[args.dataset])
        X_raw = X_df.head(max(1, args.rows))

    out = predict_from_artifact_dir(args.artifact, X_raw)
    print("pred_labels:", out["pred_labels"].tolist())
    print("pred_encoded:", out["pred_encoded"].tolist())
    if out["proba"] is not None:
        print("proba shape:", out["proba"].shape)


if __name__ == "__main__":
    main()
