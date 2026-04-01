from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binary_roc_auc_or_nan(y_true: np.ndarray, y_score: Optional[np.ndarray]) -> float:
    if y_score is None:
        return float("nan")
    unique_classes = np.unique(y_true)
    if len(unique_classes) != 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _multiclass_roc_auc_ovr_or_nan(
    y_true: np.ndarray,
    y_score: Optional[np.ndarray],
    average: str,
) -> float:
    if y_score is None:
        return float("nan")
    if y_score.ndim != 2:
        return float("nan")
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 3:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average=average))
    except Exception:
        return float("nan")


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    y_score: Optional[Any] = None,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Evaluate predictions with common metrics.

    y_score should be:
    - positive-class probability vector for binary ROC-AUC, or
    - None (ROC-AUC returned as nan).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = None if y_score is None else np.asarray(y_score)
    report_dict = classification_report(y_true_arr, y_pred_arr, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "roc_auc_binary": _binary_roc_auc_or_nan(y_true_arr, y_score_arr),
        "roc_auc_multiclass_ovr": _multiclass_roc_auc_ovr_or_nan(y_true_arr, y_score_arr, average=average),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr),
        "classification_report": classification_report(y_true_arr, y_pred_arr, zero_division=0),
        "classification_report_dict": report_dict,
        "support": int(len(y_true_arr)),
    }
    return metrics
