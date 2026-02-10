"""Evaluation metrics for multi-label ECG classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def macro_auc_roc(
    labels: np.ndarray,
    probs: np.ndarray,
) -> float:
    """Macro-averaged AUC-ROC over all classes.

    Args:
        labels: Binary ground-truth array of shape (N, C).
        probs:  Predicted probabilities of shape (N, C).

    Returns:
        Scalar macro AUC-ROC.
    """
    return float(roc_auc_score(labels, probs, average="macro"))


def per_class_auc_roc(
    labels: np.ndarray,
    probs: np.ndarray,
) -> list[float]:
    """AUC-ROC for each class individually.

    Returns:
        List of C floats.
    """
    n_classes = labels.shape[1]
    return [float(roc_auc_score(labels[:, c], probs[:, c])) for c in range(n_classes)]


def macro_auprc(
    labels: np.ndarray,
    probs: np.ndarray,
) -> float:
    """Macro-averaged Area Under the Precision-Recall Curve.

    More informative than AUC-ROC under class imbalance.
    """
    return float(average_precision_score(labels, probs, average="macro"))


def per_class_auprc(
    labels: np.ndarray,
    probs: np.ndarray,
) -> list[float]:
    n_classes = labels.shape[1]
    return [float(average_precision_score(labels[:, c], probs[:, c])) for c in range(n_classes)]


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Optimal per-class threshold via Youden's J statistic (sensitivity + specificity − 1).

    Args:
        labels: (N, C) binary.
        probs:  (N, C) probabilities.

    Returns:
        (C,) array of optimal thresholds.
    """
    from sklearn.metrics import roc_curve

    n_classes = labels.shape[1]
    thresholds = np.zeros(n_classes)
    for c in range(n_classes):
        fpr, tpr, thresh = roc_curve(labels[:, c], probs[:, c])
        j = tpr - fpr
        thresholds[c] = thresh[np.argmax(j)]
    return thresholds


def compute_all_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute and return all evaluation metrics as a dict."""
    per_auc = per_class_auc_roc(labels, probs)
    per_ap = per_class_auprc(labels, probs)
    return {
        "macro_auc": macro_auc_roc(labels, probs),
        "macro_auprc": macro_auprc(labels, probs),
        "per_class_auc": dict(zip(class_names, per_auc)),
        "per_class_auprc": dict(zip(class_names, per_ap)),
    }
