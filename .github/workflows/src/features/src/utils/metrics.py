"""Imbalance-aware evaluation metrics (pure scikit-learn, model-agnostic)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve, as average precision.

    Primary metric: with ~3% positives, ROC-AUC looks optimistic while PR-AUC
    is sensitive to false alarms on the rare class.
    """
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def f1_at_threshold(y_true, y_score, threshold: float = 0.5) -> float:
    """F1 when predicting positive for ``y_score >= threshold``."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def recall_at_precision(y_true, y_score, min_precision: float = 0.8) -> float:
    """Best recall achievable while keeping precision >= ``min_precision``.

    Returns 0.0 if no threshold reaches that precision.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0
