"""Metrics suited to a heavily imbalanced failure-prediction task."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def imbalance_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def best_f1_threshold(y_true, y_score) -> float:
    """Threshold maximising F1. Call on validation data only, never on test data."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    candidates = np.unique(y_score)
    if len(candidates) > 400:
        candidates = np.quantile(y_score, np.linspace(0, 1, 400))
    best_t, best_f = 0.5, -1.0
    for t in candidates:
        f = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def bootstrap_ci(y_true, y_score, threshold: float, n_boot: int = 1000, seed: int = 42) -> dict:
    """Percentile 95% intervals for each metric by resampling the evaluation set."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:
            continue
        for k, v in imbalance_metrics(y_true[idx], y_score[idx], threshold).items():
            draws.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}
