"""Bootstrap uncertainty for saved predictions. Never re-scores a model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.classical import best_f1_threshold
from src.utils.metrics import f1_at_threshold, pr_auc, recall_at_precision, roc_auc

PRED_PATH = Path("results/predictions/classical_predictions.csv")
OUT = Path("results/bootstrap_classical.json")
N_BOOT = 2000
SEED = 42
MIN_PRECISION = 0.8


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample rows with replacement inside each class, so class counts are kept."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])


def bootstrap_ci(y, score, metric, n=N_BOOT, seed=SEED, alpha=0.05):
    """Point estimate and percentile CI for metric(y, score)."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    draws = [
        metric(y[i], score[i]) for i in (stratified_indices(y, rng) for _ in range(n))
    ]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"point": float(metric(y, score)), "ci95": [float(lo), float(hi)]}


def paired_diff_ci(y, s1, s2, metric, n=N_BOOT, seed=SEED, alpha=0.05):
    """CI for metric(y, s1) - metric(y, s2), same resampled rows for both."""
    y, s1, s2 = np.asarray(y), np.asarray(s1), np.asarray(s2)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        i = stratified_indices(y, rng)
        draws.append(metric(y[i], s1[i]) - metric(y[i], s2[i]))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    point = metric(y, s1) - metric(y, s2)
    return {"point": float(point), "ci95": [float(lo), float(hi)]}


def main(pred_path: Path = PRED_PATH, out_path: Path = OUT) -> None:
    preds = pd.read_csv(pred_path)
    results: dict = {"n_boot": N_BOOT, "seed": SEED, "models": {}}
    test_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, grp in preds.groupby("model", sort=False):
        val = grp[grp["split"] == "val"]
        test = grp[grp["split"] == "test"]
        y = test["y_true"].to_numpy()
        s = test["score"].to_numpy()
        thr, _ = best_f1_threshold(val["y_true"], val["score"])  # validation only
        test_scores[name] = (y, s)
        results["models"][name] = {
            "threshold": thr,
            "pr_auc": bootstrap_ci(y, s, pr_auc),
            "roc_auc": bootstrap_ci(y, s, roc_auc),
            "f1": bootstrap_ci(y, s, lambda a, b: f1_at_threshold(a, b, thr)),
            f"recall_at_precision_{MIN_PRECISION}": bootstrap_ci(
                y, s, lambda a, b: recall_at_precision(a, b, MIN_PRECISION)
            ),
        }
    if {"random_forest", "xgboost"} <= test_scores.keys():
        y, s_rf = test_scores["random_forest"]
        _, s_xgb = test_scores["xgboost"]
        results["random_forest_minus_xgboost_pr_auc"] = paired_diff_ci(
            y, s_rf, s_xgb, pr_auc
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
