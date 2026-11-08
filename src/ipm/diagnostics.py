"""Diagnostic plots for the best classical model. Run: python -m ipm.diagnostics

Uses the seed-42 split: precision-recall curve, calibration curve and permutation importance
(all on the test split; the model is fitted on train only).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve

from ipm.classical import build_model
from ipm.data import NUMERIC_FEATURES, SEED, load_raw, split_xy, train_val_test_split
from ipm.features import ENGINEERED, add_engineered

ROOT = Path(__file__).resolve().parents[2]


def compute(df: pd.DataFrame, seed: int = SEED, n_repeats: int = 10) -> dict:
    numeric = NUMERIC_FEATURES + ENGINEERED
    train, _, test = train_val_test_split(df, seed=seed)
    Xtr, ytr = split_xy(train)
    Xte, yte = split_xy(test)
    Xtr, Xte = add_engineered(Xtr), add_engineered(Xte)
    model = build_model("hist_gb", numeric, seed=seed).fit(Xtr, ytr)
    score = model.predict_proba(Xte)[:, 1]
    prec, rec, _ = precision_recall_curve(yte, score)
    frac_pos, mean_pred = calibration_curve(yte, score, n_bins=5, strategy="quantile")
    imp = permutation_importance(
        model, Xte, yte, scoring="average_precision", n_repeats=n_repeats, random_state=seed
    )
    order = np.argsort(imp.importances_mean)[::-1]
    return {
        "seed": seed,
        "pr_auc": float(average_precision_score(yte, score)),
        "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
        "calibration": {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()},
        "permutation_importance": [
            {"feature": Xte.columns[i], "mean": float(imp.importances_mean[i]), "std": float(imp.importances_std[i])}
            for i in order
        ],
    }


def plot(res: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    ax.plot(res["pr_curve"]["recall"], res["pr_curve"]["precision"])
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall (PR-AUC {res['pr_auc']:.3f})")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(res["calibration"]["mean_predicted"], res["calibration"]["fraction_positive"], marker="o", label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed failure fraction", title="Calibration (5 quantile bins)")
    ax.legend(fontsize=8)
    ax = axes[2]
    imp = res["permutation_importance"][::-1]
    ax.barh([d["feature"] for d in imp], [d["mean"] for d in imp], xerr=[d["std"] for d in imp])
    ax.set(xlabel="Drop in average precision when shuffled", title="Permutation importance")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    path = ROOT / "data" / "raw" / "ai4i2020.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    res = compute(load_raw(path))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "diagnostics.json").write_text(json.dumps(res, indent=2))
    plot(res, ROOT / "reports" / "figures" / "diagnostics.png")
    print(f"PR-AUC (seed {res['seed']}): {res['pr_auc']:.3f}")
    for d in res["permutation_importance"][:5]:
        print(f"  {d['feature']:28s} {d['mean']:.3f} +/- {d['std']:.3f}")


if __name__ == "__main__":
    main()
