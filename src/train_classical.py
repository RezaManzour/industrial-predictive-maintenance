"""Train and score the Phase 2 classical models.

Run from the repo root: ``python -m src.train_classical`` (expects
``data/raw/ai4i2020.csv``). Thresholds are chosen on the validation split;
the test split is scored once per model, for the final table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.splits import TARGET, stratified_split
from src.features.engineering import add_features
from src.models.classical import (
    FEATURES,
    best_f1_threshold,
    build_models,
    scale_pos_weight,
)
from src.utils.metrics import f1_at_threshold, pr_auc, recall_at_precision, roc_auc

DATA = Path("data/raw/ai4i2020.csv")
OUT = Path("results/metrics_classical.json")
PRED_OUT = Path("results/predictions/classical_predictions.csv")
MIN_PRECISION = 0.8


def main(data_path: Path = DATA, out_path: Path = OUT) -> None:
    df = add_features(pd.read_csv(data_path))
    train, val, test = stratified_split(df)
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]

    rows = []
    pred_frames = []
    for name, model in build_models(scale_pos_weight(y_tr)).items():
        model.fit(X_tr, y_tr)
        val_score = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = best_f1_threshold(y_val, val_score)
        print(
            f"{name:20s} val PR-AUC={pr_auc(y_val, val_score):.4f} val F1={val_f1:.4f}"
        )

        test_score = model.predict_proba(X_te)[:, 1]  # the only test access
        pred_frames += [
            pd.DataFrame(
                {
                    "model": name,
                    "split": "val",
                    "UDI": val["UDI"].to_numpy(),
                    "y_true": y_val.to_numpy(),
                    "score": val_score,
                }
            ),
            pd.DataFrame(
                {
                    "model": name,
                    "split": "test",
                    "UDI": test["UDI"].to_numpy(),
                    "y_true": y_te.to_numpy(),
                    "score": test_score,
                }
            ),
        ]
        rows.append(
            {
                "model": name,
                "pr_auc": pr_auc(y_te, test_score),
                "roc_auc": roc_auc(y_te, test_score),
                "f1": f1_at_threshold(y_te, test_score, threshold),
                "threshold": threshold,
                f"recall_at_precision_{MIN_PRECISION}": recall_at_precision(
                    y_te, test_score, MIN_PRECISION
                ),
            }
        )

    result = {
        "phase": 2,
        "split": "stratified 70/15/15, seed=42",
        "status": "complete",
        "models": rows,
    }
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_frames).to_csv(PRED_OUT, index=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path} and {PRED_OUT}")


if __name__ == "__main__":
    main()
