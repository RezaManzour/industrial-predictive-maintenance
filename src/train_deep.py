"""Train and score the Phase 3 deep models (PyTorch MLP, Keras mirror).

Run from the repo root: ``python -m src.train_deep`` (expects
``data/raw/ai4i2020.csv``). Same split and threshold rule as Phase 2: the
threshold is chosen on val, and the test split is scored once per model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import TARGET, stratified_split
from src.features.engineering import add_features
from src.models.classical import NUM_COLS, best_f1_threshold
from src.models.deep_utils import Standardizer, encode_type
from src.utils.metrics import f1_at_threshold, pr_auc, recall_at_precision, roc_auc

DATA = Path("data/raw/ai4i2020.csv")
OUT = Path("results/metrics_deep.json")
MIN_PRECISION = 0.8


def _run(name, framework, fit, predict, prep, out_rows):
    (X_tr, t_tr, y_tr), (X_val, t_val, y_val), (X_te, t_te, y_te) = prep
    model, val_score = fit(X_tr, t_tr, y_tr, X_val, t_val, y_val)
    threshold, _ = best_f1_threshold(y_val, val_score)
    test_score = predict(model, X_te, t_te)  # the only test access
    out_rows.append(
        {
            "model": name,
            "framework": framework,
            "pr_auc": pr_auc(y_te, test_score),
            "roc_auc": roc_auc(y_te, test_score),
            "f1": f1_at_threshold(y_te, test_score, threshold),
            "threshold": threshold,
            f"recall_at_precision_{MIN_PRECISION}": recall_at_precision(
                y_te, test_score, MIN_PRECISION
            ),
        }
    )
    print(f"{name:12s} val PR-AUC={pr_auc(y_val, val_score):.4f} test done")


def main(
    data_path: Path = DATA, out_path: Path = OUT, frameworks=("torch", "keras")
) -> None:
    df = add_features(pd.read_csv(data_path))
    train, val, test = stratified_split(df)
    scaler = Standardizer().fit(
        train[NUM_COLS].to_numpy(dtype=np.float64)
    )  # train only

    def prep(part):
        X = scaler.transform(part[NUM_COLS].to_numpy(dtype=np.float64))
        return X, encode_type(part["Type"]), part[TARGET].to_numpy()

    parts = (prep(train), prep(val), prep(test))
    rows: list[dict] = []
    if "torch" in frameworks:
        from src.models.torch_mlp import fit_torch_mlp, predict_torch_mlp

        _run("torch_mlp", "pytorch", fit_torch_mlp, predict_torch_mlp, parts, rows)
    if "keras" in frameworks:
        from src.models.keras_mlp import fit_keras_mlp, predict_keras_mlp

        _run("keras_mlp", "tensorflow", fit_keras_mlp, predict_keras_mlp, parts, rows)

    result = {
        "phase": 3,
        "split": "stratified 70/15/15, seed=42",
        "status": "complete",
        "models": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frameworks", default="torch,keras")
    args = parser.parse_args()
    main(frameworks=tuple(args.frameworks.split(",")))
