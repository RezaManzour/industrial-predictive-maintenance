"""Classical baselines. Run: python -m ipm.classical

Protocol: fit on train, choose the F1-optimal threshold on validation, evaluate test once.
Each model is run with and without the engineered features so their value is measured.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ipm.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEED,
    load_raw,
    split_xy,
    train_val_test_split,
)
from ipm.features import ENGINEERED, add_engineered
from ipm.metrics import best_f1_threshold, bootstrap_ci, imbalance_metrics

ROOT = Path(__file__).resolve().parents[2]


def build_model(name: str, numeric: list[str]) -> Pipeline:
    cat = OneHotEncoder(handle_unknown="ignore")
    if name == "logreg":
        pre = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", cat, CATEGORICAL_FEATURES)])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    else:
        pre = ColumnTransformer([("num", "passthrough", numeric), ("cat", cat, CATEGORICAL_FEATURES)])
        if name == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
            )
        elif name == "hist_gb":
            clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
        else:
            raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])


def run(df: pd.DataFrame, n_boot: int = 1000) -> dict:
    train, val, test = train_val_test_split(df)
    out: dict = {"protocol": {"seed": SEED, "split": "70/15/15 stratified", "n_boot": n_boot}, "runs": {}}
    for use_eng in (False, True):
        numeric = NUMERIC_FEATURES + (ENGINEERED if use_eng else [])
        sets = []
        for part in (train, val, test):
            X, y = split_xy(part)
            sets.append((add_engineered(X) if use_eng else X, y))
        (Xtr, ytr), (Xva, yva), (Xte, yte) = sets
        for name in ("logreg", "random_forest", "hist_gb"):
            model = build_model(name, numeric).fit(Xtr, ytr)
            thr = best_f1_threshold(yva, model.predict_proba(Xva)[:, 1])
            score = model.predict_proba(Xte)[:, 1]
            key = f"{name}{'+eng' if use_eng else ''}"
            out["runs"][key] = {
                "threshold_from_val": thr,
                "test": imbalance_metrics(yte, score, thr),
                "test_ci95": bootstrap_ci(yte, score, thr, n_boot=n_boot),
            }
    return out


def main() -> None:
    path = ROOT / "data" / "raw" / "ai4i2020.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    res = run(load_raw(path))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics_classical.json").write_text(json.dumps(res, indent=2))
    for k, v in res["runs"].items():
        t = v["test"]
        print(f"{k:22s} PR-AUC={t['pr_auc']:.3f} P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f}")


if __name__ == "__main__":
    main()
