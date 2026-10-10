"""Classical baselines and XGBoost. Every preprocessing step is fit on train only."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
CAT_COLS = ["Type"]
NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power_w",
    "strain",
]
FEATURES = CAT_COLS + NUM_COLS


def scale_pos_weight(y_train) -> float:
    """Negative/positive ratio, computed on the training labels only."""
    y = np.asarray(y_train)
    return float((y == 0).sum() / (y == 1).sum())


def best_f1_threshold(y_true, y_score) -> tuple[float, float]:
    """Threshold that maximises F1 on these labels, and that F1. Use on val only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]  # the last PR point has no threshold
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def _preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )


def build_models(pos_weight: float) -> dict[str, Pipeline]:
    """Unfitted pipelines, from the weakest baseline to XGBoost."""
    from xgboost import XGBClassifier

    clfs = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=SEED),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            random_state=SEED,
        ),
    }
    return {
        name: Pipeline([("prep", _preprocess()), ("clf", clf)])
        for name, clf in clfs.items()
    }
