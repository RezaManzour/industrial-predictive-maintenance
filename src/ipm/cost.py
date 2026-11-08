"""Cost-aware decision threshold. Run: python -m ipm.cost

A missed failure costs c_fn and an unnecessary maintenance action costs c_fp. The costs are assumptions,
so results are reported for several cost ratios. Thresholds are chosen on validation data only.
Compared per machine: act on nobody, act on everybody, F1-optimal threshold, cost-optimal threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipm.classical import build_model
from ipm.data import NUMERIC_FEATURES, load_raw, split_xy, train_val_test_split
from ipm.features import ENGINEERED, add_engineered
from ipm.metrics import best_f1_threshold, confusion_cost, cost_optimal_threshold

ROOT = Path(__file__).resolve().parents[2]
RATIOS = (5, 10, 20)  # c_fn / c_fp, with c_fp = 1


def run_seed(df: pd.DataFrame, seed: int, ratios=RATIOS, model_name: str = "hist_gb") -> dict:
    numeric = NUMERIC_FEATURES + ENGINEERED
    parts = []
    for part in train_val_test_split(df, seed=seed):
        X, y = split_xy(part)
        parts.append((add_engineered(X), y.to_numpy()))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts
    model = build_model(model_name, numeric, seed=seed).fit(Xtr, ytr)
    sva = model.predict_proba(Xva)[:, 1]
    ste = model.predict_proba(Xte)[:, 1]
    f1_thr = best_f1_threshold(yva, sva)
    out = {}
    for r in ratios:
        thr = cost_optimal_threshold(yva, sva, c_fn=r, c_fp=1.0)
        out[str(r)] = {
            "act_on_nobody": confusion_cost(yte, ste, 2.0, r, 1.0),
            "act_on_everybody": confusion_cost(yte, ste, -1.0, r, 1.0),
            "f1_threshold": confusion_cost(yte, ste, f1_thr, r, 1.0),
            "cost_threshold": confusion_cost(yte, ste, thr, r, 1.0),
        }
    return out


def summarize(per_seed: list[dict]) -> dict:
    out = {}
    for r in per_seed[0]:
        out[r] = {
            k: {
                "mean": float(np.mean([s[r][k] for s in per_seed])),
                "std": float(np.std([s[r][k] for s in per_seed], ddof=1)),
            }
            for k in per_seed[0][r]
        }
    return out


def main(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    path = ROOT / "data" / "raw" / "ai4i2020.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    df = load_raw(path)
    per_seed = [run_seed(df, s) for s in seeds]
    res = {"seeds": list(seeds), "model": "hist_gb+eng", "cost_per_machine": summarize(per_seed)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "cost_thresholds.json").write_text(json.dumps(res, indent=2))
    for r, v in res["cost_per_machine"].items():
        row = "  ".join(f"{k}={m['mean']:.3f}+/-{m['std']:.3f}" for k, m in v.items())
        print(f"c_fn/c_fp={r:>2s}: {row}")


if __name__ == "__main__":
    main()
