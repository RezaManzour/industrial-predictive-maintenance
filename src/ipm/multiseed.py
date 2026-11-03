"""Repeat the classical protocol over several split seeds and compare paired runs.
Run: python -m ipm.multiseed

Each seed gives a different stratified split and different model randomness. Test sets from different
seeds overlap, so seeds are not independent samples; treat the spread as a stability check, not a
significance test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipm.classical import build_model
from ipm.data import NUMERIC_FEATURES, load_raw, split_xy, train_val_test_split
from ipm.features import ENGINEERED, add_engineered
from ipm.metrics import best_f1_threshold, imbalance_metrics

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("logreg", "random_forest", "hist_gb")


def run_seed(df: pd.DataFrame, seed: int) -> dict[str, dict[str, float]]:
    parts = train_val_test_split(df, seed=seed)
    res = {}
    for use_eng in (False, True):
        numeric = NUMERIC_FEATURES + (ENGINEERED if use_eng else [])
        sets = []
        for part in parts:
            X, y = split_xy(part)
            sets.append((add_engineered(X) if use_eng else X, y))
        (Xtr, ytr), (Xva, yva), (Xte, yte) = sets
        for name in MODELS:
            model = build_model(name, numeric, seed=seed).fit(Xtr, ytr)
            thr = best_f1_threshold(yva, model.predict_proba(Xva)[:, 1])
            score = model.predict_proba(Xte)[:, 1]
            res[f"{name}{'+eng' if use_eng else ''}"] = imbalance_metrics(yte, score, thr)
    return res


def summarize(per_seed: list[dict]) -> dict:
    keys = per_seed[0].keys()
    out: dict = {"n_seeds": len(per_seed), "models": {}, "paired_eng_minus_base": {}, "paired_rf_minus_hgb_eng": {}}
    for k in keys:
        vals = {m: np.array([r[k][m] for r in per_seed]) for m in ("pr_auc", "f1")}
        out["models"][k] = {m: {"mean": float(v.mean()), "std": float(v.std(ddof=1))} for m, v in vals.items()}
    for name in MODELS:
        d = np.array([r[f"{name}+eng"]["pr_auc"] - r[name]["pr_auc"] for r in per_seed])
        out["paired_eng_minus_base"][name] = {
            "mean_pr_auc_diff": float(d.mean()),
            "std": float(d.std(ddof=1)),
            "seeds_where_eng_better": int((d > 0).sum()),
        }
    d = np.array([r["random_forest+eng"]["pr_auc"] - r["hist_gb+eng"]["pr_auc"] for r in per_seed])
    out["paired_rf_minus_hgb_eng"] = {
        "mean_pr_auc_diff": float(d.mean()),
        "std": float(d.std(ddof=1)),
        "seeds_where_rf_better": int((d > 0).sum()),
    }
    return out


def main(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    path = ROOT / "data" / "raw" / "ai4i2020.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    df = load_raw(path)
    per_seed = [run_seed(df, s) for s in seeds]
    res = {"seeds": list(seeds), "per_seed": per_seed, **summarize(per_seed)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics_multiseed.json").write_text(json.dumps(res, indent=2))
    for k, v in res["models"].items():
        print(f"{k:20s} PR-AUC {v['pr_auc']['mean']:.3f} +/- {v['pr_auc']['std']:.3f}   F1 {v['f1']['mean']:.3f} +/- {v['f1']['std']:.3f}")
    for k, v in res["paired_eng_minus_base"].items():
        print(f"eng-base {k:14s} dPR-AUC {v['mean_pr_auc_diff']:+.3f} +/- {v['std']:.3f}  better in {v['seeds_where_eng_better']}/{len(seeds)}")
    v = res["paired_rf_minus_hgb_eng"]
    print(f"rf-hgb (eng)  dPR-AUC {v['mean_pr_auc_diff']:+.3f} +/- {v['std']:.3f}  rf better in {v['seeds_where_rf_better']}/{len(seeds)}")


if __name__ == "__main__":
    main()
