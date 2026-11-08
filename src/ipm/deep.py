"""PyTorch MLP baseline (CPU). Run: python -m ipm.deep   (needs: pip install -e ".[deep]")

Same protocol as the classical models: fit on train, early stopping and threshold from validation,
test evaluated once. Class imbalance is handled with a positive-class weight in the loss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ipm.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_raw,
    split_xy,
    train_val_test_split,
)
from ipm.features import ENGINEERED, add_engineered
from ipm.metrics import best_f1_threshold, imbalance_metrics

ROOT = Path(__file__).resolve().parents[2]


def _torch():
    import torch

    return torch


def make_mlp(n_in: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.1):
    nn = _torch().nn
    layers: list = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def fit_predict(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, Xte: np.ndarray,
    seed: int, max_epochs: int = 200, patience: int = 20, batch_size: int = 256, lr: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train with early stopping on validation PR-AUC. Returns (val_scores, test_scores, best_epoch)."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_mlp(Xtr.shape[1])
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    Xv, Xe = torch.tensor(Xva), torch.tensor(Xte)

    def score(X):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(X).squeeze(1)).numpy()

    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(Xt))
        for i in range(0, len(order), batch_size):
            idx = torch.tensor(order[i : i + batch_size])
            opt.zero_grad()
            loss_fn(model(Xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
        ap = average_precision_score(yva, score(Xv))
        if ap > best:
            best, best_epoch, bad = ap, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return score(Xv), score(Xe), best_epoch


def run_seed(df: pd.DataFrame, seed: int, use_eng: bool) -> dict:
    numeric = NUMERIC_FEATURES + (ENGINEERED if use_eng else [])
    pre = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)]
    )
    parts = []
    for part in train_val_test_split(df, seed=seed):
        X, y = split_xy(part)
        parts.append((add_engineered(X) if use_eng else X, y.to_numpy()))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts
    A = pre.fit_transform(Xtr).astype("float32")
    B = pre.transform(Xva).astype("float32")
    C = pre.transform(Xte).astype("float32")
    sv, st, epoch = fit_predict(A, ytr, B, yva, C, seed)
    thr = best_f1_threshold(yva, sv)
    return {**imbalance_metrics(yte, st, thr), "best_epoch": epoch}


def summarize(runs: dict[str, list[dict]], reference: list[dict] | None = None) -> dict:
    out: dict = {"models": {}}
    for name, rs in runs.items():
        out["models"][name] = {
            m: {"mean": float(np.mean([r[m] for r in rs])), "std": float(np.std([r[m] for r in rs], ddof=1))}
            for m in ("pr_auc", "f1")
        }
    if reference is not None and "mlp+eng" in runs:
        d = np.array([a["pr_auc"] - b["hist_gb+eng"]["pr_auc"] for a, b in zip(runs["mlp+eng"], reference, strict=True)])
        out["paired_mlp_eng_minus_hist_gb_eng"] = {
            "mean_pr_auc_diff": float(d.mean()), "std": float(d.std(ddof=1)),
            "seeds_where_mlp_better": int((d > 0).sum()),
        }
    return out


def main(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    path = ROOT / "data" / "raw" / "ai4i2020.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    torch = _torch()
    torch.set_num_threads(4)
    df = load_raw(path)
    runs = {"mlp": [run_seed(df, s, False) for s in seeds], "mlp+eng": [run_seed(df, s, True) for s in seeds]}
    ref_path = ROOT / "results" / "metrics_multiseed.json"
    ref = None
    if ref_path.exists():
        saved = json.loads(ref_path.read_text())
        if saved["seeds"] == list(seeds):
            ref = saved["per_seed"]
    res = {"seeds": list(seeds), "per_seed": runs, **summarize(runs, ref)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics_deep.json").write_text(json.dumps(res, indent=2))
    for k, v in res["models"].items():
        print(f"{k:10s} PR-AUC {v['pr_auc']['mean']:.3f} +/- {v['pr_auc']['std']:.3f}   F1 {v['f1']['mean']:.3f} +/- {v['f1']['std']:.3f}")
    if "paired_mlp_eng_minus_hist_gb_eng" in res:
        v = res["paired_mlp_eng_minus_hist_gb_eng"]
        print(f"mlp+eng - hist_gb+eng  dPR-AUC {v['mean_pr_auc_diff']:+.3f} +/- {v['std']:.3f}  mlp better in {v['seeds_where_mlp_better']}/{len(seeds)}")


if __name__ == "__main__":
    main()
