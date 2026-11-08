"""Tiny tool-replacement MDP solved with tabular Q-learning. Run: python -m ipm.rl

State: tool-wear bin (10-minute bins). Each step the machine runs one cycle and wear advances one bin.
Actions: 0 = keep running, 1 = replace the tool now (cost c_replace, wear resets).
If the machine fails during a cycle it costs c_fail and wear resets.

The per-cycle failure probability at each wear level is estimated from the AI4I rows. These rows are
independent snapshots, not time series, so treating that rate as a per-cycle hazard is a modelling
assumption. Q-learning is checked against exact value iteration, since the MDP is small and known.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipm.data import TARGET, load_raw, train_val_test_split

ROOT = Path(__file__).resolve().parents[2]
BIN_MIN = 10
N_BINS = 26
GAMMA = 0.99


def hazard_by_wear(df: pd.DataFrame, n_bins: int = N_BINS, prior_strength: float = 20.0) -> np.ndarray:
    """Smoothed failure rate per wear bin (shrunk toward the overall rate)."""
    b = np.minimum(df["Tool wear [min]"].to_numpy() // BIN_MIN, n_bins - 1).astype(int)
    y = df[TARGET].to_numpy()
    overall = y.mean()
    n = np.bincount(b, minlength=n_bins)
    f = np.bincount(b, weights=y, minlength=n_bins)
    return (f + prior_strength * overall) / (n + prior_strength)


def value_iteration(h: np.ndarray, c_fail: float, c_replace: float, gamma: float = GAMMA, iters: int = 5000):
    n = len(h)
    V = np.zeros(n)
    for _ in range(iters):
        nxt = np.minimum(np.arange(n) + 1, n - 1)
        q_keep = h * (c_fail + gamma * V[0]) + (1 - h) * gamma * V[nxt]
        q_rep = np.full(n, c_replace + gamma * V[0])
        V_new = np.minimum(q_keep, q_rep)
        if np.max(np.abs(V_new - V)) < 1e-10:
            V = V_new
            break
        V = V_new
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    q_keep = h * (c_fail + gamma * V[0]) + (1 - h) * gamma * V[nxt]
    q_rep = np.full(n, c_replace + gamma * V[0])
    return (q_rep < q_keep).astype(int), V


def q_learning(
    h: np.ndarray, c_fail: float, c_replace: float, seed: int, steps: int = 400_000,
    gamma: float = GAMMA, eps: float = 0.1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(h)
    Q = np.zeros((n, 2))
    visits = np.zeros((n, 2))
    s = 0
    for t in range(steps):
        if t % 50 == 0:
            s = int(rng.integers(0, n))  # exploring starts so every wear level is visited
        a = int(rng.integers(0, 2)) if rng.random() < eps else int(np.argmin(Q[s]))
        if a == 1:
            cost, s2 = c_replace, 0
        elif rng.random() < h[s]:
            cost, s2 = c_fail, 0
        else:
            cost, s2 = 0.0, min(s + 1, n - 1)
        visits[s, a] += 1
        alpha = 1.0 / (1.0 + 0.05 * visits[s, a])
        Q[s, a] += alpha * (cost + gamma * Q[s2].min() - Q[s, a])
        s = s2
    return np.argmin(Q, axis=1)


def evaluate(policy: np.ndarray, h: np.ndarray, c_fail: float, c_replace: float,
             steps: int = 200_000, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    n = len(h)
    s, total, fails, reps = 0, 0.0, 0, 0
    for _ in range(steps):
        if policy[s] == 1:
            total += c_replace
            reps += 1
            s = 0
        elif rng.random() < h[s]:
            total += c_fail
            fails += 1
            s = 0
        else:
            s = min(s + 1, n - 1)
    return {"cost_per_cycle": total / steps, "failures_per_1000": 1000 * fails / steps,
            "replacements_per_1000": 1000 * reps / steps}


def fixed_policy(n: int, k: int) -> np.ndarray:
    """Replace when wear bin reaches k (k >= n means never replace)."""
    return (np.arange(n) >= k).astype(int)


def best_fixed(h: np.ndarray, c_fail: float, c_replace: float) -> tuple[int, np.ndarray]:
    n = len(h)
    best = min(range(1, n + 1), key=lambda k: evaluate(fixed_policy(n, k), h, c_fail, c_replace, 30_000)["cost_per_cycle"])
    return best, fixed_policy(n, best)


def main(seeds: tuple[int, ...] = (0, 1, 2, 3, 4), cost_pairs=((100, 10), (50, 10), (200, 10))) -> None:
    path = ROOT / "data" / "raw" / "ai4i2020.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    df = load_raw(path)
    train, val, test = train_val_test_split(df)
    h_train = hazard_by_wear(train)
    h_eval = hazard_by_wear(pd.concat([val, test]))
    res = {"bin_minutes": BIN_MIN, "gamma": GAMMA, "hazard_train": h_train.tolist(),
           "hazard_heldout": h_eval.tolist(), "cases": {}}
    for c_fail, c_rep in cost_pairs:
        k, p_fix = best_fixed(h_train, c_fail, c_rep)
        p_vi, _ = value_iteration(h_train, c_fail, c_rep)
        pols = {"run_to_failure": fixed_policy(len(h_train), len(h_train)),
                f"fixed_replace_at_bin_{k}": p_fix, "value_iteration_optimal": p_vi}
        for s in seeds:
            pols[f"q_learning_seed{s}"] = q_learning(h_train, c_fail, c_rep, seed=s)
        case = {n: {"train_env": evaluate(p, h_train, c_fail, c_rep),
                    "heldout_env": evaluate(p, h_eval, c_fail, c_rep), "policy": p.tolist()}
                for n, p in pols.items()}
        res["cases"][f"c_fail={c_fail},c_replace={c_rep}"] = case
        print(f"--- c_fail={c_fail}, c_replace={c_rep}  (cost per cycle: train env / held-out env)")
        for n, v in case.items():
            print(f"{n:28s} {v['train_env']['cost_per_cycle']:.3f} / {v['heldout_env']['cost_per_cycle']:.3f}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rl_maintenance.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
