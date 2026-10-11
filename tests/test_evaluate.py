import numpy as np

from src.evaluate import bootstrap_ci, paired_diff_ci
from src.utils.metrics import pr_auc


def _data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    score = y * 0.6 + rng.random(n) * 0.4
    return y, score


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    y, s = _data()
    r = bootstrap_ci(y, s, pr_auc, n=200)
    lo, hi = r["ci95"]
    assert lo <= r["point"] <= hi


def test_paired_diff_is_zero_for_identical_scores():
    y, s = _data()
    r = paired_diff_ci(y, s, s, pr_auc, n=50)
    assert r["point"] == 0.0
    assert r["ci95"] == [0.0, 0.0]
