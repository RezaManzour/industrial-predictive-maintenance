import numpy as np

from ipm.cost import run_seed
from ipm.metrics import confusion_cost, cost_optimal_threshold
from ipm.rl import evaluate, fixed_policy, q_learning, value_iteration
from tests.test_data import make_df


def test_cost_threshold_beats_trivial_policies_on_validation():
    y = np.array([0] * 90 + [1] * 10)
    s = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.4, 1, 10)])
    t = cost_optimal_threshold(y, s, c_fn=10, c_fp=1)
    c = confusion_cost(y, s, t, 10, 1)
    assert c <= confusion_cost(y, s, 2.0, 10, 1) and c <= confusion_cost(y, s, -1.0, 10, 1)


def test_cost_run_seed_structure():
    out = run_seed(make_df(n=1500, rate=0.05), 0, ratios=(5, 10))
    assert set(out) == {"5", "10"}
    assert all(v >= 0 for v in out["5"].values())


def make_hazard(n=10):
    return np.linspace(0.002, 0.25, n)


def test_value_iteration_is_no_worse_than_fixed_policies():
    h = make_hazard()
    p, _ = value_iteration(h, c_fail=100, c_replace=10)
    opt = evaluate(p, h, 100, 10, steps=60_000)["cost_per_cycle"]
    for k in (1, 3, 6, len(h)):
        assert opt <= evaluate(fixed_policy(len(h), k), h, 100, 10, steps=60_000)["cost_per_cycle"] + 1e-2


def test_q_learning_matches_value_iteration_cost():
    h = make_hazard()
    p_vi, _ = value_iteration(h, 100, 10)
    p_q = q_learning(h, 100, 10, seed=0, steps=150_000)
    c_vi = evaluate(p_vi, h, 100, 10, steps=60_000)["cost_per_cycle"]
    c_q = evaluate(p_q, h, 100, 10, steps=60_000)["cost_per_cycle"]
    assert c_q <= c_vi * 1.10
