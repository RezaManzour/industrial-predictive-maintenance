import pandas as pd

from ipm.classical import run
from ipm.features import ENGINEERED, add_engineered
from ipm.metrics import best_f1_threshold, bootstrap_ci
from tests.test_data import make_df


def test_engineered_features_values():
    X = pd.DataFrame(
        {
            "Rotational speed [rpm]": [60.0],
            "Torque [Nm]": [10.0],
            "Process temperature [K]": [310.0],
            "Air temperature [K]": [300.0],
            "Tool wear [min]": [5.0],
        }
    )
    out = add_engineered(X)
    assert set(ENGINEERED) <= set(out.columns)
    assert abs(out["power_w"][0] - 10 * 2 * 3.141592653589793) < 1e-9
    assert out["temp_diff_k"][0] == 10.0 and out["wear_x_torque"][0] == 50.0
    assert "power_w" not in X.columns  # input is not mutated


def test_threshold_and_bootstrap():
    y = [0, 0, 0, 1, 1, 0, 1, 0]
    s = [0.1, 0.2, 0.3, 0.9, 0.8, 0.4, 0.7, 0.2]
    t = best_f1_threshold(y, s)
    assert 0.4 < t <= 0.7
    ci = bootstrap_ci(y, s, t, n_boot=50)
    assert ci["f1"][0] <= ci["f1"][1]


def test_pipeline_runs_end_to_end_on_small_data():
    res = run(make_df(n=1500, rate=0.05), n_boot=20)
    assert set(res["runs"]) == {
        f"{m}{s}" for m in ("logreg", "random_forest", "hist_gb") for s in ("", "+eng")
    }
    for v in res["runs"].values():
        assert 0.0 <= v["test"]["pr_auc"] <= 1.0
