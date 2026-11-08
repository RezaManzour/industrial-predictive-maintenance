import numpy as np

from ipm.diagnostics import calibration_bins, compute, plot
from tests.test_data import make_df


def test_diagnostics_compute_and_plot(tmp_path):
    res = compute(make_df(n=1500, rate=0.05), n_repeats=2)
    assert 0 <= res["pr_auc"] <= 1
    assert len(res["permutation_importance"]) == 9  # 5 sensors + type + 3 engineered
    plot(res, tmp_path / "d.png")
    assert (tmp_path / "d.png").stat().st_size > 1000


def test_calibration_bins_counts_and_values():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.005, 0.02, 0.3, 0.6, 0.9])
    bins = calibration_bins(y, s)
    assert sum(b["n"] for b in bins) == 5
    top = bins[-1]
    assert top["n"] == 2 and top["fraction_positive"] == 1.0
