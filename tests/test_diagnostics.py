from ipm.diagnostics import compute, plot
from tests.test_data import make_df


def test_diagnostics_compute_and_plot(tmp_path):
    res = compute(make_df(n=1500, rate=0.05), n_repeats=2)
    assert 0 <= res["pr_auc"] <= 1
    assert len(res["permutation_importance"]) == 9  # 5 sensors + type + 3 engineered
    plot(res, tmp_path / "d.png")
    assert (tmp_path / "d.png").stat().st_size > 1000
