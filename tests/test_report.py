from ipm.report import fig_cost, fig_models, fig_rl


def stat(m, s=0.01):
    return {"pr_auc": {"mean": m, "std": s}, "f1": {"mean": m, "std": s}}


def test_figures_are_written(tmp_path):
    ms = {"models": {k: stat(0.5) for k in ("logreg", "hist_gb+eng")}}
    deep = {"models": {"mlp": stat(0.4)}}
    fig_models(ms, deep, tmp_path / "a.png")
    cost = {"cost_per_machine": {"5": {p: {"mean": 0.1, "std": 0.01} for p in
            ("act_on_nobody", "act_on_everybody", "f1_threshold", "cost_threshold")}}}
    fig_cost(cost, tmp_path / "b.png")
    rl = {"bin_minutes": 10, "hazard_train": [0.01, 0.02, 0.2], "hazard_heldout": [0.02, 0.02, 0.1],
          "cases": {"c_fail=100,c_replace=10": {"value_iteration_optimal": {"policy": [0, 0, 1]}}}}
    fig_rl(rl, tmp_path / "c.png")
    assert all((tmp_path / n).stat().st_size > 1000 for n in ("a.png", "b.png", "c.png"))
