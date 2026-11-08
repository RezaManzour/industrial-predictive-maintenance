"""Build the final figures from the saved results. Run: python -m ipm.report"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["logreg", "random_forest", "hist_gb", "logreg+eng", "random_forest+eng", "hist_gb+eng", "mlp", "mlp+eng"]
LABELS = {
    "logreg": "LogReg", "random_forest": "RandForest", "hist_gb": "GradBoost",
    "logreg+eng": "LogReg+eng", "random_forest+eng": "RandForest+eng", "hist_gb+eng": "GradBoost+eng",
    "mlp": "MLP", "mlp+eng": "MLP+eng",
}


def model_means(multiseed: dict, deep: dict) -> dict[str, tuple[float, float]]:
    out = {k: (v["pr_auc"]["mean"], v["pr_auc"]["std"]) for k, v in multiseed["models"].items()}
    out.update({k: (v["pr_auc"]["mean"], v["pr_auc"]["std"]) for k, v in deep["models"].items()})
    return out


def fig_models(multiseed: dict, deep: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = model_means(multiseed, deep)
    names = [k for k in ORDER if k in m]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["#4c78a8" if "+eng" not in k else "#f58518" for k in names]
    ax.bar(range(len(names)), [m[k][0] for k in names], yerr=[m[k][1] for k in names], color=colors, capsize=3)
    ax.set_xticks(range(len(names)), [LABELS[k] for k in names], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Test PR-AUC (mean ± std, 5 seeds)")
    ax.set_title("Model comparison (orange = with engineered features)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_cost(cost: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cost["cost_per_machine"]
    ratios = list(data)
    pols = ["act_on_nobody", "act_on_everybody", "f1_threshold", "cost_threshold"]
    names = ["Act on nobody", "Act on everybody", "F1-optimal threshold", "Cost-optimal threshold"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.2
    for i, (p, n) in enumerate(zip(pols, names, strict=True)):
        ax.bar([j + (i - 1.5) * w for j in range(len(ratios))], [data[r][p]["mean"] for r in ratios],
               w, yerr=[data[r][p]["std"] for r in ratios], label=n, capsize=2)
    ax.set_xticks(range(len(ratios)), [f"c_fn/c_fp = {r}" for r in ratios])
    ax.set_ylabel("Cost per machine (test)")
    ax.set_yscale("log")
    ax.set_ylim(top=8)
    ax.legend(fontsize=7, ncol=2, loc="upper center")
    ax.set_title("Cost of decision rules (gradient boosting + engineered features)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_rl(rl: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = range(len(rl["hazard_train"]))
    x = [b * rl["bin_minutes"] for b in bins]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, rl["hazard_train"], marker="o", ms=3, label="Failure rate per cycle (train)")
    ax.plot(x, rl["hazard_heldout"], marker="o", ms=3, label="Failure rate per cycle (validation+test)")
    for (case, pols), ls in zip(rl["cases"].items(), (":", "--", "-."), strict=False):
        pol = pols["value_iteration_optimal"]["policy"]
        first = next((i for i, a in enumerate(pol) if a == 1), None)
        if first is not None:
            ax.axvline(first * rl["bin_minutes"], color="gray", ls=ls, lw=1,
                       label=f"Optimal replace from {first * rl['bin_minutes']} min ({case.split(',')[0]})")
    ax.set_xlabel("Tool wear (min)")
    ax.set_ylabel("Failure probability")
    ax.legend(fontsize=6)
    ax.set_title("Estimated hazard and exact optimal replacement point", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(root: Path | None = None) -> None:
    root = root or ROOT
    res = root / "results"
    figs = root / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    load = lambda n: json.loads((res / n).read_text())
    fig_models(load("metrics_multiseed.json"), load("metrics_deep.json"), figs / "model_comparison.png")
    fig_cost(load("cost_thresholds.json"), figs / "cost_thresholds.png")
    fig_rl(load("rl_maintenance.json"), figs / "rl_hazard_policy.png")
    print("wrote model_comparison.png, cost_thresholds.png, rl_hazard_policy.png")


if __name__ == "__main__":
    main()
