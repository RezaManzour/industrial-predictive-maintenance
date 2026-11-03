"""Exploratory analysis: writes a numeric summary and figures. Run: python -m ipm.eda"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ipm.data import LEAKY_COLUMNS, NUMERIC_FEATURES, TARGET, load_raw

ROOT = Path(__file__).resolve().parents[2]


def summarize(df: pd.DataFrame) -> dict:
    flags = df[LEAKY_COLUMNS]
    any_flag = flags.max(axis=1)
    out = {
        "rows": len(df),
        "failure_rate": float(df[TARGET].mean()),
        "failures": int(df[TARGET].sum()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "failure_rate_by_type": df.groupby("Type")[TARGET].mean().round(4).to_dict(),
        "failure_mode_counts": {c: int(df[c].sum()) for c in LEAKY_COLUMNS},
        # rows where target and flags disagree (known quirk of this dataset)
        "failure_without_any_mode_flag": int(((df[TARGET] == 1) & (any_flag == 0)).sum()),
        "mode_flag_without_failure": int(((df[TARGET] == 0) & (any_flag == 1)).sum()),
        "correlation_with_target": df[NUMERIC_FEATURES + [TARGET]]
        .corr()[TARGET].drop(TARGET).round(4).to_dict(),
    }
    return out


def make_figures(df: pd.DataFrame, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    files = []

    fig, ax = plt.subplots(figsize=(4, 3))
    df[TARGET].value_counts().sort_index().plot.bar(ax=ax, color=["#4c78a8", "#e45756"])
    ax.set_title("Class balance")
    ax.set_xticklabels(["no failure", "failure"], rotation=0)
    fig.tight_layout()
    fig.savefig(outdir / "class_balance.png", dpi=150)
    plt.close(fig)
    files.append("class_balance.png")

    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(3 * len(NUMERIC_FEATURES), 3))
    for ax, col in zip(axes, NUMERIC_FEATURES, strict=True):
        for label, g in df.groupby(TARGET):
            ax.hist(g[col], bins=30, alpha=0.6, density=True, label=str(label))
        ax.set_title(col, fontsize=8)
    axes[0].legend(title="failure")
    fig.tight_layout()
    fig.savefig(outdir / "feature_distributions.png", dpi=150)
    plt.close(fig)
    files.append("feature_distributions.png")

    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(outdir / "correlation.png", dpi=150)
    plt.close(fig)
    files.append("correlation.png")
    return files


def main(path: Path | None = None) -> None:
    path = path or ROOT / "data" / "raw" / "ai4i2020.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    df = load_raw(path)
    summary = summarize(df)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(df, ROOT / "reports" / "figures")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
