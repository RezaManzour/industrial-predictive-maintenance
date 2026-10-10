"""01_eda.py - same analysis as 01_eda.ipynb. Run from repo root: python notebooks/01_eda.py"""

# Cost intuition: see the markdown cell in the notebook (missed failure >> false alarm -> PR-AUC, recall@precision).
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 42
np.random.seed(SEED)
ROOT = (
    Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
)  # works from repo root or notebooks/
DATA = ROOT / "data/raw/ai4i2020.csv"
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

TARGET = "Machine failure"
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUM = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("saved", FIG / name)


def load_and_sanity():
    global df, counts, r
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print("shape:", df.shape)
    print(df.dtypes, "\n")
    print("nulls per column:\n", df.isna().sum(), "\n")
    counts = df[TARGET].value_counts()
    print(
        pd.DataFrame({"count": counts, "pct": (100 * counts / len(df)).round(2)}), "\n"
    )
    print("failure-mode counts:\n", df[MODES].sum())
    print(
        "failures with no mode flag:",
        int(((df[TARGET] == 1) & (df[MODES].sum(axis=1) == 0)).sum()),
    )


def class_balance():
    global df, counts, r
    # ~3% positives: accuracy is meaningless (all-zero model ≈ 97%) -> use PR-AUC and tune the decision threshold.
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["No failure (0)", "Failure (1)"],
        counts.sort_index().values,
        color=["#4C72B0", "#C44E52"],
    )
    for b, n in zip(bars, counts.sort_index().values):
        ax.annotate(
            f"{n} ({100 * n / len(df):.1f}%)",
            (b.get_x() + b.get_width() / 2, n),
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("rows")
    ax.set_title("Class balance: Machine failure")
    save(fig, "01_class_balance.png")


def feature_dists():
    global df, counts, r
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, NUM):
        sns.kdeplot(
            data=df, x=col, hue=TARGET, common_norm=False, fill=True, ax=ax
        )  # per-class density, not counts
        ax.set_title(col)
    axes.flat[-1].axis("off")
    save(fig, "02_feature_dists.png")


def temp_correlation():
    global df, counts, r
    r = df["Air temperature [K]"].corr(df["Process temperature [K]"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df.sort_values(TARGET),
        x="Air temperature [K]",
        y="Process temperature [K]",
        hue=TARGET,
        s=8,
        alpha=0.6,
        ax=ax,
    )
    ax.text(
        0.03,
        0.95,
        f"Pearson r = {r:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
    )
    ax.set_title("Air vs process temperature")
    save(fig, "03_temp_correlation.png")
    # Highly correlated -> in Phase 2 compare dropping one vs a derived delta-T = process - air feature.


def toolwear_drift():
    global df, counts, r
    # Caveat: AI4I is synthetic and UDI is a row index, not a timestamp. Treat this as a sanity view, not a time series.
    d = df.sort_values("UDI")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        d["UDI"],
        d["Tool wear [min]"],
        c=d[TARGET].map({0: "#4C72B0", 1: "#C44E52"}),
        s=4,
        alpha=0.4,
    )
    ax.plot(
        d["UDI"],
        d["Tool wear [min]"].rolling(200).mean(),
        color="black",
        lw=1.5,
        label="rolling mean (200)",
    )
    ax.set_xlabel("UDI (row order)")
    ax.set_ylabel("Tool wear [min]")
    ax.legend(loc="upper right")
    ax.set_title("Tool wear over row order (red = failure)")
    save(fig, "04_toolwear_drift.png")
    # Failures concentrate at high tool wear -> candidate engineered features (wear x torque = "strain").


def failure_modes():
    global df, counts, r
    modes = pd.DataFrame(
        {f"Machine failure = {k}": df[df[TARGET] == k][MODES].sum() for k in (0, 1)}
    )
    print(modes)
    fig, ax = plt.subplots(figsize=(8, 5))
    modes.plot(kind="bar", stacked=True, color=["#4C72B0", "#C44E52"], ax=ax)
    ax.set_ylabel("rows with mode flag = 1")
    ax.set_title("Failure modes split by Machine failure label")
    ax.tick_params(axis="x", rotation=0)
    save(fig, "05_failure_modes.png")
    # Note: RNF (random) is mostly flagged on rows labelled 0 -> label noise we cannot predict from sensors.


def correlation_heatmap():
    global df, counts, r
    corr = df.drop(columns=["UDI"]).select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=np.triu(np.ones_like(corr, dtype=bool)),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Correlation matrix (lower triangle)")
    save(fig, "06_correlation_heatmap.png")


def summary():
    global df, counts, r
    g = df.groupby(TARGET)
    summary = pd.DataFrame(
        {
            "no failure (0)": [
                g["Tool wear [min]"].mean()[0],
                g["Torque [Nm]"].mean()[0],
            ],
            "failure (1)": [g["Tool wear [min]"].mean()[1], g["Torque [Nm]"].mean()[1]],
        },
        index=["mean tool wear [min]", "mean torque [Nm]"],
    ).round(2)
    print(summary, "\n")
    print(f"imbalance ratio (neg:pos) = {counts[0] / counts[1]:.1f} : 1")
    print(f"air/process temperature Pearson r = {r:.3f}")


if __name__ == "__main__":
    load_and_sanity()
    class_balance()
    feature_dists()
    temp_correlation()
    toolwear_drift()
    failure_modes()
    correlation_heatmap()
    summary()
