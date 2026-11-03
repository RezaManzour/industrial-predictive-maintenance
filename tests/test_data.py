import numpy as np
import pandas as pd

from ipm.data import LEAKY_COLUMNS, TARGET, split_xy, train_val_test_split


def make_df(n=2000, rate=0.035, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(n),
            "Product ID": [f"M{i}" for i in range(n)],
            "Type": rng.choice(list("LMH"), n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": rng.normal(40, 10, n),
            "Tool wear [min]": rng.integers(0, 250, n),
            TARGET: y,
            **{c: np.zeros(n, dtype=int) for c in LEAKY_COLUMNS},
        }
    )


def test_no_leaky_columns_in_features():
    X, _ = split_xy(make_df())
    assert not set(LEAKY_COLUMNS) & set(X.columns)
    assert TARGET not in X.columns and "UDI" not in X.columns


def test_splits_disjoint_and_complete():
    df = make_df()
    tr, va, te = train_val_test_split(df)
    ids = [set(s["UDI"]) for s in (tr, va, te)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(s) for s in (tr, va, te)) == len(df)


def test_splits_keep_class_ratio_and_are_reproducible():
    df = make_df()
    tr, va, te = train_val_test_split(df)
    for s in (tr, va, te):
        assert abs(s[TARGET].mean() - df[TARGET].mean()) < 0.01
    tr2, _, _ = train_val_test_split(df)
    assert list(tr["UDI"]) == list(tr2["UDI"])
