import numpy as np
import pandas as pd

from src.data.splits import stratified_split


def _make_df():
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Torque [Nm]": rng.normal(40, 10, n),
            "Machine failure": (rng.random(n) < 0.03).astype(int),
        }
    )


def test_no_leakage():
    df = _make_df()
    train, val, test = stratified_split(df)
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_sizes():
    df = _make_df()
    train, val, test = stratified_split(df)
    assert abs(len(train) - 700) <= 10
    assert abs(len(val) - 150) <= 10
    assert abs(len(test) - 150) <= 10


def test_stratification():
    df = _make_df()
    overall = df["Machine failure"].mean()
    for part in stratified_split(df):
        assert abs(part["Machine failure"].mean() - overall) <= 0.01
