import numpy as np
import pandas as pd
import pytest

from src.data.splits import stratified_split


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """1000 independent rows, ~3% positives, AI4I-like column names."""
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Torque [Nm]": rng.normal(40, 10, n),
            "Machine failure": (rng.random(n) < 0.03).astype(int),
        }
    )


def test_no_leakage(synthetic_df):
    train, val, test = stratified_split(synthetic_df)
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)
    assert set(train["UDI"]).isdisjoint(set(val["UDI"]) | set(test["UDI"]))
    assert len(train) + len(val) + len(test) == len(synthetic_df)


def test_split_sizes(synthetic_df):
    train, val, test = stratified_split(synthetic_df)
    assert abs(len(train) - 700) <= 10
    assert abs(len(val) - 150) <= 10
    assert abs(len(test) - 150) <= 10


def test_stratification_preserves_positive_rate(synthetic_df):
    overall = synthetic_df["Machine failure"].mean()
    for part in stratified_split(synthetic_df):
        assert abs(part["Machine failure"].mean() - overall) <= 0.01
