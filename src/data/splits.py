"""Leakage-safe train/val/test split for AI4I 2020.

AI4I has no machine or time column: each row is an independent synthetic
observation and ``UDI`` is just a sequential row id. A stratified row split is
therefore group-safe, as long as no ``UDI`` appears twice. We check that
explicitly instead of assuming it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = "Machine failure"
ID_COL = "UDI"


def stratified_split(
    df: pd.DataFrame,
    target: str = TARGET,
    id_col: str = ID_COL,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if id_col in df.columns and df[id_col].duplicated().any():
        raise ValueError(f"Duplicate {id_col!r} values: rows are not independent.")
    holdout = val_size + test_size
    train_df, temp_df = train_test_split(
        df, test_size=holdout, stratify=df[target], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / holdout,
        stratify=temp_df[target],
        random_state=seed,
    )
    assert_disjoint(train_df, val_df, test_df)
    return train_df, val_df, test_df


def assert_disjoint(*parts: pd.DataFrame) -> None:
    seen: set = set()
    for part in parts:
        overlap = seen.intersection(part.index)
        if overlap:
            raise AssertionError(f"Leakage: {len(overlap)} rows shared between splits")
        seen.update(part.index)
