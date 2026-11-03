"""Loading and leakage-safe splitting of the AI4I 2020 dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Machine failure"
# Failure-mode flags are components of the target, so using them as features leaks the label.
LEAKY_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
SEED = 42


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target with ids and failure-mode flags removed."""
    X = df.drop(columns=[TARGET, *LEAKY_COLUMNS, *ID_COLUMNS], errors="ignore")
    return X, df[TARGET]


def train_val_test_split(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split on the target, fixed seed, no row overlap."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[TARGET], random_state=seed
    )
    return train, val, test
