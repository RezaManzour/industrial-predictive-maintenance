"""Framework-free helpers for the deep models. Fit on train only, testable offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPE_LEVELS = ["L", "M", "H"]


def encode_type(types: pd.Series) -> np.ndarray:
    """Map Type (L/M/H) to ids 0/1/2. Unknown values raise, never map silently."""
    unknown = set(types.unique()) - set(TYPE_LEVELS)
    if unknown:
        raise ValueError(f"Unknown Type values: {sorted(unknown)}")
    return types.map({lvl: i for i, lvl in enumerate(TYPE_LEVELS)}).to_numpy()


class Standardizer:
    """Mean/std fit on the training matrix only, applied unchanged to val and test."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.scale_).astype(np.float32)


class EarlyStopping:
    """Stop when the score (higher is better) stops improving for `patience` steps."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, score: float, epoch: int) -> bool:
        """Record one epoch. Returns True when training should stop."""
        if score > self.best + self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
