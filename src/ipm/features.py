"""Physically motivated features for the AI4I 2020 machine data."""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED = ["power_w", "temp_diff_k", "wear_x_torque"]


def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    """Add mechanical power, process-air temperature gap and wear-torque product."""
    X = X.copy()
    omega = X["Rotational speed [rpm]"] * 2 * np.pi / 60.0
    X["power_w"] = X["Torque [Nm]"] * omega
    X["temp_diff_k"] = X["Process temperature [K]"] - X["Air temperature [K]"]
    X["wear_x_torque"] = X["Tool wear [min]"] * X["Torque [Nm]"]
    return X
