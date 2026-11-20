"""Data-quality checks run before any modelling."""
from __future__ import annotations

import pandas as pd

REQUIRED = ["ClaimNb", "Exposure"]


class DataQualityError(ValueError):
    """Raised when the input data fails a quality check."""


def check(df: pd.DataFrame, required_features: list[str] | None = None) -> list[str]:
    """Return a list of problems found (empty list = data passes)."""
    problems: list[str] = []
    cols = REQUIRED + list(required_features or [])
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        return [f"missing columns: {missing_cols}"]
    nulls = df[cols].isna().sum()
    if nulls.any():
        problems.append(f"null values: {nulls[nulls > 0].to_dict()}")
    if (df["Exposure"] <= 0).any():
        problems.append("exposure must be > 0")
    if (df["Exposure"] > 1).any():
        problems.append("exposure must be <= 1 year")
    if (df["ClaimNb"] < 0).any():
        problems.append("claim counts must be >= 0")
    return problems


def validate(df: pd.DataFrame, required_features: list[str] | None = None) -> pd.DataFrame:
    """Raise DataQualityError if any check fails; otherwise return the data unchanged."""
    problems = check(df, required_features)
    if problems:
        raise DataQualityError("; ".join(problems))
    return df
