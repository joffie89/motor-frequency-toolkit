"""Feature preparation: simulated periods, drift injection and area grouping."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_periods(df: pd.DataFrame, n_periods: int, seed: int) -> pd.DataFrame:
    """Assign each policy to a SIMULATED period 1..n (the public data has no dates).

    Illustrative only: lets us demonstrate time-based validation and monitoring.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["Period"] = rng.integers(1, n_periods + 1, len(out))
    return out


def inject_drift(df: pd.DataFrame, feature: str, level: str, period: int, uplift: float, seed: int) -> pd.DataFrame:
    """Add extra claims to one segment in one period to mimic an emerging risk (e.g. a new theft method)."""
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    mask = (out[feature] == level) & (out["Period"] == period)
    base_rate = out.loc[mask, "ClaimNb"].sum() / out.loc[mask, "Exposure"].sum()
    extra = rng.poisson(uplift * base_rate * out.loc[mask, "Exposure"])
    out.loc[mask, "ClaimNb"] = out.loc[mask, "ClaimNb"] + extra
    out["Freq"] = out["ClaimNb"] / out["Exposure"]
    return out


def fit_area_groups(train: pd.DataFrame, column: str, n_groups: int, credibility_k: float) -> dict[str, int]:
    """Group a high-cardinality location column into risk bands.

    Each level's frequency is shrunk towards the overall mean in proportion to how much
    exposure it has (credibility weighting), so sparse areas are not over-trusted.
    Levels are then ranked and cut into n_groups bands (0 = lowest risk).
    """
    overall = train["ClaimNb"].sum() / train["Exposure"].sum()
    g = train.groupby(column).agg(claims=("ClaimNb", "sum"), exposure=("Exposure", "sum"))
    z = g["exposure"] / (g["exposure"] + credibility_k)
    smoothed = z * (g["claims"] / g["exposure"]) + (1 - z) * overall
    bands = pd.qcut(smoothed.rank(method="first"), n_groups, labels=False)
    return {str(k): int(v) for k, v in bands.items()}


def apply_area_groups(df: pd.DataFrame, column: str, mapping: dict[str, int], name: str = "AreaGroup") -> pd.DataFrame:
    """Map levels to their band; unseen levels go to the middle band."""
    out = df.copy()
    middle = int(np.median(list(mapping.values())))
    out[name] = out[column].astype(str).map(mapping).fillna(middle).astype(int).astype(str)
    return out
