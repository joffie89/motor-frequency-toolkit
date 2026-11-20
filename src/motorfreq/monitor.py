"""Monitoring: find segments where actual claims drift away from predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def segment_oe(df: pd.DataFrame, pred_freq: np.ndarray, segment: str, min_exposure: float = 1000,
               threshold: float = 1.2) -> pd.DataFrame:
    """O/E by segment for the latest period, flagging segments above the threshold."""
    t = df[[segment, "ClaimNb", "Exposure"]].copy()
    t["expected"] = pred_freq * t["Exposure"]
    g = t.groupby(segment).agg(exposure=("Exposure", "sum"), observed=("ClaimNb", "sum"),
                               expected=("expected", "sum"))
    g = g[g["exposure"] >= min_exposure]
    g["oe"] = g["observed"] / g["expected"]
    g["alert"] = g["oe"] > threshold
    return g.sort_values("oe", ascending=False)
