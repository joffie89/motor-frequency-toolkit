"""Evaluation: splits and pricing-relevant metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance


def time_split(df: pd.DataFrame, test_period: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier periods, test on a later one: mirrors how the model is used."""
    return df[df["Period"] < test_period], df[df["Period"] == test_period]


def random_split(df: pd.DataFrame, test_frac: float = 0.25, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A random split mixes periods, so it can hide changes over time."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_frac
    return df[~mask], df[mask]


def observed_expected(claims: np.ndarray, pred_freq: np.ndarray, exposure: np.ndarray) -> float:
    """O/E: actual claims divided by predicted claims. 1.0 = well calibrated; >1 = under-predicting."""
    return float(np.sum(claims) / np.sum(pred_freq * exposure))


def lift(claims: np.ndarray, pred_freq: np.ndarray, exposure: np.ndarray, n_groups: int = 10) -> float:
    """Observed frequency in the highest-risk group divided by the lowest-risk group."""
    t = pd.DataFrame({"y": claims, "w": exposure, "p": pred_freq})
    t["g"] = pd.qcut(t["p"].rank(method="first"), n_groups, labels=False)
    f = t.groupby("g").apply(lambda d: d["y"].sum() / d["w"].sum(), include_groups=False)
    return float(f.iloc[-1] / f.iloc[0])


def lift_table(claims, pred_freq, exposure, n_groups: int = 10) -> pd.DataFrame:
    """Observed vs predicted frequency by predicted-risk decile (for charts)."""
    t = pd.DataFrame({"y": claims, "w": exposure, "p": pred_freq})
    t["decile"] = pd.qcut(t["p"].rank(method="first"), n_groups, labels=False) + 1
    return t.groupby("decile").apply(
        lambda d: pd.Series({"observed": d["y"].sum() / d["w"].sum(),
                             "predicted": (d["p"] * d["w"]).sum() / d["w"].sum()}),
        include_groups=False,
    )


def metrics(df: pd.DataFrame, pred_freq: np.ndarray) -> dict[str, float]:
    """Poisson deviance (lower = better fit), O/E (calibration) and lift (risk separation)."""
    y, w, c = df["Freq"].to_numpy(), df["Exposure"].to_numpy(), df["ClaimNb"].to_numpy()
    return {
        "poisson_deviance": float(mean_poisson_deviance(y, pred_freq, sample_weight=w)),
        "oe": observed_expected(c, pred_freq, w),
        "lift_top_bottom_decile": lift(c, pred_freq, w),
    }
