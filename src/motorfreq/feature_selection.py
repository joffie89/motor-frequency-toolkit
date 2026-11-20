"""Feature selection for frequency models: evidence for which rating factors to keep.

Three views, each answering a different question:
- one-way analysis:   does frequency vary across the feature's values? (exposure-adjusted)
- drop-one deviance:  does the model predict a later period worse without it?
- stability:          is the feature's effect consistent over time? (unstable = drift risk)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models
from .evaluate import metrics


def _bucket(series: pd.Series, n_bins: int = 10) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > n_bins:
        return pd.qcut(series, n_bins, duplicates="drop").astype(str)
    return series.astype(str)


def one_way(df: pd.DataFrame, feature: str, n_bins: int = 10) -> pd.DataFrame:
    """Exposure-weighted frequency and relativity (vs portfolio average) for each feature value."""
    overall = df["ClaimNb"].sum() / df["Exposure"].sum()
    g = df.assign(_v=_bucket(df[feature], n_bins)).groupby("_v").agg(
        exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"))
    g["frequency"] = g["claims"] / g["exposure"]
    g["relativity"] = g["frequency"] / overall
    return g.rename_axis(feature)


def spread(df: pd.DataFrame, feature: str, min_exposure: float = 1000) -> float:
    """Ratio of highest to lowest relativity among credible values (how strongly frequency varies)."""
    t = one_way(df, feature)
    t = t[t["exposure"] >= min_exposure]
    return float(t["relativity"].max() / t["relativity"].min())


def stability(df: pd.DataFrame, feature: str, min_exposure: float = 1000) -> float:
    """Largest change in a value's relativity between the first and last period (as a ratio, >= 1)."""
    periods = sorted(df["Period"].unique())
    first, last = (one_way(df[df["Period"] == p], feature) for p in (periods[0], periods[-1]))
    joined = first.join(last, lsuffix="_first", rsuffix="_last")
    joined = joined[(joined["exposure_first"] >= min_exposure) & (joined["exposure_last"] >= min_exposure)]
    ratio = joined["relativity_last"] / joined["relativity_first"]
    return float(np.exp(np.abs(np.log(ratio)).max()))


def drop_one_deviance(train: pd.DataFrame, valid: pd.DataFrame, spec: models.FeatureSpec,
                      glm_params: dict) -> pd.DataFrame:
    """Refit the GLM without each feature; report the change in validation deviance.

    Positive = removing the feature makes predictions worse (the feature helps).
    """
    base_model = models.fit(models.build_glm(spec, **glm_params), train, spec)
    base = metrics(valid, models.predict(base_model, valid, spec))["poisson_deviance"]
    rows = []
    for f in spec.columns:
        s = spec.without(f)
        m = models.fit(models.build_glm(s, **glm_params), train, s)
        dev = metrics(valid, models.predict(m, valid, s))["poisson_deviance"]
        rows.append({"feature": f, "deviance_increase_when_dropped": dev - base})
    return pd.DataFrame(rows).set_index("feature")


def report(df: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame, spec: models.FeatureSpec,
           glm_params: dict, stability_threshold: float = 1.3) -> pd.DataFrame:
    """One table combining all three views, with a simple recommendation per feature."""
    table = drop_one_deviance(train, valid, spec, glm_params)
    table["one_way_spread"] = [spread(df, f) for f in table.index]
    table["stability_ratio"] = [stability(df, f) for f in table.index]

    def recommend(r):
        if r["stability_ratio"] > stability_threshold:
            return "review: unstable over time"
        if r["deviance_increase_when_dropped"] > 0:
            return "keep"
        return "candidate to drop or merge"

    table["recommendation"] = table.apply(recommend, axis=1)
    return table.sort_values("deviance_increase_when_dropped", ascending=False)
