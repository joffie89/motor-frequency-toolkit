"""Claim-frequency models: Poisson GLM (baseline) and Poisson GBM (modern ML).

Both predict claims per vehicle-year and are trained with exposure as the sample weight,
which is equivalent to a Poisson model on claim counts with a log(exposure) offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)


@dataclass
class FeatureSpec:
    categorical: list[str] = field(default_factory=list)
    binned: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)
    n_bins: int = 10

    @property
    def columns(self) -> list[str]:
        return self.categorical + self.binned + self.log + self.passthrough

    def without(self, feature: str) -> "FeatureSpec":
        """A copy of this spec with one feature removed (used in feature selection)."""
        return FeatureSpec(
            [c for c in self.categorical if c != feature],
            [c for c in self.binned if c != feature],
            [c for c in self.log if c != feature],
            [c for c in self.passthrough if c != feature],
            self.n_bins,
        )


def _glm_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    steps = []
    if spec.categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), spec.categorical))
    if spec.binned:
        steps.append((
            "bin",
            KBinsDiscretizer(n_bins=spec.n_bins, encode="onehot", strategy="quantile",
                             subsample=200_000, quantile_method="averaged_inverted_cdf", random_state=0),
            spec.binned,
        ))
    if spec.log:
        steps.append(("log", Pipeline([("log1p", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]), spec.log))
    if spec.passthrough:
        steps.append(("num", StandardScaler(), spec.passthrough))
    return ColumnTransformer(steps)


def build_glm(spec: FeatureSpec, alpha: float = 1e-4, max_iter: int = 300) -> Pipeline:
    return Pipeline([("pre", _glm_preprocessor(spec)), ("model", PoissonRegressor(alpha=alpha, max_iter=max_iter))])


def build_gbm(spec: FeatureSpec, learning_rate: float = 0.1, max_iter: int = 200,
              max_leaf_nodes: int = 64) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), spec.categorical)],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=learning_rate, max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes, categorical_features=list(range(len(spec.categorical))), random_state=0,
    )
    return Pipeline([("pre", pre), ("model", model)])


def fit(model: Pipeline, df: pd.DataFrame, spec: FeatureSpec) -> Pipeline:
    """Fit on claim frequency, weighted by exposure."""
    model.fit(df[spec.columns], df["Freq"], model__sample_weight=df["Exposure"])
    return model


def predict(model: Pipeline, df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """Predicted claims per vehicle-year."""
    return model.predict(df[spec.columns])
