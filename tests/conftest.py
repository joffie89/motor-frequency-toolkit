import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """Small fake portfolio with the same columns as freMTPL2freq and a known risk structure."""
    rng = np.random.default_rng(0)
    n = 6000
    df = pd.DataFrame({
        "VehBrand": rng.choice(["B1", "B2", "B12"], n),
        "VehGas": rng.choice(["Regular", "Diesel"], n),
        "Area": rng.choice(list("ABCDEF"), n),
        "Region": rng.choice([f"R{i}" for i in range(12)], n),
        "VehAge": rng.integers(0, 20, n),
        "DrivAge": rng.integers(18, 85, n),
        "VehPower": rng.integers(4, 15, n),
        "Density": rng.integers(1, 20000, n),
        "BonusMalus": rng.integers(50, 150, n),
        "Exposure": rng.uniform(0.05, 1.0, n),
    })
    rate = 0.08 * np.exp(0.01 * (df["BonusMalus"] - 50))
    df["ClaimNb"] = rng.poisson(rate * df["Exposure"]).astype(float)
    return df


@pytest.fixture
def cfg(tmp_path) -> dict:
    return {
        "data": {"local_path": None, "max_claims": 4, "max_exposure": 1.0},
        "simulation": {"seed": 1, "n_periods": 4, "test_period": 4,
                       "drift": {"enabled": True, "feature": "VehBrand", "level": "B12", "period": 4, "uplift": 0.6}},
        "features": {"categorical": ["VehBrand", "VehGas", "Area"], "binned": ["VehAge", "DrivAge", "VehPower"],
                     "log": ["Density"], "passthrough": ["BonusMalus"], "n_bins": 5,
                     "area_grouping": {"column": "Region", "n_groups": 3, "credibility_k": 50}},
        "models": {"glm": {"alpha": 1e-4, "max_iter": 200},
                   "gbm": {"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 15}},
        "monitoring": {"segment": "VehBrand", "oe_threshold": 1.2, "min_exposure": 10},
        "output_dir": str(tmp_path / "reports"),
    }
